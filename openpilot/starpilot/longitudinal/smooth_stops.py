"""Smooth Stops: let off the brake through the last metres of every stop so the wheels stop under a whisper of braking."""
from dataclasses import dataclass
import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.starpilot.longitudinal.smooth_stops_preferences import supported

# Without it, a stop ends in LongControl's stopping state: once the plan's stop bit rises (on HKG cars at walking pace, about a
# second before the wheels stop) LongControl no longer follows the plan and ramps the request toward CP.stopAccel, so braking rises
# into the stop and the body rocks forward and back as the wheels stop.
#
# Smooth Stops shapes only the landing, for every stop the planner makes: behind a lead, the model's in Experimental mode, cruise's,
# and Enhanced Stops' approach and committed stop. A landing starts below LANDING_SPEED when the plan means to stop and is braking,
# or on the frame an Enhanced Stops commitment raises its stop bit. It only ever eases what StarPilot itself asks: its plan, or,
# once StarPilot's own stop bit has stayed up ANCHOR_TIME, the stopping ramp LongControl would be running. The eased request stays
# inside a corridor that falls with speed to one small "kiss" of braking at walking pace; its edges are requested one actuator delay
# ahead of the car. Braking beyond what StarPilot asks comes only from the anti-stall press, while the car is delivering less than
# StarPilot asks (creep, a weak brake, a car not slowing). StarPilot's stop bit is held back until the wheels have nearly stopped, so
# LongControl's stopping ramp and the car's standstill hold start at rest; at rest the plan passes and the bit stays up until the
# plan asks to move.
#
# The rest point is bounded: an Enhanced Stops commitment does not rest past its own stop point (the tracked line moved by its Stop
# Distance Adjustment) unless StarPilot's own stop already would, in which case the overrun hand-back returns the stop; any other
# stop rests at most LEEWAY further on than where StarPilot's stopping ramp would have ended it, and never past the nearest point
# the model's request has asked to stop at, both fixed to the road. Where the soft tail no longer fits,
# the landing eases less, up to letting StarPilot's own request through unchanged. Whatever stops the car LEAD_STANDING_GAP short of
# a radar lead passes as well. Anything a landing cannot absorb hands the stop back to StarPilot at once: the plan asking to move or
# no longer meaning to stop, a demand of URGENT_DECEL or more, a lead closer than the kiss can stop short of, no room left inside
# the bound, or a car that keeps not slowing (the press saturated). The plan's request then passes unchanged; if StarPilot's stop
# bit is already up, LongControl's stopping ramp continues from the request last sent at its usual rate. A launch hand-back keeps
# StarPilot's held bit unpublished while the plan asks to move and is under way, and a bit StarPilot has published is never
# withdrawn by a later landing.
## The corridor was measured on a Hyundai Palisade; smooth_stops_preferences.supported() limits it to cars like it.
LANDING_SPEED = 3.5          # m/s, the corridor acts below this; at the top the bound (2.75 m/s^2) is above any comfortable approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's planned speed at the end of its horizon below which a slowdown counts as a stop ...
GO_SPEED = 1.5               # m/s ... and above which, through a landing, the stop is over
KISS_SPEED = 0.40            # m/s, from here down the car rides the kiss alone
KISS_DECEL = 0.15            # m/s^2, the braking the wheels stop under; the stopping ramp and the car's hold take over at rest
CREEP_SPEED = 1.0            # m/s, the floor peaks at CREEP_DECEL here: enough to keep slowing against creep torque
CREEP_DECEL = 0.40           # m/s^2
CREEP_FADE_SPEED = 1.5       # m/s, and it is gone here: a queue rolling at 2 m/s is not held to a stop's floor
BOUND_BP = [KISS_SPEED, 0.9, LANDING_SPEED]    # the car's deceleration against its speed: at most 0.70 * v + 0.30 above
BOUND_V = [KISS_DECEL, 0.93, 2.75]             # 0.9 m/s, straight down to the kiss below
FLOOR_BP = [KISS_SPEED, CREEP_SPEED, CREEP_FADE_SPEED]
FLOOR_V = [KISS_DECEL, CREEP_DECEL, 0.0]
LEEWAY = 0.5                 # m, how much further on than StarPilot's own stop point a landing may come to rest
# m, the gap the landing always keeps to a radar lead, inside where StarPilot's own lead stops come to rest (the leeway keeps the rest
# point near StarPilot's own). A car the kiss cannot stop outside it after the actuator delay is handed back
LEAD_STANDING_GAP = 2.0
LEAD_ABSENCE_TIME = 0.35     # s a lead's requirement is kept after its reading drops out
STANDSTILL_SPEED = 0.10      # m/s, about the car's standstill flag (0.104 m/s of wheel speed on Hyundai)
URGENT_DECEL = 2.0           # m/s^2, a plan asking this much ...
URGENT_TIME = 0.1            # s ... for this long passes unchanged for as long as it asks; the landing then takes the tail back
URGENT_CLEAR = 0.2           # m/s^2 the plan must ease above URGENT_DECEL before the demand counts as over, so a plan near it
URGENT_RELEASE_JERK = 1.5    # m/s^3 does not chatter; braking then comes off no faster than this toward the corridor
CRAWL_BIT_SPEED = 0.05       # m/s, below walking pace the stop bit waits for this (or the standstill flag) ...
CRAWL_TIME = 1.0             # s ... or for the car to have crawled under STANDSTILL_SPEED this long
ANCHOR_TIME = 0.15           # s StarPilot's own stop bit must stay up before its stop counts: a blip of the bit is not a stop
# A far backstop for a car that is neither slowing nor running out of room; the leeway, not this, bounds a stop
HOLD_DISTANCE = 10.0         # m, the most the car travels with StarPilot's own stop bit held back ...
HOLD_TIME = 10.0             # s ... and for how long
# Anti-stall: while rolling, the car's shortfall against the floor presses both edges toward more braking and drains as the car
# slows again. It fades out with the floor and is not applied at rest. A press that reaches STALL_MAX hands the stop back
STALL_GAIN = 3.0             # 1/s
STALL_DEADBAND = 0.05        # m/s^2, about one standard deviation of aEgo while braking
STALL_MAX = 0.5              # m/s^2
# m/s^3, LongControl's stopping ramp on a car without a LongitudinalExtension, which every supported car is
STOPPING_DECEL_RATE = 1.0
MODEL_STOP_TIME = 4.0        # s, the model's request, held, bringing the car to rest within this is a stop
SEED_PLAN = True             # the MPC continues from its own unsoftened plan through a landing
E2E = log.LongitudinalPlan.LongitudinalPlanSource.e2e


def landing_bound(v_ego):
  # the most braking the car should carry at this speed
  return float(np.interp(v_ego, BOUND_BP, BOUND_V))


def landing_floor(v_ego):
  # the least braking the car should carry at this speed through a landing
  return float(np.interp(v_ego, FLOOR_BP, FLOOR_V))


# the distance a car riding the bound travels from each speed to rest
_SPEEDS = np.linspace(0.0, LANDING_SPEED, 141)
_RIDE = np.concatenate(([0.0], np.cumsum(np.diff(_SPEEDS) * 0.5 * (_SPEEDS[1:] / [landing_bound(v) for v in _SPEEDS[1:]] +
                                                                     _SPEEDS[:-1] / [landing_bound(v) for v in _SPEEDS[:-1]]))))


def corridor_distance(v_ego):
  return float(np.interp(v_ego, _SPEEDS, _RIDE))


KISS_DISTANCE = corridor_distance(KISS_SPEED)


def stop_within(speed, room):
  # the deceleration that sheds speed within room; with no room left any braking the plan asks is needed
  if speed <= 0.0:
    return 0.0
  return speed * speed / (2.0 * room) if room > 0.0 else math.inf


def held_stop_distance(v_ego, a_target):
  # where the planner's request, held, would bring the car to rest
  return v_ego * v_ego / (-2.0 * a_target) if a_target < -1e-3 else math.inf

def request_breakpoints(bp, values, action_t, steps=100):
  # each breakpoint below LANDING_SPEED moves up to the speed from which a car riding that edge reaches it after action_t
  h = action_t / steps
  out = []
  for v in bp:
    if v < LANDING_SPEED:
      for _ in range(steps):
        v += h * float(np.interp(v + 0.5 * h * float(np.interp(v, bp, values)), bp, values))
    out.append(v)
  return out


@dataclass(frozen=True)
class Lead:
  present: bool = False
  distance: float = math.inf
  speed: float = 0.0
  acceleration: float = 0.0

  @classmethod
  def from_radar(cls, lead):
    if lead is None or not lead.present:
      return cls()
    values = (float(lead.dRel), float(lead.vLeadK), float(lead.aLeadK))
    if not all(math.isfinite(v) for v in values):
      return cls()
    # a lead read at or behind the bumper is in contact, not gone
    return cls(True, max(values[0], 0.0), max(values[1], 0.0), float(np.clip(values[2], -10.0, 5.0)))


def owner_for(cp, dt):
  try:
    if cp.openpilotLongitudinalControl and not cp.pcmCruise and supported(cp):
      return SmoothStops(dt, cp.longitudinalActuatorDelay + DT_MDL, stop_accel=cp.stopAccel)
  except (AttributeError, TypeError, ValueError):
    pass
  return None


class SmoothStops:
  def __init__(self, dt, action_t, stop_accel=-2.0, stopping_decel_rate=STOPPING_DECEL_RATE):
    self.dt = dt
    self.action_t = action_t
    self.stop_accel = stop_accel
    self.stopping_decel_rate = stopping_decel_rate
    self.set_action_t(action_t)
    self.absence_frames = max(round(LEAD_ABSENCE_TIME / dt), 1)
    self.urgent_frames = max(round(URGENT_TIME / dt), 1)
    self.anchor_frames = max(round(ANCHOR_TIME / dt), 1)
    self._last_output = None
    self._last_bit = False
    self.reset()

  def reset(self):
    # what was last published survives a reset: a bit StarPilot published is never withdrawn by the next landing
    self.landing = False
    self.handed_back = False
    self.committed = False
    self.stop_target = None
    self.going = self.stopping = False
    self.urgent = self.urgent_release = False
    self.slow_time = 0.0
    self.launching = False
    self.stop_bit = False
    self.bit_frames = 0
    self.ramp_request = None
    self.exit_reason = ''
    self.integral = 0.0
    self.required_decel = 0.0
    self.overrun_decel = 0.0
    self.held_distance = 0.0
    self.held_time = 0.0
    self.travelled = 0.0
    self.ramp_anchor = None
    self.model_anchor = math.inf
    self.plan_target = None
    self._urgent = 0
    self._lead_requirements = [0.0, 0.0]
    self._lead_missing = [self.absence_frames] * 2

  @property
  def idle(self):
    return not (self.landing or self.handed_back or any(self._lead_requirements))

  def set_action_t(self, action_t):
    # the planner's effective actuator delay (with any developer override) sets how far ahead the corridor's edges are requested
    self.action_t = action_t
    self.bound_bp = request_breakpoints(BOUND_BP, BOUND_V, action_t)
    self.floor_bp = request_breakpoints(FLOOR_BP, FLOOR_V, action_t)

  def plan_seed(self, published):
    # the acceleration the MPC continues from: the plan the landing softened last frame, else what was published
    return published if self.plan_target is None or not SEED_PLAN else max(self.plan_target, ACCEL_MIN)

  def bound(self, v_ego):
    # the most braking requested at this speed
    return float(np.interp(v_ego, self.bound_bp, BOUND_V))

  def floor(self, v_ego):
    # the least braking requested at this speed through a landing
    return float(np.interp(v_ego, self.floor_bp, FLOOR_V))

  def lead_requirement(self, v_ego, lead):
    # the braking that stops the car LEAD_STANDING_GAP behind the lead, or behind where a braking lead will come to rest, in the
    # room left once the requests already on their way have played out; it grows without limit as that room closes
    if not lead.present:
      return 0.0
    closing = max(v_ego - lead.speed, 0.0)
    requirement = stop_within(closing, lead.distance - closing * self.action_t - LEAD_STANDING_GAP)
    if lead.acceleration < 0.0:
      lead_stop_distance = lead.speed * lead.speed / (-2.0 * lead.acceleration)
      requirement = max(requirement, stop_within(v_ego, lead.distance + lead_stop_distance - v_ego * self.action_t - LEAD_STANDING_GAP))
    return requirement

  def ramp_rest(self, v_ego, a_ego, request):
    # how far StarPilot's own stop would still run from here: LongControl's stopping state ramps the request from the last one sent,
    # or from zero if that asked to accelerate, toward stopAccel, and the car's acceleration follows it one actuator delay behind
    h = 0.02
    v, x, t = v_ego, 0.0, 0.0
    a = a_ego if a_ego is not None and math.isfinite(a_ego) else request
    start = min(request, 0.0)
    while v > 0.0 and t < 10.0:
      target = min(start, max(start - self.stopping_decel_rate * t, self.stop_accel))
      a += (target - a) * min(h / self.action_t, 1.0)
      x += max(v + 0.5 * a * h, 0.0) * h
      v += a * h
      t += h
    return x
  def overrun_requirement(self, v_ego, plan_stop, a_ego=None):
    # the braking that brings the car to rest within LEEWAY of the plan's own stop point once the requests already on their way
    # have played out: none while riding the corridor rests inside it, else what sheds the extra speed before the kiss, or, with
    # no room for the kiss, what stops the car in the room left
    slowing = min(a_ego, 0.0) if a_ego is not None and math.isfinite(a_ego) else 0.0
    room = plan_stop + LEEWAY - max(v_ego * self.action_t + 0.5 * slowing * self.action_t ** 2, 0.0)
    if room <= 0.0:
      return math.inf
    if corridor_distance(v_ego) <= room:
      return 0.0
    # shedding the speed before the kiss is asked only up to the corridor's own bound: beyond that, with the car still answering
    # the last request, the plain stop in the room left is what is needed
    shed = (v_ego * v_ego - KISS_SPEED * KISS_SPEED) / (2.0 * (room - KISS_DISTANCE)) if v_ego > KISS_SPEED and room > KISS_DISTANCE else 0.0
    return max(stop_within(v_ego, room), min(shed, self.bound(v_ego)))

  def sample(self, sm, *, target, stopping, enabled, active, plan_speeds, target_source, forcing, stop_target=None, action_t=None):
    """The planner's hook: returns (target, stopping) untouched unless Smooth Stops is enabled, the planner allows it and a
    landing is under way. plan_speeds is the MPC's speed solution, target_source the candidate that set target, forcing says
    Enhanced Stops is committed, stop_target is where its committed or approaching stop means to end and action_t is the planner's
    effective actuator delay."""
    if not (enabled and active):
      if not self.idle:
        self.reset()
      self._last_output, self._last_bit = target, bool(stopping)
      return target, stopping
    try:
      if action_t is not None and math.isfinite(action_t) and action_t != self.action_t:
        self.set_action_t(float(action_t))
      car, radar = sm['carState'], sm['radarState']
      leads = (Lead.from_radar(radar.leadOne), Lead.from_radar(radar.leadTwo))
      v_ego, a_target = float(car.vEgo), float(target)
      plan_end = float(plan_speeds[-1])
      # the plan intends to stop: StarPilot's own stop bit (MPC, cruise, the model, Enhanced Stops), the MPC's plan coming to rest
      # within its horizon, or an Enhanced Stops commitment
      model = target_source == E2E
      model_stop = model and a_target < 0.0 and v_ego < -a_target * MODEL_STOP_TIME
      intent = bool(stopping or forcing or model_stop or plan_end < (GO_SPEED if self.landing else STOP_INTENT_SPEED))
      # in Experimental mode the model's own stop (a red light, say) keeps StarPilot's bit, whatever a departing lead does
      model_holds = self._model_holds(sm)
      output, bit = self.update(a_target, v_ego, leads, intent, bool(stopping), a_ego=float(car.aEgo),
                                standstill=bool(car.standstill), model=model, committed=bool(forcing),
                                stop_target=float(stop_target) if stop_target is not None else None,
                                going=plan_end >= GO_SPEED and not model_holds)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
      self.reset()
      self._last_output, self._last_bit = target, bool(stopping)
      return target, stopping
    if not self.landing:
      # StarPilot's own target, and its own bit unless a launch hand-back keeps a held bit unpublished
      self.plan_target = None
      return target, bit
    self.plan_target = a_target if output != a_target else None
    return output, bit

  @staticmethod
  def _model_holds(sm):
    try:
      return bool(sm['selfdriveState'].experimentalMode and sm['modelV2'].action.shouldStop)
    except (AttributeError, KeyError, TypeError):
      return False

  def update(self, a_target, v_ego, leads, intent, stopping, a_ego=None, standstill=False, model=False, committed=False,
             stop_target=None, going=False):
    """Bound the planner's target through a landing and hold back its stop bit while the kiss lands a rolling car.

    leads are the planner's two radar leads (Lead), intent says the plan means to stop, stopping is StarPilot's own stop bit, a_ego
    the car's measured acceleration (None: the anti-stall press holds), model says the model's request set a_target and committed
    that Enhanced Stops is committed to this stop, stop_target where that stop means to end, and going that the MPC's plan is
    under way again. Returns (target, stop bit), the planner's own whenever no landing is under way.
    """
    if not math.isfinite(v_ego):
      self.reset()
      return a_target, stopping
    v_ego = max(v_ego, 0.0)
    self.committed = committed
    self.stop_target = stop_target
    self.going, self.stopping = going, stopping
    # keep each lead's last requirement through a short dropout of its reading
    for i, lead in enumerate(leads):
      if lead.present:
        self._lead_requirements[i] = self.lead_requirement(v_ego, lead)
        self._lead_missing[i] = 0
      else:
        self._lead_missing[i] = min(self._lead_missing[i] + 1, self.absence_frames)
        if self._lead_missing[i] >= self.absence_frames:
          self._lead_requirements[i] = 0.0
    self.required_decel = max(self._lead_requirements)
    if not math.isfinite(a_target):
      # a bad solver frame passes through and touches neither the latch nor the integral
      return a_target, stopping


    rolling = v_ego > STANDSTILL_SPEED and not standstill
    measured = rolling and a_ego is not None and math.isfinite(a_ego)
    # a stop handed back stays StarPilot's until the plan no longer means to stop
    if self.handed_back and (not intent or v_ego >= LANDING_SPEED):
      self.handed_back = False
    if self.landing:
      self.travelled += v_ego * self.dt
      plan = self.starpilot_request(v_ego, a_target, stopping, a_ego)
      if measured:
        # the car delivering less than StarPilot itself asks (up to the floor): creep, a weak brake, a car not slowing
        self.integral += STALL_GAIN * (a_ego + min(max(-plan, 0.0), landing_floor(v_ego)) - STALL_DEADBAND) * self.dt
      # the press fades out with the floor it is measured against
      self.integral = min(max(self.integral, 0.0), float(np.interp(v_ego, [CREEP_SPEED, CREEP_FADE_SPEED], [STALL_MAX, 0.0])))
      if rolling and stopping and not self.stop_bit:
        self.held_distance += v_ego * self.dt
        self.held_time += self.dt
      self._urgent = self._urgent + 1 if a_target < -URGENT_DECEL else 0
      # a demand of URGENT_DECEL or more passes unchanged while it lasts; the landing takes the tail back once it eases
      self.urgent = self._urgent >= self.urgent_frames or (self.urgent and a_target < -(URGENT_DECEL - URGENT_CLEAR))
      self.urgent_release |= self.urgent
      self.slow_time = self.slow_time + self.dt if v_ego <= STANDSTILL_SPEED else 0.0
      self.overrun_decel = self.overrun_requirement(v_ego, self.plan_stop(v_ego, a_target, model), a_ego) if rolling else 0.0
      self.exit_reason = 'go' if not intent else self._exit_reason(a_target, v_ego)
      if self.exit_reason:
        # a plan asking to move, and going, while the bit is held back from a rolling car launches it: StarPilot's held bit is not
        # published. A plan barely positive with StarPilot's bit up is StarPilot's stop, as it would be without Smooth Stops
        self.launching = self.exit_reason == 'launch' and going and rolling and not self.stop_bit
        self.landing = False
        self.handed_back = True
    elif (not self.handed_back and not self._last_bit and (v_ego < LANDING_SPEED or stopping) and intent and
          (a_target < -KISS_DECEL or committed and stopping)):
      self.landing = True
      self.stop_bit = False
      self.exit_reason = ''
      self.integral = 0.0
      self.held_distance = self.held_time = self.travelled = 0.0
      self.ramp_anchor = self.ramp_request = None
      self.model_anchor = math.inf
      self.bit_frames = 0
      self._urgent = 0
      self.urgent = self.urgent_release = False
      self.slow_time = 0.0
      plan = self.starpilot_request(v_ego, a_target, stopping, a_ego)
      self.overrun_decel = self.overrun_requirement(v_ego, self.plan_stop(v_ego, a_target, model), a_ego) if rolling else 0.0
    if not self.landing:
      # after a launch hand-back, StarPilot's held bit stays unpublished while the plan asks to move and is going
      if self.launching and stopping and a_target > 0.0 and going:
        self._last_output, self._last_bit = a_target, False
        return a_target, False
      self.launching = False
      self._last_output, self._last_bit = a_target, stopping
      return a_target, stopping

    if rolling or not self.stop_bit:
      # the landing only eases what StarPilot itself asks; braking beyond it comes only from the press, while the car is not
      # delivering it. Under an Enhanced Stops commitment or StarPilot's own stop bit, StarPilot stops the car whatever the plan
      # asks, so the landing never asks to accelerate
      plan = min(plan, 0.0) if self.committed or stopping else plan
      press = self.integral if rolling else 0.0
      bound = max(self.bound(v_ego), self.required_decel, self.overrun_decel)
      if self.urgent:
        # an urgent demand passes as StarPilot itself would send it: its plan, or its stopping state, which keeps a request already
        # below stopAccel (see starpilot_request)
        output = plan
      else:
        output = max(plan, -bound)
        if self.urgent_release:
          # once it eases, braking comes off at URGENT_RELEASE_JERK instead of stepping to the corridor, but never stays harder
          # than StarPilot itself now asks
          eased = max(plan, min(output, self._last_output + URGENT_RELEASE_JERK * self.dt))
          self.urgent_release = eased < output
          output = eased
      if press > 0.0:
        output = min(output, -(min(max(-plan, 0.0), self.floor(v_ego)) + press))
      output = max(output, ACCEL_MIN)
    else:
      # the wheels have stopped under the kiss and the stop bit is up: LongControl's stopping state holds the car and the plan
      # passes as it is, so the car launches from rest when it would without Smooth Stops
      output = a_target
    # the bit waits for the wheels to have all but stopped: under CRAWL_BIT_SPEED, the standstill flag, or a crawl that has lasted
    bit_ready = standstill or v_ego <= CRAWL_BIT_SPEED or self.slow_time >= CRAWL_TIME
    if not bit_ready and not self.stop_bit:
      bit = False
    else:
      # rises as the wheels stop and stays up, through wheel-speed ticks at rest, until the plan asks to move
      bit = stopping or should_stop(v_ego, a_target)
      self.stop_bit |= bit
    self._last_output, self._last_bit = output, bit
    return output, bit

  def starpilot_request(self, v_ego, a_target, stopping, a_ego):
    # what StarPilot itself would ask now: its plan, or once its own stop bit has stayed up ANCHOR_TIME, LongControl's stopping ramp
    # from the plan's request at the rise toward stopAccel. Where that ramp would end the stop is fixed to the road then; the bit
    # dropping again clears both
    if not stopping:
      self.bit_frames = 0
      self.ramp_request = self.ramp_anchor = None
      return a_target
    self.bit_frames += 1
    if self.bit_frames == 1:
      self._rise_request = min(a_target, 0.0)
    if self.bit_frames < self.anchor_frames:
      return a_target
    # LongControl's stopping state ramps toward stopAccel, but keeps a request already below it
    self.ramp_request = min(self._rise_request,
                            max(self._rise_request - self.stopping_decel_rate * (self.bit_frames - 1) * self.dt, self.stop_accel))
    if self.ramp_anchor is None:
      self.ramp_anchor = self.travelled + self.ramp_rest(v_ego, a_ego, self.ramp_request)
    return self.ramp_request

  def plan_stop(self, v_ego, a_target, model):
    # how far ahead StarPilot itself would bring the car to rest, fixed to the road so a car that keeps rolling uses it up: where its
    # stopping ramp would have ended the stop, and the nearest point the model's request, held once the car answers it, has asked
    # to stop at (negative once the car has passed it)
    if self.committed and self.stop_target is not None:
      # an Enhanced Stops commitment knows its own stop point; the landing may use it all but never pass it
      return self.stop_target - LEEWAY
    # approaching a light Enhanced Stops tracks, the landing never passes its line either
    line = self.stop_target - LEEWAY if self.stop_target is not None else math.inf
    if model:
      # the model's request rises against a car that falls behind its stop point, so a model stop gets no leeway past that point
      self.model_anchor = min(self.model_anchor, self.travelled + v_ego * self.action_t + held_stop_distance(v_ego, a_target) - LEEWAY)
    return min(min(self.ramp_anchor if self.ramp_anchor is not None else math.inf, self.model_anchor) - self.travelled, line)
  def _exit_reason(self, a_target, v_ego):
    # a plan asking to move ends the landing when the plan is under way or StarPilot's own bit is down; a plan barely positive under
    # StarPilot's bit (a lead creeping, the model's red light) is still StarPilot's stop, and the landing keeps easing it to rest
    if a_target > 0.0 and not self.committed and (self.going or not self.stopping):
      return 'launch'
    if v_ego >= LANDING_SPEED and not self.committed and not self.stopping:
      return 'window'
    if self.required_decel > self.bound(v_ego):
      return 'lead'
    if self.overrun_decel == math.inf:
      return 'overrun'
    if self.integral >= STALL_MAX - 1e-9:
      return 'stall'
    # (summed per frame: allow for the float sum landing a hair short of the budget)
    if self.held_distance >= HOLD_DISTANCE - 1e-6 or self.held_time >= HOLD_TIME - 1e-6:
      return 'hold'
    return ''
