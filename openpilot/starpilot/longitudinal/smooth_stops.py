"""Smooth Stops: let off the brake through the last metres of a stop so the wheels stop under a whisper of braking."""
from dataclasses import dataclass
import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.starpilot.longitudinal.smooth_stops_preferences import supported

# Without it, a stop ends in LongControl's stopping state: once the plan's stop bit rises (on HKG cars at walking pace, about a
# second before the wheels stop) LongControl no longer follows the plan and ramps the request toward CP.stopAccel, so braking rises
# into the stop and the body rocks forward and back as the wheels stop.
#
# Smooth Stops shapes the planner's output through the last few metres of a stop instead. A landing starts below LANDING_SPEED when
# the plan intends to stop and is braking. While it lasts, the requested deceleration stays inside a corridor that falls with speed:
#
# * a bound that only removes surplus braking. It never makes the stop the plan is asking for more than STOP_ALLOWANCE longer, and
#   whatever stops the car short of a lead in the room left after the actuator delay always passes;
# * a floor, so the stop completes against creep torque instead of coasting;
# * both taper to one small "kiss" of braking at walking pace.
#
# The car answers a request one actuator delay later, so each edge of the corridor is requested that much earlier. The stop bit is
# held back while the kiss lands a rolling car and rises once the wheels have nearly stopped, so LongControl's stopping ramp and the
# car's standstill hold start at rest. A lead nearer than the kiss can stop short of gets the stop bit at once. Once the bit has
# risen, the landing keeps it up until StarPilot's own plan drops it, so the hold and the launch from rest stay StarPilot's.
#
# The corridor was measured on a Hyundai Palisade; smooth_stops_preferences.supported() limits it to cars whose stop runs through
# LongControl's own stopping state and the SCC stop request.
LANDING_SPEED = 3.5          # m/s, the corridor acts below this; at the top the bound (2.75 m/s^2) is above any comfortable approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's planned speed must fall below this for a slowdown to count as a stop
KISS_SPEED = 0.40            # m/s, from here down the car rides the kiss alone
KISS_DECEL = 0.15            # m/s^2, the braking the wheels stop under; the stopping ramp and the car's hold take over at rest
CREEP_SPEED = 1.0            # m/s, the floor peaks at CREEP_DECEL here: enough to keep slowing against creep torque
CREEP_DECEL = 0.40           # m/s^2
CREEP_FADE_SPEED = 1.5       # m/s, and it is gone here: a queue rolling at 2 m/s is not held to a stop's floor
BOUND_BP = [KISS_SPEED, 0.9, LANDING_SPEED]    # the car's deceleration against its speed: at most 0.70 * v + 0.30 above
BOUND_V = [KISS_DECEL, 0.93, 2.75]             # 0.9 m/s, straight down to the kiss below
FLOOR_BP = [KISS_SPEED, CREEP_SPEED, CREEP_FADE_SPEED]
FLOOR_V = [KISS_DECEL, CREEP_DECEL, 0.0]
STOP_ALLOWANCE = 0.4         # m, the most the corridor may lengthen the stop the plan asks for
LAUNCH_FRAMES = 3            # frames of a positive plan above KISS_SPEED that end a landing: a hover alternates, a launch climbs
LEAD_MIN_GAP_BUDGET = 0.5    # m, the least room a stop behind a lead is planned in, and the least it plans to leave to the lead
# m, the gap the landing protects behind a lead: about half the MPC's STOP_DISTANCE, inside where StarPilot's lead stops come to
# rest. A car the kiss cannot stop outside it (a weak brake, creep, a lead appearing late) gets the stop bit and the full plan
LEAD_STANDING_GAP = 3.0
LEAD_ABSENCE_FRAMES = 7      # frames a lead's requirement is kept after its reading drops out
STANDSTILL_SPEED = 0.10      # m/s, about the car's standstill flag (0.104 m/s of wheel speed on Hyundai)
CRAWL_TIME = 2.0             # s below KISS_SPEED after which the stop bit is no longer held back, however the car is creeping
# Anti-stall: while rolling, the car's shortfall against the floor presses both edges toward more braking and drains as the car
# slows again. It fades out with the floor and is not applied at rest, where the hold clamps the car.
STALL_GAIN = 1.0             # 1/s
STALL_DEADBAND = 0.05        # m/s^2, about one standard deviation of aEgo while braking
STALL_MAX = 0.5              # m/s^2, the most the corridor is pressed toward more braking


def landing_bound(v_ego):
  # the most braking the car should carry at this speed
  return float(np.interp(v_ego, BOUND_BP, BOUND_V))


def landing_floor(v_ego):
  # the least braking the car should carry at this speed through a landing
  return float(np.interp(v_ego, FLOOR_BP, FLOOR_V))


def stop_within(speed, room):
  # the deceleration that sheds speed within room; with no room left any braking the plan asks is needed
  if speed <= 0.0:
    return 0.0
  return speed * speed / (2.0 * room) if room > 0.0 else math.inf


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


def owner_for(cp, dt, release_jerk):
  try:
    if cp.openpilotLongitudinalControl and not cp.pcmCruise and supported(cp):
      return SmoothStops(dt, cp.longitudinalActuatorDelay + DT_MDL, release_jerk)
  except (AttributeError, TypeError, ValueError):
    pass
  return None


class SmoothStops:
  def __init__(self, dt, action_t, release_jerk):
    self.dt = dt
    self.action_t = action_t
    # (speed breakpoints, m/s^3): the rate a held output is handed back at, the planner's cruise jerk
    self.release_jerk = release_jerk
    self.bound_bp = request_breakpoints(BOUND_BP, BOUND_V, action_t)
    self.floor_bp = request_breakpoints(FLOOR_BP, FLOOR_V, action_t)
    self.crawl_frames = round(CRAWL_TIME / dt)
    self.reset()

  def reset(self):
    self.landing = False
    self.releasing = False
    self.stop_bit = False
    self.rested = False
    self.output = 0.0
    self.integral = 0.0
    self.required_decel = 0.0
    self._held = False
    self._positive_frames = 0
    self._crawl = 0
    self._lead_requirements = [0.0, 0.0]
    self._lead_missing = [LEAD_ABSENCE_FRAMES, LEAD_ABSENCE_FRAMES]

  @property
  def idle(self):
    return not (self.landing or self.releasing or self.rested or any(self._lead_requirements))

  def bound(self, v_ego):
    # the most braking requested at this speed
    return float(np.interp(v_ego, self.bound_bp, BOUND_V))

  def floor(self, v_ego):
    # the least braking requested at this speed through a landing
    return float(np.interp(v_ego, self.floor_bp, FLOOR_V))

  def lead_requirement(self, v_ego, lead):
    # the braking that stops the car LEAD_STANDING_GAP behind the lead, planned in at least LEAD_MIN_GAP_BUDGET of room; and
    # whatever stops it LEAD_MIN_GAP_BUDGET short of the lead, or of where a braking lead will come to rest, in the room left once
    # the requests already on their way have played out, which grows without limit as that room closes
    if not lead.present:
      return 0.0
    closing = max(v_ego - lead.speed, 0.0)
    gap_budget = max(lead.distance - LEAD_STANDING_GAP, LEAD_MIN_GAP_BUDGET)
    requirement = max(closing * closing / (2.0 * gap_budget),
                      stop_within(closing, lead.distance - closing * self.action_t - LEAD_MIN_GAP_BUDGET))
    if lead.acceleration < 0.0:
      lead_stop_distance = lead.speed * lead.speed / (-2.0 * lead.acceleration)
      requirement = max(requirement, v_ego * v_ego / (2.0 * (gap_budget + lead_stop_distance)),
                        stop_within(v_ego, lead.distance + lead_stop_distance - v_ego * self.action_t - LEAD_MIN_GAP_BUDGET))
    return requirement

  @staticmethod
  def plan_requirement(v_ego, a_target):
    # the braking that still stops the car within STOP_ALLOWANCE of where the plan's own request would: a sudden stop for something
    # only the model sees, or any other urgent request, keeps nearly all of its braking
    braking = -a_target
    if braking <= 0.0 or v_ego <= 0.0:
      return 0.0
    return braking * v_ego * v_ego / (v_ego * v_ego + 2.0 * STOP_ALLOWANCE * braking)

  def sample(self, sm, *, target, stopping, enabled, active, plan_speeds):
    """The planner's hook: returns (target, stopping) untouched unless Smooth Stops is enabled and the planner allows it."""
    if not (enabled and active):
      if not self.idle:
        self.reset()
      return target, stopping
    try:
      car, radar, model = sm['carState'], sm['radarState'], sm['modelV2']
      stop_intent = bool(stopping or float(np.min(plan_speeds)) < STOP_INTENT_SPEED or
                         sm['selfdriveState'].experimentalMode and model.action.shouldStop)
      return self.update(float(target), float(car.vEgo), (Lead.from_radar(radar.leadOne), Lead.from_radar(radar.leadTwo)),
                         stop_intent, bool(stopping), a_ego=float(car.aEgo), standstill=bool(car.standstill))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      self.reset()
      return target, stopping

  def update(self, a_target, v_ego, leads, stop_intent, stopping, a_ego=None, standstill=False):
    """Bound the planner's target through a landing and hold back its stop bit while the kiss lands a rolling car.

    leads are the planner's two radar leads (Lead), stop_intent says the plan ends in a stop, stopping is StarPilot's own stop bit
    and a_ego the car's measured acceleration (None: the anti-stall integral holds). Returns (target, stop bit).
    """
    if not math.isfinite(v_ego):
      self.reset()
      return a_target, stopping
    v_ego = max(v_ego, 0.0)
    # Keep each lead's last requirement through a short dropout; this only lets planned braking through the bound, it adds none
    expired_requirement = 0.0
    for i, lead in enumerate(leads):
      if lead.present:
        self._lead_requirements[i] = self.lead_requirement(v_ego, lead)
        self._lead_missing[i] = 0
      else:
        self._lead_missing[i] = min(self._lead_missing[i] + 1, LEAD_ABSENCE_FRAMES)
        if self._lead_missing[i] >= LEAD_ABSENCE_FRAMES:
          expired_requirement = max(expired_requirement, self._lead_requirements[i])
          self._lead_requirements[i] = 0.0
    self.required_decel = max(self._lead_requirements)
    if not math.isfinite(a_target):
      # a bad solver frame passes through and touches neither the latch nor the integral
      return a_target, stopping

    rolling = v_ego > STANDSTILL_SPEED and not standstill
    if self.rested and v_ego >= KISS_SPEED:
      self.rested = False
    if self.landing:
      # a climbing plan ends the landing only while rolling; at rest StarPilot's own stop bit owns the hold and the launch, and its
      # drop ends the landing at once with nothing held back
      self._positive_frames = self._positive_frames + 1 if a_target > 0.0 and v_ego > KISS_SPEED else 0
      if v_ego >= LANDING_SPEED or self._positive_frames >= LAUNCH_FRAMES:
        self.landing = False
        self.releasing = self._held
      elif self.stop_bit and not rolling and not stopping:
        self.landing = False
        self.releasing = False
        self.rested = True
    elif not self.rested and v_ego < LANDING_SPEED and stop_intent and a_target < -KISS_DECEL:
      # a landing starts on intent with the plan braking more than the kiss; after a stop the car is StarPilot's until it rolls
      # faster than walking pace again
      self.landing = True
      self.stop_bit = False
      self._positive_frames = 0
      self._crawl = 0
      self.integral = 0.0

    output = a_target
    # once the stop bit is up and the wheels have stopped, LongControl's stopping state holds the car and the plan passes as it is,
    # so the MPC, which continues from the published target, launches from rest exactly when it would without the landing
    if self.landing and (rolling or not self.stop_bit):
      if rolling and a_ego is not None and math.isfinite(a_ego):
        self.integral += STALL_GAIN * (a_ego + landing_floor(v_ego) - STALL_DEADBAND) * self.dt
      # the press fades out with the floor it is measured against
      self.integral = min(max(self.integral, 0.0), float(np.interp(v_ego, [CREEP_SPEED, CREEP_FADE_SPEED], [STALL_MAX, 0.0])))
      press = self.integral if rolling else 0.0
      floor = self.floor(v_ego) + press
      bound = max(self.bound(v_ego) + press, self.required_decel, self.plan_requirement(v_ego, a_target))
      # a lead's requirement ending after its dropout must not jump braking into the corridor
      self.releasing |= expired_requirement > bound and self.output < -bound
      output = min(a_target, -floor) if floor > 0.0 else a_target
      output = max(output, -bound, ACCEL_MIN)
    if self.releasing:
      # whatever the landing held below the plan goes back at the planner's cruise jerk; braking passes at once
      ceiling = self.output + float(np.interp(v_ego, *self.release_jerk)) * self.dt
      self.releasing = output > ceiling
      output = min(output, ceiling)
    self._held = output < a_target
    self.output = output

    bit = stopping
    if self.landing:
      self._crawl = self._crawl + 1 if rolling and v_ego < KISS_SPEED else 0
      if (rolling and not self.stop_bit and self.required_decel <= self.bound(v_ego) and self._crawl < self.crawl_frames):
        bit = False
      else:
        bit = stopping or should_stop(v_ego, output)
        self.stop_bit |= bit
    return output, bit
