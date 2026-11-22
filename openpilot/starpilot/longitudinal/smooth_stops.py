"""Smooth Stops: let off the brake through the last metres of a stop behind a lead so the wheels stop under a whisper of braking."""
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
# Smooth Stops shapes the planner's output through the last metres of a stop behind a lead instead. A landing starts below
# LANDING_SPEED when the MPC plans to stop behind a lead and the plan is braking. While it lasts, the requested deceleration stays
# inside a corridor that falls with speed:
#
# * a bound that only removes surplus braking; whatever stops the car short of the lead always passes;
# * a floor, so the stop completes against creep torque instead of coasting;
# * both taper to one small "kiss" of braking at walking pace.
#
# The car answers a request one actuator delay later, so each edge of the corridor is requested that much earlier. Once StarPilot's
# own stop bit rises, the landing holds it back for at most HOLD_DISTANCE of travel and HOLD_TIME, so LongControl's stopping ramp
# and the car's standstill hold start at or near rest. At rest the plan passes unchanged and the bit stays up until the plan asks
# to move.
#
# Anything a landing does not expect hands the stop back to StarPilot at once, unchanged: the plan asking to move or no longer
# stopping behind a lead, another candidate (the model, cruise) asking clearly harder than the corridor, a lead closer than the
# kiss can stop short of, a car that keeps not slowing (the anti-stall press saturated), or the hold-back budget spent. The rest of that stop is StarPilot's; a
# hand-back that lets the plan's own braking through steps the request in one frame, toward more braking.
#
# The corridor was measured on a Hyundai Palisade; smooth_stops_preferences.supported() limits it to cars like it.
LANDING_SPEED = 3.5          # m/s, the corridor acts below this; at the top the bound (2.75 m/s^2) is above any comfortable approach
STOP_INTENT_SPEED = 0.5      # m/s, the MPC's planned speed must fall below this behind a lead for a slowdown to count as a stop
KISS_SPEED = 0.40            # m/s, from here down the car rides the kiss alone
KISS_DECEL = 0.15            # m/s^2, the braking the wheels stop under; the stopping ramp and the car's hold take over at rest
CREEP_SPEED = 1.0            # m/s, the floor peaks at CREEP_DECEL here: enough to keep slowing against creep torque
CREEP_DECEL = 0.40           # m/s^2
CREEP_FADE_SPEED = 1.5       # m/s, and it is gone here: a queue rolling at 2 m/s is not held to a stop's floor
BOUND_BP = [KISS_SPEED, 0.9, LANDING_SPEED]    # the car's deceleration against its speed: at most 0.70 * v + 0.30 above
BOUND_V = [KISS_DECEL, 0.93, 2.75]             # 0.9 m/s, straight down to the kiss below
FLOOR_BP = [KISS_SPEED, CREEP_SPEED, CREEP_FADE_SPEED]
FLOOR_V = [KISS_DECEL, CREEP_DECEL, 0.0]
LEAD_MIN_GAP_BUDGET = 0.5    # m, the least room a stop behind a lead is planned in, and the least it plans to leave to the lead
# m, the gap the landing protects behind a lead: about half the MPC's STOP_DISTANCE, inside where StarPilot's lead stops come to
# rest. A car the kiss cannot stop outside it (a weak brake, creep, a lead appearing late) is handed back
LEAD_STANDING_GAP = 3.0
LEAD_ABSENCE_TIME = 0.35     # s a lead's requirement is kept after its reading drops out
STANDSTILL_SPEED = 0.10      # m/s, about the car's standstill flag (0.104 m/s of wheel speed on Hyundai)
GO_SPEED = 1.5               # m/s, the MPC's planned speed at the end of its horizon above which the lead stop is over
URGENT_MARGIN = 0.3          # m/s^2 beyond the corridor, from a candidate other than the MPC's lead stop (the model, cruise) ...
URGENT_TIME = 0.1            # s ... for this long: the plan is stopping for something else
HOLD_DISTANCE = 0.5          # m, the most the car travels with StarPilot's own stop bit held back ...
HOLD_TIME = 1.5              # s ... and for how long
RESUME_SPEED = CREEP_FADE_SPEED  # m/s, after a stop is handed back, the next landing waits for the car to be rolling this fast
# Anti-stall: while rolling, the car's shortfall against the floor presses both edges toward more braking and drains as the car
# slows again. It fades out with the floor and is not applied at rest. A press that reaches STALL_MAX hands the stop back
STALL_GAIN = 1.0             # 1/s
STALL_DEADBAND = 0.05        # m/s^2, about one standard deviation of aEgo while braking
STALL_MAX = 0.5              # m/s^2
LEAD_SOURCES = (log.LongitudinalPlan.LongitudinalPlanSource.lead0, log.LongitudinalPlan.LongitudinalPlanSource.lead1)


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


def owner_for(cp, dt):
  try:
    if cp.openpilotLongitudinalControl and not cp.pcmCruise and supported(cp):
      return SmoothStops(dt, cp.longitudinalActuatorDelay + DT_MDL)
  except (AttributeError, TypeError, ValueError):
    pass
  return None


class SmoothStops:
  def __init__(self, dt, action_t):
    self.dt = dt
    self.action_t = action_t
    self.bound_bp = request_breakpoints(BOUND_BP, BOUND_V, action_t)
    self.floor_bp = request_breakpoints(FLOOR_BP, FLOOR_V, action_t)
    self.absence_frames = max(round(LEAD_ABSENCE_TIME / dt), 1)
    self.urgent_frames = max(round(URGENT_TIME / dt), 1)
    self.reset()

  def reset(self):
    self.landing = False
    self.handed_back = False
    self.stop_bit = False
    self.exit_reason = ''
    self.integral = 0.0
    self.required_decel = 0.0
    self.held_distance = 0.0
    self.held_time = 0.0
    self.plan_target = None
    self._urgent = 0
    self._lead_requirements = [0.0, 0.0]
    self._lead_missing = [self.absence_frames] * 2

  @property
  def idle(self):
    return not (self.landing or self.handed_back or any(self._lead_requirements))

  def plan_seed(self, published):
    # the acceleration the MPC continues from: the plan the landing softened last frame, else what was published
    return published if self.plan_target is None else max(self.plan_target, ACCEL_MIN)

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

  def sample(self, sm, *, target, stopping, enabled, active, plan_source, plan_speeds, target_source):
    """The planner's hook: returns (target, stopping) untouched unless Smooth Stops is enabled, the planner allows it and a
    landing is under way. plan_source and plan_speeds are the MPC's own, target_source the candidate that set target."""
    if not (enabled and active):
      if not self.idle:
        self.reset()
      return target, stopping
    try:
      car, radar = sm['carState'], sm['radarState']
      leads = (Lead.from_radar(radar.leadOne), Lead.from_radar(radar.leadTwo))
      # the MPC's own plan comes to rest behind a lead within its horizon, with radarState's leadOne present (as Conditional mode's
      # launch scene reads it, so a held-back stop bit never looks like an open road there)
      mpc_lead = plan_source in LEAD_SOURCES and leads[0].present
      plan_end = float(plan_speeds[-1])
      lead_stop = bool(mpc_lead and plan_end < (GO_SPEED if self.landing else STOP_INTENT_SPEED))
      output, bit = self.update(float(target), float(car.vEgo), leads, lead_stop, bool(stopping), a_ego=float(car.aEgo),
                                standstill=bool(car.standstill), other_target=target_source not in LEAD_SOURCES)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      self.reset()
      return target, stopping
    if not self.landing:
      self.plan_target = None
      return target, stopping
    self.plan_target = float(target) if output != target else None
    return output, bit

  def update(self, a_target, v_ego, leads, lead_stop, stopping, a_ego=None, standstill=False, other_target=False):
    """Bound the planner's target through a landing and hold back its stop bit while the kiss lands a rolling car.

    leads are the planner's two radar leads (Lead), lead_stop says the MPC plans to stop behind one, stopping is StarPilot's own stop
    bit, a_ego the car's measured acceleration (None: the anti-stall integral holds) and other_target says a candidate other than
    the MPC's lead stop set a_target. Returns (target, stop bit), the planner's own whenever no landing is under way.
    """
    if not math.isfinite(v_ego):
      self.reset()
      return a_target, stopping
    v_ego = max(v_ego, 0.0)
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
    if self.handed_back and v_ego >= RESUME_SPEED:
      self.handed_back = False
    if self.landing:
      if rolling and a_ego is not None and math.isfinite(a_ego):
        self.integral += STALL_GAIN * (a_ego + landing_floor(v_ego) - STALL_DEADBAND) * self.dt
      # the press fades out with the floor it is measured against
      self.integral = min(max(self.integral, 0.0), float(np.interp(v_ego, [CREEP_SPEED, CREEP_FADE_SPEED], [STALL_MAX, 0.0])))
      if rolling and stopping and not self.stop_bit:
        self.held_distance += v_ego * self.dt
        self.held_time += self.dt
      # the MPC's own late ramp behind the lead is what the corridor softens (the lead's requirement guards its geometry); a harder
      # demand from anything else is stopping for something the lead does not explain
      urgent = other_target and a_target < -(self.bound(v_ego) + self.integral + URGENT_MARGIN)
      self._urgent = self._urgent + 1 if urgent else 0
      self.exit_reason = 'go' if not lead_stop else self._exit_reason(a_target, v_ego)
      if self.exit_reason:
        self.landing = False
        self.handed_back = True
    elif not self.handed_back and v_ego < LANDING_SPEED and lead_stop and a_target < -KISS_DECEL:
      self.landing = True
      self.stop_bit = False
      self.exit_reason = ''
      self.integral = 0.0
      self.held_distance = self.held_time = 0.0
      self._urgent = 0
    if not self.landing:
      return a_target, stopping

    if rolling or not self.stop_bit:
      press = self.integral if rolling else 0.0
      floor = self.floor(v_ego) + press
      bound = max(self.bound(v_ego) + press, self.required_decel)
      output = min(a_target, -floor) if floor > 0.0 else a_target
      output = max(output, -bound, ACCEL_MIN)
    else:
      # the wheels have stopped under the kiss and the stop bit is up: LongControl's stopping state holds the car and the plan
      # passes as it is, so the car launches from rest when it would without Smooth Stops
      output = a_target
    if rolling and not self.stop_bit:
      bit = False
    else:
      # rises as the wheels stop and stays up, through wheel-speed ticks at rest, until the plan asks to move
      bit = stopping or should_stop(v_ego, a_target)
      self.stop_bit |= bit
    return output, bit

  def _exit_reason(self, a_target, v_ego):
    if a_target > 0.0:
      return 'launch'
    if v_ego >= LANDING_SPEED:
      return 'window'
    if self._urgent >= self.urgent_frames:
      return 'urgent'
    if self.required_decel > self.bound(v_ego):
      return 'lead'
    if self.integral >= STALL_MAX - 1e-9:
      return 'stall'
    if self.held_distance >= HOLD_DISTANCE or self.held_time >= HOLD_TIME:
      return 'hold'
    return ''
