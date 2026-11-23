"""Smooth Stops through StarPilot's own planner and LongControl, closed loop on a lagged car, against the same drive with it off."""
from collections import deque
from dataclasses import dataclass
import math

import pytest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.starpilot.longitudinal.force_stop import StopPlan
from openpilot.starpilot.longitudinal.smooth_stops import LEEWAY, STANDSTILL_SPEED
from openpilot.starpilot.longitudinal.tests.test_cruise_ceiling import messages

BASE_NS = 10_000_000_000
DRIVE_NS = BASE_NS - 1_000_000_000
SERVICES = ('radarState', 'controlsState', 'selfdriveState', 'carState', 'vehicleParameters', 'carControl', 'modelV2')
# the plants the review asked for: a brake as asked, one 0.1 short with creep torque, one 0.2 short with more creep
PLANTS = ((0.0, 0.0), (0.1, 0.25), (0.2, 0.3))
# the worst extra time to rest a landing costs on these drives, measured 2.55 s on the stopped-lead grid (a brake 0.1 short with
# creep); an unbounded crawl takes several seconds more
EXTRA_TIME = 3.0
# peak deceleration and peak jerk against the same drive without Smooth Stops: a landing only eases what StarPilot asks (measured
# worst +0.07 m/s^2 and +3.0 m/s^3, both on a sudden model stop the landing hands back as urgent)
PEAK_DECEL = 0.08
PEAK_JERK = 3.5


def palisade():
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.openpilotLongitudinalControl, cp.pcmCruise = True, False
  return cp


class Frame(dict):
  # every source current and valid for this drive, as plannerd's SubMaster has them, so the HKG vision stop buffer runs
  def __init__(self, payloads, now_ns):
    super().__init__(payloads)
    self.logMonoTime = dict.fromkeys(SERVICES, now_ns - 10_000_000)
    self.valid = dict.fromkeys(SERVICES, True)
    self.alive = dict.fromkeys(SERVICES, True)


@dataclass
class Lead:
  x: float              # where it is, in the car's starting frame
  speed: float = 0.0
  acceleration: float = 0.0
  radar: bool = True


class Drive:
  """The planner at 20 Hz and LongControl at 100 Hz on a car with an actuator dead time, apply and release lags, a brake that
  may deliver less than asked, and creep torque the brake has to beat below walking pace."""
  def __init__(self, v0, *, smooth, shortfall=0.0, creep=0.0, dead=0.3, tau_apply=0.45, tau_release=0.40, a0=0.0):
    self.cp = palisade()
    self.planner = LongitudinalPlanner(self.cp, init_v=v0, init_a=a0)
    self.lc = LongControl(self.cp)
    self.smooth = smooth
    self.v, self.a, self.x = v0, a0, 0.0
    self.pipe = deque([a0] * round(dead / DT_CTRL))
    self.shortfall, self.creep = shortfall, creep
    self.tau_apply, self.tau_release = tau_apply, tau_release
    self.frame = 0
    self.trace = []
    self.car_state = messaging.new_message('carState').carState

  def step(self, *, lead=None, experimental=False, model_stop=False, e2e=0.0, v_cruise=10.0, force_decel=False,
           force_stop=None):
    payloads, _ = messages(e2e=experimental, force=force_decel)
    car = payloads['carState']
    car.vEgo, car.aEgo, car.standstill, car.canValid = self.v, self.a, self.v < 0.01, True
    car.vCruise = v_cruise * 3.6
    payloads['carControl'].longActive = True
    payloads['modelV2'].action.desiredAcceleration = e2e
    payloads['modelV2'].action.shouldStop = model_stop
    radar = payloads['radarState']
    for slot in (radar.leadOne, radar.leadTwo):
      slot.present = False
    if lead is not None:
      radar.leadOne.present = True
      radar.leadOne.dRel = lead.x - self.x
      radar.leadOne.vLead = radar.leadOne.vLeadK = lead.speed
      radar.leadOne.vRel = lead.speed - self.v
      radar.leadOne.aLeadK = lead.acceleration
      radar.leadOne.modelProb = 1.0
      radar.leadOne.radar = lead.radar
    now_ns = BASE_NS + self.frame * 50_000_000
    provider = (lambda follow, plan=force_stop: plan) if force_stop is not None else None
    self.planner.update(Frame(payloads, now_ns), now_ns=now_ns, drive_id=DRIVE_NS, smooth_stops=self.smooth,
                        force_stop_provider=provider)
    target, stop = float(self.planner.output_a_target), bool(self.planner.output_should_stop)
    for _ in range(round(DT_MDL / DT_CTRL)):
      self.car_state.vEgo, self.car_state.aEgo, self.car_state.standstill = self.v, self.a, self.v < 0.01
      request = float(self.lc.update(True, self.car_state, target, stop, (-3.5, 2.0)))
      self.pipe.append(request)
      self._advance(self.pipe.popleft())
    self.trace.append((self.frame * DT_MDL, self.x, self.v, self.a, target, stop, self.lc.long_control_state))
    self.frame += 1

  def _advance(self, request):
    if request < 0.0:
      request = min(request + self.shortfall, 0.0)
    if self.v < 1.5 and request > -0.3:
      request += self.creep * max(0.0, 1.0 - self.v)
    tau = self.tau_apply if request < self.a else self.tau_release
    self.a += (request - self.a) * DT_CTRL / tau
    if self.v <= 0.0 and self.a < 0.0:
      self.a = 0.0 if request <= 0.0 else self.a
    v_new = max(self.v + self.a * DT_CTRL, 0.0)
    self.x += 0.5 * (self.v + v_new) * DT_CTRL
    self.v = v_new

  @property
  def rest(self):
    stopped = [row for row in self.trace if row[2] <= 0.0]
    return (stopped[0][0], stopped[0][1]) if stopped else (math.inf, self.x)

  @property
  def stop_bit_pulses(self):
    # runs of the published stop bit, of any length, that end without the car having come to (about) rest under them
    pulses, run = 0, []
    for row in self.trace:
      if row[5]:
        run.append(row[2])
      elif run:
        pulses += int(min(run) > STANDSTILL_SPEED + 0.01)
        run = []
    return pulses

  def peaks(self):
    # peak deceleration and peak jerk (frame-to-frame change of the car's acceleration) through the stop and a second of the hold,
    # so the jolt as the wheels stop counts
    rows = [row for row in self.trace if row[0] <= self.rest[0] + 1.0]
    return -min(row[3] for row in rows), max(abs(b[3] - a[3]) / DT_MDL for a, b in zip(rows, rows[1:], strict=False))


def untracked_object(v0, distance, smooth, shortfall=0.0, creep=0.0, early_bit=False, seconds=16.0):
  # Experimental: the model stops for something radar does not track, aiming 1 m short of it at the constant deceleration that
  # does so; the car is at v0 and not braking yet. The model's stop bit comes at walking pace, as in the logs, or at once
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep)
  for _ in range(round(seconds / DT_MDL)):
    room = max(distance - drive.x - 1.0, 0.05)
    e2e = -min(drive.v ** 2 / (2.0 * room), 4.0) if drive.v > 0.0 else -0.3
    drive.step(experimental=True, model_stop=early_bit or drive.v < 0.3, e2e=e2e)
  return drive


def stopped_lead(v0, distance, smooth, *, shortfall=0.0, creep=0.0, experimental=False, radar=True, seconds=20.0):
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep)
  lead = Lead(distance, radar=radar)
  for _ in range(round(seconds / DT_MDL)):
    drive.step(lead=lead, experimental=experimental, model_stop=experimental and drive.v < 0.3, e2e=-0.5 if experimental else 0.0)
  gaps = [lead.x - row[1] for row in drive.trace]
  return drive, min(gaps), lead.x - drive.rest[1]


def rolling_launch(v_trigger, smooth, *, distance=14.0, v0=2.0, lead_accel=2.5, seconds=14.0):
  # the stopped lead pulls away briskly once the car has slowed below v_trigger, still rolling
  drive = Drive(v0, smooth=smooth)
  lead = Lead(distance)
  t_trigger = None
  # frames a landing softened a plan that asked to move
  drive.held_positive = 0
  for i in range(round(seconds / DT_MDL)):
    if t_trigger is None and drive.v < v_trigger and drive.frame > 5:
      t_trigger = i * DT_MDL
    if t_trigger is not None:
      lead.speed = min(lead.speed + lead_accel * DT_MDL, 8.0)
      lead.x += lead.speed * DT_MDL
    drive.step(lead=lead, v_cruise=10.0)
    owner = drive.planner.smooth_stops
    drive.held_positive += int(owner is not None and owner.plan_target is not None and owner.plan_target > 0.0)
  # time from the lead's first move to the car passing 2 m/s
  t_go = next((row[0] for row in drive.trace if t_trigger is not None and row[0] >= t_trigger and row[2] > 2.0), math.inf)
  return drive, t_go - (t_trigger if t_trigger is not None else 0.0)


def at_stop(drive):
  # braking still on as the wheels stop
  t_rest = drive.rest[0]
  return -sum(r[3] for r in drive.trace if t_rest - 0.3 <= r[0] <= t_rest - 0.1) / 5.0


@pytest.mark.parametrize('v0,distance', [(2.0, 4.0), (3.0, 6.0), (2.0, 8.0), (3.4, 10.0)])
@pytest.mark.parametrize('shortfall,creep', PLANTS)
@pytest.mark.parametrize('early_bit', [False, True])
def test_a_stop_for_something_only_the_model_sees_rests_within_the_leeway(v0, distance, shortfall, creep, early_bit):
  # the review's weak and creeping brakes rolled past the object when an earlier landing was not bounded
  off = untracked_object(v0, distance, False, shortfall, creep, early_bit)
  on = untracked_object(v0, distance, True, shortfall, creep, early_bit)
  assert math.isfinite(on.rest[0]), 'the car must come to rest'
  assert on.rest[1] <= off.rest[1] + LEEWAY + 0.05
  assert on.rest[1] < distance or off.rest[1] >= distance
  assert on.rest[0] - off.rest[0] <= EXTRA_TIME
  assert on.stop_bit_pulses == 0
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.peaks()[1] <= off.peaks()[1] + PEAK_JERK


@pytest.mark.parametrize('v0,distance', [(1.0, 8.0), (2.0, 9.0), (3.0, 10.0), (3.4, 9.5), (3.4, 12.0), (2.0, 12.0), (2.5, 14.0)])
@pytest.mark.parametrize('shortfall,creep', [(0.0, 0.0), (0.1, 0.0), (0.1, 0.25)])
@pytest.mark.parametrize('experimental', [False, True])
def test_a_stopped_lead_stop_is_bounded_on_weak_and_creeping_brakes(v0, distance, shortfall, creep, experimental):
  off, off_least, off_rest = stopped_lead(v0, distance, False, shortfall=shortfall, creep=creep, experimental=experimental)
  on, on_least, on_rest = stopped_lead(v0, distance, True, shortfall=shortfall, creep=creep, experimental=experimental)
  assert math.isfinite(on.rest[0]), 'the car must come to rest'
  assert off_rest - on_rest <= LEEWAY + 0.05, 'extra roll'
  assert on.rest[0] - off.rest[0] <= EXTRA_TIME, 'extra time'
  assert on_least >= min(off_least - 0.5, 3.0)
  assert on.stop_bit_pulses == 0
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.peaks()[1] <= off.peaks()[1] + PEAK_JERK


def test_a_normal_stop_behind_a_lead_ends_under_the_kiss():
  off, _, _ = stopped_lead(2.0, 12.0, False)
  on, _, _ = stopped_lead(2.0, 12.0, True)
  assert at_stop(off) > 0.6 and at_stop(on) < 0.3, (at_stop(off), at_stop(on))

@pytest.mark.parametrize('distance', [14.0, 18.0])
@pytest.mark.parametrize('v_trigger', [1.0, 0.7, 0.5, 0.38, 0.3])
def test_a_lead_pulling_away_in_the_last_metre_is_followed(distance, v_trigger):
  off, off_go = rolling_launch(v_trigger, False, distance=distance)
  on, on_go = rolling_launch(v_trigger, True, distance=distance)
  # a plan that asks to move is never softened, and the stop bit never pulses
  assert on.held_positive == 0
  assert on.stop_bit_pulses == 0
  # the car reaches 2 m/s within a frame of StarPilot's, except when the lead leaves while the landed car is still rolling at a
  # crawl: StarPilot launches a rolling car at the MPC's jerk limit and a stopped one without it, which costs up to 0.45 s here
  assert on_go <= off_go + 0.45 + 1e-9, (off_go, on_go)


def enhanced_stop(v0, line, smooth, *, shortfall=0.0, creep=0.0, release_after=3.0, seconds=30.0):
  # a committed Enhanced Stops stop at a line, its plan as force_stop.py builds it with the default tune (speed ceiling falling to
  # the handoff 6 m before the target, the native stop bit below 0.5 m/s, the MPC obstacle), released some seconds after rest
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep)
  rested = None
  for _ in range(round(seconds / DT_MDL)):
    left = line - drive.x
    if rested is None and drive.v <= 0.0 and drive.frame > 5:
      rested = drive.frame
    model_ns = BASE_NS + drive.frame * 50_000_000 - 10_000_000
    if rested is not None and drive.frame - rested > release_after / DT_MDL:
      plan = StopPlan(model_ns)
    else:
      ceiling = math.sqrt(2.0 * 0.65 * max(left - 6.0, 0.0))
      plan = StopPlan(model_ns, ceiling, max(0.0, left * 0.93), True, ceiling <= 0.5, max(0.0, left), 0.0, 0.32, False,
                      target_m=max(0.0, left))
    drive.step(force_stop=plan, v_cruise=max(v0, 3.0))
  return drive, rested


@pytest.mark.parametrize('v0,line', [(3.0, 20.0), (5.0, 35.0), (8.0, 35.0)])
@pytest.mark.parametrize('shortfall,creep', [(0.0, 0.0), (0.2, 0.3)])
def test_an_enhanced_stops_committed_stop_is_landed_short_of_its_line(v0, line, shortfall, creep):
  off, off_rested = enhanced_stop(v0, line, False, shortfall=shortfall, creep=creep)
  on, on_rested = enhanced_stop(v0, line, True, shortfall=shortfall, creep=creep)
  # rests no further on than its target point, or than StarPilot's own stop if that passes it; peaks no harder; no pulse
  assert on.rest[1] <= max(line, off.rest[1]) + 0.1
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.stop_bit_pulses == 0
  if shortfall == 0.0 and v0 < 8.0:
    assert at_stop(on) < at_stop(off) - 0.5
  # held at rest by LongControl's stopping state, and released with StarPilot when the commitment ends
  held = [row for row in on.trace if row[0] > on.rest[0] + 0.5 and row[0] < on.rest[0] + 2.5]
  assert held and all(row[6] == LongCtrlState.stopping and row[2] <= 0.1 for row in held)
  release = [next(row[0] for row in drive.trace if row[0] > drive.rest[0] + 3.0 and not row[5]) - drive.rest[0] for drive in (off, on)]
  assert release[1] <= release[0] + DT_MDL + 1e-9, release


@pytest.mark.parametrize('shortfall,creep', [(0.0, 0.0), (0.2, 0.3)])
@pytest.mark.parametrize('v0,line', [(3.0, 15.0), (5.0, 25.0)])
def test_an_enhanced_stops_approach_never_rests_past_its_line(v0, line, shortfall, creep):
  # Enhanced Stops tracking a light it has not committed to: the MPC stops for its obstacle, and the landing never takes the car past
  # the tracked line (unless StarPilot's own stop passes it too)
  rests = []
  for smooth in (False, True):
    drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep)
    for _ in range(round(30.0 / DT_MDL)):
      left = line - drive.x
      plan = StopPlan(BASE_NS + drive.frame * 50_000_000 - 10_000_000, None, max(0.0, left * 0.93), False, False, 0.0,
                      max(left, 0.0), 0.32, False, target_m=max(left, 0.0))
      drive.step(force_stop=plan, v_cruise=max(v0, 3.0))
    rests.append(drive.rest[1])
    assert drive.stop_bit_pulses == 0
  assert rests[1] <= max(line, rests[0]) + 0.05, rests


def test_a_lead_pulling_away_does_not_take_the_car_through_the_models_stop():
  # Experimental mode at a red light: the model says stop and asks about nothing (0 to +0.1) while the lead ahead pulls away below
  # walking pace. StarPilot's stop bit comes from the model and must hold the car, with Smooth Stops as without it
  runs = []
  for smooth in (False, True):
    drive = Drive(1.2, smooth=smooth)
    lead = Lead(7.0)
    departed = None
    for i in range(round(16.0 / DT_MDL)):
      if departed is None and drive.v < 0.3 and drive.frame > 5:
        departed = i
      if departed is not None:
        lead.speed = min(lead.speed + 2.5 * DT_MDL, 5.0)
        lead.x += lead.speed * DT_MDL
      e2e = -0.6 if departed is None else 0.02 + 0.08 * ((i - departed) % 20) / 20.0
      drive.step(lead=lead, experimental=True, model_stop=True, e2e=e2e)
    after = [row for row in drive.trace if departed is not None and row[0] > departed * DT_MDL + 2.0]
    runs.append((max(row[2] for row in after), all(row[5] for row in after), drive.stop_bit_pulses))
  (off_v, off_bit, _), (on_v, on_bit, on_pulses) = runs
  assert off_bit and on_bit and on_v <= off_v + 0.02 and on_pulses == 0, runs


@pytest.mark.parametrize('bit_speed', [3.5, 2.5])
@pytest.mark.parametrize('shortfall,creep', [(0.0, 0.0), (0.2, 0.3)])
def test_an_enhanced_stops_bit_raised_at_speed_is_landed(bit_speed, shortfall, creep):
  # the road's harsh case: Enhanced Stops commits and raises the stop bit at 3.5 m/s, which hands the last metres to LongControl's
  # ramp to -2.0. With Smooth Stops the landing takes them: the bit waits for the wheel stop, braking never peaks harder, and the
  # car never rests past the line further than StarPilot's own stop does
  runs = []
  for smooth in (False, True):
    drive = Drive(bit_speed + 0.6, smooth=smooth, shortfall=shortfall, creep=creep)
    bit_at = None
    for _ in range(round(25.0 / DT_MDL)):
      left = 20.0 - drive.x
      if bit_at is None and drive.v <= bit_speed and drive.frame > 5:
        bit_at = drive.x
      plan = StopPlan(BASE_NS + drive.frame * 50_000_000 - 10_000_000, 0.0 if bit_at is not None else None, max(0.0, left * 0.93),
                      True, bit_at is not None, max(0.0, left), 0.0, 0.32, False, target_m=max(0.0, left))
      drive.step(force_stop=plan, v_cruise=bit_speed + 0.6)
    runs.append(drive)
  off, on = runs
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.stop_bit_pulses == 0
  assert on.rest[1] <= max(20.0, off.rest[1]) + 0.1
  if shortfall == 0.0:
    first_bit = next(row for row in on.trace if row[5])
    assert first_bit[2] <= 0.11 and at_stop(on) < at_stop(off) - 0.5, (first_bit, at_stop(on), at_stop(off))
  # a brake too weak to stop short of the line hands the stop back to StarPilot's own ramp: no softer, never harder


def late_commit(v0, line, smooth, commit_frame=10, seconds=14.0):
  # Enhanced Stops commits late at speed: the line is close, its plan asks 3 m/s^2 or more and StarPilot's bit rises at once
  drive = Drive(v0, smooth=smooth)
  at = None
  for _ in range(round(seconds / DT_MDL)):
    model_ns = BASE_NS + drive.frame * 50_000_000 - 10_000_000
    if drive.frame < commit_frame:
      plan = StopPlan(model_ns)
    else:
      at = drive.x + line if at is None else at
      left = at - drive.x
      ceiling = math.sqrt(2.0 * 0.65 * max(left - 6.0, 0.0))
      plan = StopPlan(model_ns, ceiling, max(0.0, left * 0.93), True, ceiling <= 0.5, max(0.0, left), 0.0, 0.32, False,
                      target_m=max(0.0, left))
    drive.step(force_stop=plan, v_cruise=max(v0, 3.0))
  return drive.rest[1] - at, drive.peaks()[0]


@pytest.mark.parametrize('v0,line', [(6.0, 8.0), (7.0, 8.0), (8.0, 8.0), (8.0, 10.0), (8.0, 14.0)])
def test_a_late_enhanced_stops_commit_at_speed_is_braked_in_full(v0, line):
  # a hard demand passes in full, StarPilot's own stopping state included, so the car never ends further past the line than off
  (off_past, off_peak), (on_past, on_peak) = (late_commit(v0, line, smooth) for smooth in (False, True))
  assert on_past <= off_past + 0.1 and on_peak >= off_peak - 0.1, (off_past, on_past, off_peak, on_peak)


def short_enhanced_stop(smooth, short, *, v0=5.6, stop_at=12.0, shortfall=0.0, creep=0.0, seconds=14.0):
  # the road's 0d 48:06 in Experimental mode: the model asks a gentle stop 1 m short of stop_at, while Enhanced Stops' countdown reads
  # its line `short` metres nearer and so raises its stop bit at about 5 m/s
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep, a0=-1.3)
  exits = []
  for _ in range(round(seconds / DT_MDL)):
    room = max(stop_at - drive.x - 1.0, 0.05)
    e2e = -min(drive.v ** 2 / (2.0 * room), 4.0) if drive.v > 0.0 else -0.3
    target = max(0.0, stop_at - drive.x - short)
    ceiling = math.sqrt(2.0 * 0.65 * max(target - 6.0, 0.0))
    plan = StopPlan(BASE_NS + drive.frame * 50_000_000 - 10_000_000, None, None, True, ceiling <= 0.5, target, 0.0, 0.32, False,
                    target_m=target)
    drive.step(experimental=True, model_stop=drive.v < 0.3, e2e=e2e, force_stop=plan)
    owner = drive.planner.smooth_stops
    if owner is not None and owner.exit_reason and (not exits or exits[-1][0] != owner.exit_reason):
      exits.append((owner.exit_reason, drive.v))
  return drive, exits


@pytest.mark.parametrize('short', [2.0, 3.0])
@pytest.mark.parametrize('shortfall,creep', [(0.0, 0.0), (0.2, 0.3)])
def test_an_enhanced_stops_line_read_short_is_not_handed_back(short, shortfall, creep):
  # round 7 handed this stop back as soon as the short line left no room ('overrun', at about 4 m/s), and LongControl's ramp to
  # -2.0 then ran to rest. Now StarPilot's own stop plus the leeway bounds it, the landing keeps it and fades to the kiss
  (off, _), (on, exits) = (short_enhanced_stop(smooth, short, shortfall=shortfall, creep=creep) for smooth in (False, True))
  bit_v = next(row[2] for row in off.trace if row[5])
  assert 4.3 <= bit_v <= 5.4, bit_v
  # kept to walking pace (a creeping car the kiss cannot hold may still be handed back as it crawls)
  assert all(v < 0.5 for _, v in exits), exits
  assert math.isfinite(on.rest[0]) and on.rest[1] <= off.rest[1] + LEEWAY + 0.05, (off.rest, on.rest)
  # never past what the model stops for (it aims 1 m short of it)
  assert on.rest[1] < 12.0
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.stop_bit_pulses == 0
  assert at_stop(on) < at_stop(off) - 0.5, (at_stop(on), at_stop(off))
  first_bit = next(row for row in on.trace if row[5])
  assert first_bit[2] <= 0.11 or shortfall > 0.0, first_bit


def model_held_hard(smooth, *, v0=6.0, shortfall=0.0, creep=0.0, seconds=10.0):
  # the road's 0d 12:35: the model asks 2.4 m/s^2 down to 1 m/s and eases only as the wheels stop (to 1.3 m/s^2 at rest)
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep, a0=-1.8)
  for _ in range(round(seconds / DT_MDL)):
    e2e = -2.4 if drive.v > 1.0 else -(1.3 + 1.1 * drive.v) if drive.v > 0.0 else -0.3
    drive.step(experimental=True, model_stop=drive.v < 0.3, e2e=e2e)
  return drive


@pytest.mark.parametrize('shortfall,creep', PLANTS)
def test_a_model_stop_held_hard_to_rest_still_fades_to_the_kiss(shortfall, creep):
  # round 7 passed an urgent demand unchanged to rest; now its last part fades at URGENT_FADE_JERK, costing well under the leeway
  off, on = (model_held_hard(smooth, shortfall=shortfall, creep=creep) for smooth in (False, True))
  assert math.isfinite(on.rest[0]) and on.rest[1] <= off.rest[1] + LEEWAY, (off.rest, on.rest)
  assert on.peaks()[0] <= off.peaks()[0] + PEAK_DECEL and on.stop_bit_pulses == 0
  assert at_stop(on) < at_stop(off) - 0.5, (at_stop(on), at_stop(off))
  # at speed the hard braking passes as asked
  fast = [(a, b) for a, b in zip(off.trace, on.trace, strict=False) if a[2] > 3.5]
  assert fast and all(abs(a[4] - b[4]) < 1e-6 for a, b in fast)


@pytest.mark.parametrize('event', ['force_decel', 'fcw'])
def test_forcedecel_or_fcw_mid_landing_hands_the_stop_back_at_once(event):
  drive = Drive(2.0, smooth=True)
  lead = Lead(12.0)
  landed = False
  for _ in range(round(10.0 / DT_MDL)):
    owner = drive.planner.smooth_stops
    landed |= owner.landing
    if landed and event == 'force_decel':
      drive.step(lead=lead, force_decel=True)
      break
    if landed and event == 'fcw':
      lead.x = drive.x + 1.0
      drive.step(lead=lead)
      if drive.planner.fcw:
        break
      continue
    drive.step(lead=lead)
  assert landed and not owner.landing and (owner.idle or event == 'fcw')
  # nothing softened in that frame: what was published is the planner's own
  assert owner.plan_target is None and drive.stop_bit_pulses == 0


@pytest.mark.parametrize('blip_speed,lead_x', [(3.0, 30.0), (2.0, 20.0)])
@pytest.mark.parametrize('frames', [1, 2])
def test_a_blip_of_starpilots_stop_bit_does_not_end_the_stop_early(blip_speed, lead_x, frames):
  # the review's case: the model's stop bit up for a frame or two early in a stop behind a distant lead
  runs = []
  for smooth in (False, True):
    drive = Drive(min(blip_speed + 0.4, 3.4), smooth=smooth)
    lead = Lead(lead_x)
    blip_at = None
    for _ in range(round(25.0 / DT_MDL)):
      if blip_at is None and drive.v < blip_speed and drive.frame > 5:
        blip_at = drive.frame
      blip = blip_at is not None and blip_at <= drive.frame < blip_at + frames
      drive.step(lead=lead, experimental=True, model_stop=blip or drive.v < 0.3, e2e=0.0)
    runs.append((lead_x - drive.rest[1], drive.peaks()[0]))
  (off_gap, off_peak), (on_gap, on_peak) = runs
  assert on_gap <= off_gap + 0.5 and on_peak <= off_peak + PEAK_DECEL, runs


def late_object_under_a_far_line(smooth, v0, trigger_x, distance, shortfall, creep, committed=True, seconds=14.0):
  # Experimental, optionally with Enhanced Stops committed to a line read right 30 m on; past trigger_x something only the model
  # sees appears distance metres ahead and the model aims 1 m short of it
  drive = Drive(v0, smooth=smooth, shortfall=shortfall, creep=creep, a0=-0.5)
  line, x_obj = 30.0, None
  for _ in range(round(seconds / DT_MDL)):
    if x_obj is None and drive.x >= trigger_x:
      x_obj = drive.x + distance
    room = max((x_obj if x_obj is not None else line) - 1.0 - drive.x, 0.05)
    e2e = -min(drive.v ** 2 / (2.0 * room), 4.0) if drive.v > 0.0 else -0.3
    plan = None
    if committed:
      target = max(0.0, line - drive.x)
      ceiling = math.sqrt(2.0 * 0.65 * max(target - 6.0, 0.0))
      plan = StopPlan(BASE_NS + drive.frame * 50_000_000 - 10_000_000, None, None, True, ceiling <= 0.5, target, 0.0, 0.32, False,
                      target_m=target)
    drive.step(experimental=True, model_stop=drive.v < 0.3, e2e=e2e, force_stop=plan)
  return drive


@pytest.mark.parametrize('v0,trigger_x,distance', [(3.4, 1.0, 4.0), (3.4, 1.0, 3.2), (4.5, 2.0, 5.0), (3.0, 0.5, 2.8)])
@pytest.mark.parametrize('shortfall,creep', PLANTS)
def test_a_late_model_stop_under_a_far_enhanced_stops_line_is_braked_as_without_it(v0, trigger_x, distance, shortfall, creep):
  # the far line bounds the rest point, so the model's own point bounds an urgent fade: a commitment to a line further on never
  # lets the car roll further toward what the model stops for than the same stop without it, and the soft end costs at most the
  # fade's own few centimetres against the stop without Smooth Stops
  off = late_object_under_a_far_line(False, v0, trigger_x, distance, shortfall, creep)
  free = late_object_under_a_far_line(True, v0, trigger_x, distance, shortfall, creep, committed=False)
  on = late_object_under_a_far_line(True, v0, trigger_x, distance, shortfall, creep)
  assert on.rest[1] <= free.rest[1] + 0.05 and on.rest[1] <= off.rest[1] + 0.25, (off.rest[1], free.rest[1], on.rest[1])
  assert on.stop_bit_pulses == 0


@pytest.mark.parametrize('scene', ['force_decel', 'fcw'])
def test_urgent_owners_keep_their_own_stop(scene):
  # forceDecel and FCW bypass Smooth Stops: the drive is the same frame for frame
  runs = []
  for smooth in (False, True):
    drive = Drive(3.0 if scene != 'fcw' else 4.0, smooth=smooth)
    for _ in range(round(8.0 / DT_MDL)):
      if scene == 'force_decel':
        drive.step(lead=Lead(20.0), force_decel=True)
      else:
        drive.step(lead=Lead(3.0))
    runs.append(drive.trace)
  assert runs[0] == runs[1]


def test_the_vision_stop_buffer_still_stops_for_a_close_vision_lead():
  # a vision-only lead inside the buffer: the buffer still engages and the stop stays within the leeway of StarPilot's own
  runs = []
  for smooth in (False, True):
    drive, least, rest = stopped_lead(2.0, 6.0, smooth, radar=False, seconds=10.0)
    assert drive.planner.vision_stop_buffer.hold_until_ns > 0 and drive.stop_bit_pulses == 0
    runs.append(rest)
  assert runs[0] - runs[1] <= LEEWAY + 0.05


def test_the_stop_bit_rises_at_the_wheel_stop():
  drive, _, _ = stopped_lead(2.0, 12.0, True)
  t_rest, _ = drive.rest
  rises = [row for row in drive.trace if row[5]]
  assert rises and rises[0][2] <= STANDSTILL_SPEED + 0.01
  holds = [row[0] for row in drive.trace if row[6] == LongCtrlState.stopping]
  assert holds and holds[0] <= t_rest + 0.25
  # and nothing rolls again while the lead stays
  assert max(row[2] for row in drive.trace if row[0] > t_rest) == 0.0


@pytest.mark.parametrize('shortfall,creep', [(0.1, 0.25), (0.2, 0.3), (0.3, 0.6)])
def test_a_weak_brake_against_creep_still_comes_to_rest(shortfall, creep):
  off, _, off_rest = stopped_lead(1.5, 8.0, False, shortfall=shortfall, creep=creep)
  on, _, on_rest = stopped_lead(1.5, 8.0, True, shortfall=shortfall, creep=creep)
  assert math.isfinite(on.rest[0]) and on.rest[0] - off.rest[0] <= EXTRA_TIME and off_rest - on_rest <= LEEWAY + 0.05


def test_a_launch_from_rest_is_not_delayed():
  launches = []
  for smooth in (False, True):
    drive = Drive(2.0, smooth=smooth)
    lead = Lead(9.0)
    for i in range(round(16.0 / DT_MDL)):
      if i * DT_MDL > 10.0:
        lead.speed = min(lead.speed + 1.5 * DT_MDL, 5.0)
        lead.x += lead.speed * DT_MDL
      drive.step(lead=lead)
    released = [row[0] for row in drive.trace if row[0] > 10.0 and not row[5]]
    moving = [row[0] for row in drive.trace if row[0] > 10.0 and row[2] > 0.0]
    launches.append((released[0], moving[0]))
    assert drive.stop_bit_pulses == 0
  assert launches[1][0] <= launches[0][0] + DT_MDL + 1e-9 and launches[1][1] <= launches[0][1] + DT_MDL + 1e-9, launches


def test_off_is_the_planner_without_smooth_stops():
  # the planner with its Smooth Stops owner and the preference off against the same planner with no owner at all
  runs = []
  for owner in (True, False):
    drive = Drive(2.0, smooth=False)
    if not owner:
      drive.planner.smooth_stops = None
    lead = Lead(9.0)
    for i in range(round(14.0 / DT_MDL)):
      if i * DT_MDL > 9.0:
        lead.speed = min(lead.speed + 1.5 * DT_MDL, 5.0)
        lead.x += lead.speed * DT_MDL
      drive.step(lead=lead, experimental=i % 40 > 20, model_stop=drive.v < 0.3, e2e=-0.4)
    runs.append(drive.trace)
  assert runs[0] == runs[1]


def test_unsupported_cars_ignore_the_preference():
  from opendbc.car.honda.interface import CarInterface as HondaInterface
  from opendbc.car.honda.values import CAR as HONDA
  planner = LongitudinalPlanner(HondaInterface.get_non_essential_params(HONDA.HONDA_CIVIC))
  assert planner.smooth_stops is None
