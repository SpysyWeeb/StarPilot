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
from openpilot.starpilot.longitudinal.smooth_stops import LEEWAY, STANDSTILL_SPEED
from openpilot.starpilot.longitudinal.tests.test_cruise_ceiling import messages

BASE_NS = 10_000_000_000
DRIVE_NS = BASE_NS - 1_000_000_000
SERVICES = ('radarState', 'controlsState', 'selfdriveState', 'carState', 'vehicleParameters', 'carControl', 'modelV2')
# the plants the review asked for: a brake as asked, one 0.1 short with creep torque, one 0.2 short with more creep
PLANTS = ((0.0, 0.0), (0.1, 0.25), (0.2, 0.3))
# the worst extra time to rest a landing costs on these drives, measured 1.4 s on the stopped-lead grid and 1.65 s for the model's
# stops; an unbounded crawl takes several seconds more
EXTRA_TIME = 2.0


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
    # a stop bit that rises for a single frame
    bits = [row[5] for row in self.trace]
    return sum(1 for a, b, c in zip(bits, bits[1:], bits[2:], strict=False) if b and not a and not c)


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
