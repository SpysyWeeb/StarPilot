import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.starpilot.longitudinal.smooth_stops import (CREEP_FADE_SPEED, CREEP_SPEED, KISS_DECEL, KISS_SPEED, LANDING_SPEED,
                                                           LEAD_ABSENCE_TIME, LEAD_STANDING_GAP, LEEWAY, STALL_MAX,
                                                           STANDSTILL_SPEED, URGENT_DECEL, URGENT_TIME, Lead, SmoothStops,
                                                           corridor_distance, landing_bound, landing_floor, owner_for)
from openpilot.starpilot.longitudinal.smooth_stops_preferences import KEY, SmoothStopsPreferences, supported

NOW = 100_000_000_000
NO_LEADS = (Lead(), Lead())
# the planner's action_t on the Palisade: the car's actuator delay plus one model frame
ACTION_T = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE).longitudinalActuatorDelay + DT_MDL
SOURCE = log.LongitudinalPlan.LongitudinalPlanSource


def leads(distance, speed=0.0, acceleration=0.0):
  return Lead(True, distance, speed, acceleration), Lead()


def frames(seconds):
  return round(seconds / DT_MDL)


def step(law, a_target, v_ego, leads=NO_LEADS, intent=True, stopping=False, a_ego=None, standstill=False, model=False,
         committed=False, stop_target=None, going=None):
  going = a_target > 0.0 if going is None else going
  return law.update(a_target, v_ego, leads, intent, stopping, a_ego=a_ego, standstill=standstill, model=model, committed=committed,
                    stop_target=stop_target, going=going)


def out(law, a_target, v_ego, leads=NO_LEADS, intent=True, a_ego=None):
  # StarPilot's own stop bit down and no model stop point: only the corridor and the leads act
  return step(law, a_target, v_ego, leads, intent, a_ego=a_ego)[0]


def landing(v_ego=1.5, a_target=-1.0, action_t=ACTION_T):
  law = SmoothStops(DT_MDL, action_t)
  step(law, a_target, v_ego)
  assert law.landing
  return law


def corridor_ride(edge_fn, v0, h=1e-4):
  # a car riding a corridor edge exactly, v' = -edge(v), integrated forward in time from v0
  ts, vs = [0.0], [v0]
  while vs[-1] > KISS_SPEED - 0.1:
    vs.append(vs[-1] - h * edge_fn(vs[-1]))
    ts.append(ts[-1] + h)
  return np.array(ts), np.array(vs)


class TestCorridor:
  def test_the_bound_only_removes_braking_and_the_floor_only_adds_it_up_to_itself(self):
    for v_ego in np.linspace(LANDING_SPEED - 0.01, STANDSTILL_SPEED + 0.01, 36):
      for a_target in np.linspace(-2.5, -0.05, 18):
        law = landing(v_ego, -1.0)
        result = out(law, a_target, v_ego)
        if not law.landing:
          assert law.exit_reason == 'urgent' or result == a_target
          continue
        assert result >= min(a_target, -law.floor(v_ego)) - 1e-9, (v_ego, a_target, result)
        assert result <= a_target + 1e-9 or math.isclose(result, -law.bound(v_ego), abs_tol=1e-9), (v_ego, a_target, result)

  def test_without_a_delay_the_edges_are_the_car_side_corridor(self):
    law = SmoothStops(DT_MDL, 0.0)
    for v_ego in np.linspace(LANDING_SPEED - 0.01, 0.0, 50):
      assert math.isclose(law.bound(v_ego), landing_bound(v_ego), abs_tol=1e-9)
      assert math.isclose(law.floor(v_ego), landing_floor(v_ego), abs_tol=1e-9)

  def test_each_edge_is_requested_one_actuator_delay_ahead_of_the_car(self):
    # the request made at a speed is what a car riding the corridor carries action_t later
    law = SmoothStops(DT_MDL, ACTION_T)
    for edge_fn, request_fn, v0, breakpoints in ((landing_bound, law.bound, 1.8, (0.9, KISS_SPEED)),
                                                 (landing_floor, law.floor, 1.4, (CREEP_SPEED, KISS_SPEED))):
      ts, vs = corridor_ride(edge_fn, v0)
      for v_car in breakpoints:
        t_car = np.interp(-v_car, -vs, ts)
        v_request = float(np.interp(t_car - ACTION_T, ts, vs))
        assert v_request > v_car + 0.05
        assert math.isclose(request_fn(v_request), edge_fn(v_car), abs_tol=2e-3), (edge_fn.__name__, v_car)
    assert math.isclose(law.bound(LANDING_SPEED), landing_bound(LANDING_SPEED))

  def test_a_plan_inside_the_corridor_passes_untouched(self):
    law = landing()
    assert out(law, -2.0, 3.0) == -2.0
    assert out(law, -0.5, 1.5) == -0.5
    assert out(law, -1.1, 1.5) == -law.bound(1.5)

  def test_a_landing_only_eases_unless_the_car_is_not_delivering(self):
    law = landing()
    assert out(law, -0.05, 0.8, a_ego=-0.05) == -0.05 and out(law, -0.05, 2.0) == -0.05
    rng = np.random.default_rng(1)
    for v_ego, a_target in zip(rng.uniform(0.2, LANDING_SPEED - 0.01, 500), rng.uniform(-1.9, -0.2, 500), strict=True):
      law = landing(max(v_ego, 0.5), -1.0)
      assert step(law, a_target, v_ego, a_ego=a_target)[0] >= a_target - 1e-9
    # a car still not slowing under a plan that asks it to (creep, a weak brake) is pressed beyond the plan
    law = landing(0.8, -0.3)
    for _ in range(frames(0.25)):
      result, _ = step(law, -0.3, 0.8, a_ego=0.1)
    assert result < -0.3 and law.integral > 0.0

  def test_both_edges_meet_at_the_kiss_by_walking_pace_and_the_plan_passes_at_rest(self):
    law = landing()
    for v_ego in (KISS_SPEED, 0.2, 0.12):
      assert math.isclose(out(law, -0.3, v_ego), -KISS_DECEL, abs_tol=1e-9)
      # a plan already gentler than the kiss passes as it is
      assert out(law, -0.02, v_ego) == -0.02
    # the wheels stop under the kiss; once the stop bit is up at rest, LongControl's stopping state holds the car and the plan
    # passes as it is
    assert step(law, -0.3, STANDSTILL_SPEED, stopping=True) == (-KISS_DECEL, True)
    for a_target in (-0.3, -0.02, 0.0):
      assert step(law, a_target, 0.0, stopping=True) == (a_target, True)
    assert law.landing

  def test_a_landing_starts_only_below_landing_speed_on_a_braking_plan_that_means_to_stop(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    assert out(law, -2.0, LANDING_SPEED + 0.1) == -2.0 and not law.landing
    assert out(law, -2.0, 1.5, intent=False) == -2.0 and not law.landing
    # StarPilot's own stop bit, once published, is never withdrawn by a landing starting late
    assert step(law, -0.1, 1.0, stopping=True) == (-0.1, True) and not law.landing
    assert step(law, -1.0, 0.9, stopping=True) == (-1.0, True) and not law.landing
    law = SmoothStops(DT_MDL, ACTION_T)
    assert out(law, 0.5, 1.0) == 0.5 and not law.landing
    assert out(law, -KISS_DECEL / 2.0, 0.2) == -KISS_DECEL / 2.0 and not law.landing
    assert out(law, -1.0, 1.5) < -0.9 and law.landing

  def test_a_landing_never_asks_for_acceleration(self):
    rng = np.random.default_rng(0)
    for v_ego, a_target in zip(rng.uniform(0.0, LANDING_SPEED - 0.01, 3000), rng.uniform(-3.5, 1.5, 3000), strict=True):
      law = landing(max(v_ego, 0.5), -1.0)
      result, _ = step(law, a_target, v_ego, stopping=True)
      if result > 0.0:
        assert result == a_target and not law.landing


class TestHandBack:
  def test_a_plan_asking_to_move_hands_back_at_once_at_any_speed(self):
    # the review's case: a lead pulling away while the car rolls below walking pace
    for v_ego in (2.0, 0.6, 0.35, 0.05, 0.0):
      law = landing(1.0, -0.6)
      assert step(law, 0.8, v_ego, stopping=False) == (0.8, False)
      assert not law.landing and law.exit_reason == 'launch'
      assert step(law, 0.8, v_ego, stopping=True) == (0.8, True)

  def test_a_plan_that_no_longer_means_to_stop_hands_back(self):
    law = landing(1.0, -0.6)
    assert step(law, -0.4, 0.9, intent=False) == (-0.4, False)
    assert law.exit_reason == 'go'

  def test_a_sudden_hard_demand_hands_back(self):
    law = landing(1.0, -0.6)
    urgent = -(URGENT_DECEL + 0.1)
    for _ in range(max(round(URGENT_TIME / DT_MDL), 1) - 1):
      assert step(law, urgent, 0.9)[0] == -law.bound(0.9) and law.landing
    assert step(law, urgent, 0.9)[0] == urgent and law.exit_reason == 'urgent'
    # a single hard frame between ordinary ones does not
    law = landing(1.0, -0.6)
    for a_target in (urgent, -0.5, urgent, -0.5):
      step(law, a_target, 0.9)
    assert law.landing or URGENT_TIME <= DT_MDL
  def test_a_lead_the_kiss_cannot_stop_short_of_hands_back(self):
    law = landing(0.8, -0.6)
    assert out(law, -0.6, 0.8, leads(6.0)) > -0.6 and law.landing
    near = leads(LEAD_STANDING_GAP + 0.2)
    assert law.lead_requirement(0.8, near[0]) > law.bound(0.8)
    assert step(law, -0.6, 0.8, near, stopping=True) == (-0.6, True) and law.exit_reason == 'lead'

  def test_a_dropped_lead_keeps_its_requirement_for_the_absence_window(self):
    law = landing(1.0, -0.6)
    out(law, -0.6, 1.0, leads(8.0))
    kept = law.required_decel
    assert kept > 0.0
    for _ in range(round(LEAD_ABSENCE_TIME / DT_MDL) - 1):
      out(law, -0.6, 1.0)
      assert law.required_decel == kept
    out(law, -0.6, 1.0)
    assert law.required_decel == 0.0 and law.landing

  @pytest.mark.parametrize('speeds', [(0.45,), (0.36, 0.42), (1.0,)])
  @pytest.mark.parametrize('a_ego', [0.0, -0.2])
  def test_a_car_that_keeps_rolling_past_starpilots_own_stop_point_hands_back(self, speeds, a_ego):
    # the review's cases: a car stuck at 0.45 m/s, or alternating either side of walking pace, with StarPilot's bit up; without the
    # stop point fixed to the road they would crawl on under the kiss for as long as the car does
    law = landing(1.0, -0.6)
    for i in range(frames(20.0)):
      v_ego = speeds[(i // 10) % len(speeds)]
      result, bit = step(law, -0.6, v_ego, stopping=True, a_ego=a_ego)
      if not law.landing:
        break
      assert bit is False
    assert not law.landing and law.exit_reason in ('overrun', 'stall') and bit is True
    assert law.travelled <= law.ramp_anchor + LEEWAY + 1e-9 and i * DT_MDL < 4.0

  def test_a_model_stop_rests_within_the_leeway_of_its_point_on_a_weak_brake(self):
    # a car that answers one actuator delay late, delivers 70 % of what it is asked and creeps: the model asks to stop 6 m ahead
    for delivered, creep in ((1.0, 0.0), (0.7, 0.0), (0.7, 0.25)):
      law = SmoothStops(DT_MDL, ACTION_T)
      x, v, a = 0.0, 2.0, 0.0
      pipe = [0.0] * frames(ACTION_T)
      for _ in range(frames(30.0)):
        plan = -min(v * v / (2.0 * max(6.0 - x, 0.05)), 4.0) if v > 0.0 else -0.3
        request, _ = step(law, plan, v, stopping=v < 0.3, a_ego=a, model=True)
        pipe.append(request)
        request = pipe.pop(0)
        a = delivered * request + (creep if v < 1.5 and request > -0.3 else 0.0)
        if v <= 0.0 and a <= 0.0:
          break
        x, v = x + v * DT_MDL, max(v + a * DT_MDL, 0.0)
      assert v == 0.0 and x <= 6.0 + LEEWAY + 0.05, (delivered, creep, x)
  def test_a_car_that_keeps_not_slowing_hands_back(self):
    law = landing(0.8, -0.3)
    for _ in range(frames(10.0)):
      step(law, -0.3, 0.8, a_ego=0.2)
      if not law.landing:
        break
    assert law.exit_reason == 'stall' and law.integral >= STALL_MAX - 1e-9

  def test_a_stop_handed_back_is_starpilots_until_the_plan_no_longer_means_to_stop(self):
    law = landing(1.0, -0.6)
    step(law, 0.3, 0.5)
    assert step(law, -0.6, 0.4) == (-0.6, False) and not law.landing
    assert step(law, -0.6, 0.4, intent=False) == (-0.6, False) and not law.landing
    out(law, -0.6, 0.4)
    assert law.landing

  def test_nonfinite_inputs(self):
    law = landing()
    assert step(law, -2.0, math.nan, stopping=True) == (-2.0, True) and not law.landing
    law = landing()
    assert math.isnan(out(law, math.nan, 1.0)) and law.landing


class TestLeeway:
  def test_the_corridor_alone_is_left_alone_while_its_rest_point_fits(self):
    law = landing(1.5, -1.0)
    assert law.overrun_requirement(1.5, corridor_distance(1.5) - LEEWAY + 1.5 * ACTION_T + 0.01) == 0.0
    # past it, the speed is shed before the kiss, then at the corridor's bound, then the plain stop in the room left
    need = law.overrun_requirement(1.5, corridor_distance(1.5) - LEEWAY + 1.5 * ACTION_T - 0.2)
    assert 0.0 < need <= law.bound(1.5)
    assert law.overrun_requirement(1.5, 0.3) >= 1.5 * 1.5 / (2.0 * (0.3 + LEEWAY - 1.5 * ACTION_T)) - 1e-9
    assert law.overrun_requirement(1.5, -LEEWAY) == math.inf

  def test_starpilots_own_stop_point_is_fixed_to_the_road_once_its_bit_has_stayed_up(self):
    law = landing(1.0, -0.6)
    for _ in range(law.anchor_frames - 1):
      step(law, -0.6, 0.9, stopping=True, a_ego=-0.6)
      assert law.ramp_anchor is None and law.ramp_request is None
    step(law, -0.6, 0.9, stopping=True, a_ego=-0.6)
    anchor = law.ramp_anchor
    assert anchor is not None and 0.0 < anchor - law.travelled < 2.0 and law.ramp_request < -0.6
    for _ in range(frames(1.0)):
      step(law, -0.6, 0.5, stopping=True, a_ego=-0.6)
    assert law.ramp_anchor == anchor and law.travelled > 0.5
    # the bit dropping clears it; a blip of the bit never sets it
    step(law, -0.6, 0.5, stopping=False, a_ego=-0.6)
    assert law.ramp_anchor is None and law.ramp_request is None
    for _ in range(law.anchor_frames - 1):
      step(law, -0.6, 0.5, stopping=True, a_ego=-0.6)
    step(law, -0.6, 0.5, stopping=False, a_ego=-0.6)
    assert law.ramp_anchor is None and law.landing

  def test_an_enhanced_stops_commitment_uses_its_own_stop_point(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    step(law, -0.4, 2.0, committed=True, stop_target=6.0)
    assert law.landing and law.plan_stop(2.0, -0.4, False) == 6.0 - LEEWAY
    # where the soft tail no longer fits before that point the landing eases less, never more braking than StarPilot's own
    for _ in range(law.anchor_frames + 2):
      result, bit = step(law, -0.4, 2.0, stopping=True, committed=True, stop_target=1.5, a_ego=-0.4)
    assert bit is False and law.ramp_request - 1e-9 <= result <= -0.4


class TestStopBit:
  def test_the_stop_bit_is_held_back_while_the_kiss_lands_a_rolling_car(self):
    law = landing(0.6, -0.4)
    for v_ego in np.linspace(0.6, STANDSTILL_SPEED + 0.01, 10):
      assert step(law, -0.4, v_ego, stopping=False)[1] is False
    assert step(law, -0.4, STANDSTILL_SPEED, stopping=False) == (-KISS_DECEL, True)
    law = landing(0.6, -0.4)
    assert step(law, -0.4, 0.3, stopping=True, standstill=True)[1] is True

  def test_once_risen_the_bit_stays_through_wheel_speed_ticks_at_rest(self):
    law = landing(0.5, -0.4)
    assert step(law, -0.4, 0.0, stopping=True)[1] is True
    for v_ego in (0.12, 0.0, 0.2, 0.0, 0.15):
      assert step(law, -0.4, v_ego, stopping=False)[1] is True
    assert law.landing

  def test_the_bit_never_pulses_for_a_single_frame(self):
    # rolling to rest, then the plan easing and asking to move: the bit rises once and falls once
    law = landing(0.6, -0.4)
    bits = []
    for v_ego, a_target, stopping in ((0.3, -0.3, True), (0.15, -0.3, True), (0.0, -0.2, True), (0.0, -0.05, False),
                                      (0.0, 0.0, False), (0.0, 0.3, False), (0.05, 0.6, False)):
      bits.append(step(law, a_target, v_ego, stopping=stopping)[1])
    assert bits == [False, False, True, True, True, False, False]

  def test_a_launch_hand_back_does_not_publish_the_held_bit_while_the_plan_moves(self):
    law = landing(0.4, -0.4)
    assert step(law, -0.4, 0.3, stopping=True)[1] is False
    bits = [step(law, a_target, 0.25, stopping=True)[1] for a_target in (0.3, 0.4, 0.5)]
    assert bits == [False, False, False] and law.exit_reason == 'launch'
    # the plan no longer asking to move with StarPilot's bit still up: its bit is published, and stays
    assert step(law, -0.05, 0.25, stopping=True)[1] is True and step(law, 0.1, 0.25, stopping=True)[1] is True
    # a plan barely positive, not going, under StarPilot's bit is still StarPilot's stop: the landing eases it on to rest
    law = landing(0.4, -0.4)
    step(law, -0.4, 0.3, stopping=True)
    result, bit = step(law, 0.05, 0.25, stopping=True, going=False)
    assert law.landing and result <= 0.0 and bit is False

  def test_a_published_bit_is_never_withdrawn_across_a_reset(self):
    # forceDecel, FCW, the toggle or longActive dropping for a frame resets the landing; StarPilot's bit published meanwhile stays
    law = SmoothStops(DT_MDL, ACTION_T)
    sm = {'carState': NS(vEgo=0.8, aEgo=-0.5, standstill=False),
          'radarState': NS(leadOne=NS(present=False), leadTwo=NS(present=False))}
    kwargs = {'enabled': True, 'plan_speeds': [1.0, 0.5, 0.0], 'target_source': SOURCE.lead0, 'forcing': False}
    assert law.sample(sm, target=-0.8, stopping=False, active=True, **kwargs) == (pytest.approx(-law.bound(0.8)), False)
    assert law.landing
    assert law.sample(sm, target=-0.8, stopping=True, active=False, **kwargs) == (-0.8, True) and not law.landing
    for _ in range(3):
      assert law.sample(sm, target=-0.8, stopping=True, active=True, **kwargs) == (-0.8, True)
    assert not law.landing

  def test_outside_a_landing_the_target_and_bit_are_starpilots(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    assert step(law, 0.3, 0.2, stopping=True) == (0.3, True) and not law.landing
    assert step(law, -1.0, 5.0, stopping=False) == (-1.0, False)


class TestLead:
  def test_the_output_is_continuous_as_a_stopped_lead_closes(self):
    law = landing(2.0, -1.5)
    outputs = []
    for d in np.arange(12.0, 1.0, -0.001):
      result = out(law, -1.5, 2.0, leads(d))
      if not law.landing:
        break
      outputs.append(result)
    assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.01

  def test_a_lead_gets_what_stops_the_car_the_standing_gap_short_of_it_after_the_delay(self):
    # (the stop point from StarPilot's own stop is not involved: its bit is down)
    for v_ego, obstacle, plan in ((0.37, (0.32,), -0.6), (1.0, (1.0,), -2.0), (0.7, (0.6,), -1.5), (1.5, (1.1,), -3.5),
                                  (3.0, (1.0, 3.0, -4.0), -3.5), (0.5, (1.2,), -1.0), (0.5, (0.8,), -1.0), (2.0, (3.0,), -2.0)):
      law = landing(v_ego, -1.0)
      distance, speed, acceleration = (*obstacle, 0.0, 0.0)[:3]
      rest = distance + (speed * speed / (-2.0 * acceleration) if acceleration < 0.0 else 0.0)
      room = rest - v_ego * ACTION_T - LEAD_STANDING_GAP
      needs = v_ego * v_ego / (2.0 * room) if room > 0.0 else math.inf
      result = out(law, plan, v_ego, leads(*obstacle), a_ego=-0.5)
      assert -result >= min(needs, -plan) - 1e-9, (v_ego, obstacle, result)


class TestAntiStall:
  def test_a_car_not_slowing_presses_both_edges(self):
    law = landing(0.3, -0.3)
    kiss = [out(law, -0.3, 0.3, a_ego=0.1) for _ in range(frames(0.25))]
    assert kiss[0] < -KISS_DECEL
    assert all(b < a for a, b in zip(kiss, kiss[1:], strict=False))

  def test_the_press_drains_as_the_car_slows_again(self):
    law = landing(0.3, -0.3)
    for _ in range(frames(0.25)):
      pressed = out(law, -0.3, 0.3, a_ego=0.1)
    drained = [out(law, -0.3, 0.3, a_ego=-0.6) for _ in range(frames(2.0))]
    assert drained[0] > pressed
    assert all(b >= a for a, b in zip(drained, drained[1:], strict=False))
    assert drained[-1] == -KISS_DECEL

  def test_nothing_is_pressed_at_rest_or_where_the_floor_has_faded(self):
    for v_ego in (0.0, 0.05, STANDSTILL_SPEED):
      law = landing(0.3, -0.3)
      for _ in range(frames(0.25)):
        out(law, -0.3, 0.3, a_ego=0.1)
      assert law.integral > 0.1
      assert out(law, -0.3, v_ego, a_ego=0.2) == -KISS_DECEL
    law = landing(3.4, -1.0)
    for _ in range(frames(2.0)):
      out(law, -0.4, CREEP_FADE_SPEED + 0.5, a_ego=1.0)
      assert law.integral == 0.0


class TestPlannerHook:
  def sm(self, v_ego=1.0, a_ego=-0.5, standstill=False, lead=8.0):
    def radar(present, d=0.0):
      return NS(present=present, dRel=d, vLeadK=0.0, aLeadK=0.0)
    return {'carState': NS(vEgo=v_ego, aEgo=a_ego, standstill=standstill),
            'radarState': NS(leadOne=radar(lead is not None, lead or 0.0), leadTwo=radar(False))}

  def sample(self, law, sm, target, *, enabled=True, active=True, source=SOURCE.lead0, end=0.0, stopping=False, forcing=False):
    return law.sample(sm, target=target, stopping=stopping, enabled=enabled, active=active, plan_speeds=[1.0, 0.5, end],
                      target_source=source, forcing=forcing)

  def test_disabled_inactive_or_idle_returns_the_planner_values_themselves(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    target = np.float64(-1.0)
    for enabled, active in ((False, True), (True, False), (False, False)):
      result = self.sample(law, self.sm(), target, enabled=enabled, active=active, stopping=True)
      assert result[0] is target and result[1] is True
    assert law.idle and law.plan_seed(target) is target
    # a plan that does not mean to stop: nothing to land
    result = self.sample(law, self.sm(), target, source=SOURCE.cruise, end=2.0)
    assert result[0] is target and not law.landing

  @pytest.mark.parametrize('kwargs', [{}, {'lead': None}, {'source': SOURCE.cruise, 'lead': None}])
  def test_every_stop_is_landed_and_the_mpc_continues_the_unsoftened_plan(self, kwargs):
    law = SmoothStops(DT_MDL, ACTION_T)
    source = kwargs.pop('source', SOURCE.lead0)
    target, bit = self.sample(law, self.sm(v_ego=0.6, **kwargs), -0.9, source=source)
    assert law.landing and bit is False and target > -0.9
    assert law.plan_seed(target) == -0.9

  def test_the_models_stop_and_an_enhanced_stops_commitment_are_landed(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    # the model's request held stops the car within a few seconds, though the MPC's plan does not
    self.sample(law, self.sm(v_ego=0.8, lead=None), -0.5, source=SOURCE.e2e, end=2.0)
    assert law.landing and math.isfinite(law.model_anchor)
    law = SmoothStops(DT_MDL, ACTION_T)
    self.sample(law, self.sm(v_ego=0.8, lead=None), -0.05, source=SOURCE.e2e, end=2.0)
    assert not law.landing
    # Enhanced Stops hands its last metres to StarPilot's stop bit: the landing takes them from the frame it rises, never asks to
    # accelerate while committed, and keeps the stop within the leeway of where StarPilot's own ramp would have ended it
    law = SmoothStops(DT_MDL, ACTION_T)
    target, bit = law.sample(self.sm(v_ego=1.2, lead=None), target=0.3, stopping=True, enabled=True, active=True, plan_speeds=[1.0, 2.0],
                             target_source=SOURCE.cruise, forcing=True, stop_target=5.0)
    assert law.landing and bit is False and target <= 0.0 and law.stop_target == 5.0
  def test_turning_it_off_mid_landing_hands_the_plan_back(self):
    law = SmoothStops(DT_MDL, ACTION_T)
    assert self.sample(law, self.sm(), -0.9)[1] is False and law.landing
    assert self.sample(law, self.sm(), -0.9, active=False, stopping=True) == (-0.9, True)
    assert law.idle and law.plan_seed(-0.2) == -0.2

  @pytest.mark.parametrize('model_stop', [True, False])
  def test_the_models_stop_keeps_starpilots_bit_when_a_lead_pulls_away(self, model_stop):
    # Experimental mode, rolling below walking pace with StarPilot's bit held back: a lead pulls away (the MPC's plan is under way)
    # and the request turns slightly positive. If the model says stop (a red light), the stop goes on: nothing asks to accelerate
    # and the bit rises as the wheels stop. Otherwise the car follows the lead, StarPilot's held bit unpublished
    law = SmoothStops(DT_MDL, ACTION_T)
    sm = self.sm(v_ego=0.25, a_ego=-0.2)
    sm['selfdriveState'] = NS(experimentalMode=True)
    sm['modelV2'] = NS(action=NS(shouldStop=model_stop))
    kwargs = {'enabled': True, 'active': True, 'target_source': SOURCE.e2e, 'forcing': False}
    assert law.sample(sm, target=-0.5, stopping=True, plan_speeds=[0.25, 0.0], **kwargs)[1] is False and law.landing
    published = [law.sample(sm, target=0.05, stopping=True, plan_speeds=[0.25, 3.0], **kwargs) for _ in range(4)]
    if model_stop:
      assert law.landing and all(target <= 0.0 and bit is False for target, bit in published)
      sm['carState'] = NS(vEgo=0.0, aEgo=0.0, standstill=True)
      assert law.sample(sm, target=0.05, stopping=True, plan_speeds=[0.25, 3.0], **kwargs)[1] is True
    else:
      assert not law.landing and published == [(0.05, False)] * 4

  def test_the_planners_effective_delay_moves_the_corridor_edges(self):
    # testing's developer delay override reaches the landing through the planner's own action_t
    law = SmoothStops(DT_MDL, ACTION_T)
    before = list(law.bound_bp)
    law.sample(self.sm(), target=-0.9, stopping=False, enabled=True, active=True, plan_speeds=[1.0, 0.0], target_source=SOURCE.lead0,
               forcing=False, action_t=ACTION_T + 0.2)
    assert law.action_t == ACTION_T + 0.2 and law.bound_bp[0] > before[0]

  def test_owner_only_for_supported_cars_with_starpilot_longitudinal(self):
    cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
    cp.openpilotLongitudinalControl, cp.pcmCruise = True, False
    law = owner_for(cp, DT_MDL)
    assert law is not None and math.isclose(law.action_t, ACTION_T) and law.stop_accel == cp.stopAccel
    assert math.isclose(law.urgent_frames * DT_MDL, max(URGENT_TIME, DT_MDL))
    cp.openpilotLongitudinalControl, cp.pcmCruise = False, True
    assert owner_for(cp, DT_MDL) is None
    assert owner_for(NS(), DT_MDL) is None


@pytest.mark.parametrize('acceleration', [0.0, -1.0])
def test_lead_from_radar_reads_finite_filtered_values(acceleration):
  lead = Lead.from_radar(NS(present=True, dRel=-0.2, vLeadK=-0.1, aLeadK=acceleration))
  assert lead == Lead(True, 0.0, 0.0, acceleration)
  assert Lead.from_radar(NS(present=True, dRel=math.nan, vLeadK=0.0, aLeadK=0.0)) == Lead()
  assert Lead.from_radar(NS(present=False)) == Lead()
  assert Lead.from_radar(None) == Lead()


def test_output_never_goes_below_accel_min():
  law = landing(3.4, -1.0)
  assert out(law, ACCEL_MIN, 3.4, leads(0.0)) == ACCEL_MIN


@pytest.mark.parametrize('raw', [None, b'0', b'1', b'2', b'true', b'1\n', b'1'*17])
def test_saved_preference_default_off_corruption_and_safe_mode_never_rewrite(tmp_path, raw):
  params = NS(get_param_path=lambda key: str(tmp_path/key))
  path = tmp_path/KEY
  if raw is not None:
    path.write_bytes(raw)
  host = SmoothStopsPreferences(params)
  assert host.sample(NOW) == (raw == b'1')
  assert (path.read_bytes() if path.exists() else None) == raw
  (tmp_path/'SafeMode').write_bytes(b'1')
  assert host.sample(NOW + 1_000_000_000) is False
  assert (path.read_bytes() if path.exists() else None) == raw


def test_saved_preference_is_read_about_once_a_second(tmp_path):
  params = NS(get_param_path=lambda key: str(tmp_path/key))
  host = SmoothStopsPreferences(params)
  assert host.sample(NOW) is False
  (tmp_path/KEY).write_bytes(b'1')
  assert host.sample(NOW + 999_000_000) is False
  assert host.sample(NOW + 1_000_000_000) is True


@pytest.mark.parametrize('fingerprint,expected', [
  (CAR.HYUNDAI_PALISADE, True), (CAR.HYUNDAI_SONATA, True), (CAR.KIA_STINGER, True), (CAR.GENESIS_G70_2020, True),
  # electrified (regenerative braking at kiss speeds), CAN FD, the G90's own hold shaping, blended longitudinal, the Ray's
  # pedal, and cars without openpilot longitudinal
  (CAR.HYUNDAI_KONA_EV, False), (CAR.HYUNDAI_SONATA_HYBRID, False), (CAR.KIA_NIRO_PHEV_2022, False), (CAR.HYUNDAI_NEXO_1ST_GEN, False),
  (CAR.HYUNDAI_IONIQ_5, False), (CAR.KIA_EV9, False), (CAR.GENESIS_G90, False), (CAR.HYUNDAI_PALISADE_2023, False),
  (CAR.KIA_RAY_EV, False), (CAR.HYUNDAI_KONA_NON_SCC, False), (CAR.GENESIS_G70, False),
  # camera SCC: the SCC messages come from the camera's own controller, not the radar's this was measured on
  (CAR.HYUNDAI_ELANTRA_2024, False),
])
def test_supported_vehicles(fingerprint, expected):
  cp = CarInterface.get_non_essential_params(fingerprint)
  cp.openpilotLongitudinalControl, cp.pcmCruise = True, False
  assert supported(cp) is expected


def test_other_brands_are_not_supported():
  from opendbc.car.honda.interface import CarInterface as HondaInterface
  from opendbc.car.honda.values import CAR as HONDA
  from opendbc.car.toyota.interface import CarInterface as ToyotaInterface
  from opendbc.car.toyota.values import CAR as TOYOTA
  assert not supported(HondaInterface.get_non_essential_params(HONDA.HONDA_CIVIC))
  assert not supported(ToyotaInterface.get_non_essential_params(TOYOTA.TOYOTA_COROLLA_TSS2))
  cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
  cp.passive = True
  assert not supported(cp)
  assert not supported(None)
