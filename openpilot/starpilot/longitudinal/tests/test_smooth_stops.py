import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MAX_BP, J_CRUISE_VALS, get_cruise_accel
from openpilot.starpilot.longitudinal.smooth_stops import (CRAWL_TIME, CREEP_FADE_SPEED, CREEP_SPEED, KISS_DECEL, KISS_SPEED,
                                                           LANDING_SPEED, LAUNCH_FRAMES, LEAD_ABSENCE_FRAMES, LEAD_MIN_GAP_BUDGET,
                                                           STALL_MAX, STANDSTILL_SPEED, STOP_ALLOWANCE, Lead, SmoothStops,
                                                           landing_bound, landing_floor, owner_for)
from openpilot.starpilot.longitudinal.smooth_stops_preferences import KEY, SmoothStopsPreferences, supported

NOW = 100_000_000_000
NO_LEADS = (Lead(), Lead())
# the planner's action_t on the Palisade: the car's actuator delay plus one model frame
ACTION_T = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE).longitudinalActuatorDelay + DT_MDL
# and its cruise jerk, which a landing hands the output back at
RELEASE_JERK = (A_CRUISE_MAX_BP, J_CRUISE_VALS)


def leads(distance, speed=0.0, acceleration=0.0):
  return Lead(True, distance, speed, acceleration), Lead()


def frames(seconds):
  return round(seconds / DT_MDL)


def step(law, a_target, v_ego, leads=NO_LEADS, intent=True, stopping=False, a_ego=None, standstill=False):
  return law.update(a_target, v_ego, leads, intent, stopping, a_ego=a_ego, standstill=standstill)


def out(law, a_target, v_ego, leads=NO_LEADS, intent=True, a_ego=None):
  # StarPilot's own stop bit up, as it is through the end of a stop
  return step(law, a_target, v_ego, leads, intent, stopping=True, a_ego=a_ego)[0]


def landing(v_ego=1.5, a_target=-2.0, action_t=ACTION_T):
  law = SmoothStops(DT_MDL, action_t, RELEASE_JERK)
  step(law, a_target, v_ego)
  assert law.landing
  return law


def edge(law, v_ego, a_target):
  # the braking the landing allows for this plan: the corridor, or the plan's own stop lengthened by at most STOP_ALLOWANCE
  return max(law.bound(v_ego), law.plan_requirement(v_ego, a_target))


def corridor_ride(edge_fn, v0, h=1e-4):
  # a car riding a corridor edge exactly, v' = -edge(v), integrated forward in time from v0
  ts, vs = [0.0], [v0]
  while vs[-1] > KISS_SPEED - 0.1:
    vs.append(vs[-1] - h * edge_fn(vs[-1]))
    ts.append(ts[-1] + h)
  return np.array(ts), np.array(vs)


def cruise_carry(a_prev, v_ego):
  # the planner's cruise candidate carrying a hand-over up from a_prev toward a set speed far above
  return get_cruise_accel(True, v_ego + 10.0, v_ego, a_prev, 0.0, NS(steerRatio=15.0, wheelbase=2.9), DT_MDL, 0.0, True)


class TestCorridor:
  def test_the_bound_only_removes_braking_and_the_floor_only_adds_it_up_to_itself(self):
    law = landing()
    for v_ego in np.linspace(LANDING_SPEED - 0.01, 0.0, 36):
      for a_target in np.linspace(-3.5, -0.1, 18):
        result = out(law, a_target, v_ego)
        assert result >= min(a_target, -law.floor(v_ego)) - 1e-9, (v_ego, a_target, result)
        assert result <= a_target + 1e-9 or math.isclose(result, -edge(law, v_ego, a_target), abs_tol=1e-9), (v_ego, a_target, result)

  def test_a_rolling_landing_never_asks_for_acceleration(self):
    # always braking while the floor lives; above its fade a climbing plan passes as it is and is never raised
    rng = np.random.default_rng(0)
    law = landing()
    for v_ego, a_target in zip(rng.uniform(STANDSTILL_SPEED + 0.01, LANDING_SPEED - 0.01, 3000), rng.uniform(-3.5, 1.5, 3000),
                               strict=True):
      law.landing, law.stop_bit, law._positive_frames = True, False, 0
      result = out(law, a_target, v_ego)
      if law.floor(v_ego) > 0.0:
        assert result < 0.0, (v_ego, a_target, result)
      if a_target >= 0.0:
        assert result <= a_target, (v_ego, a_target, result)

  def test_without_a_delay_the_edges_are_the_car_side_corridor(self):
    law = SmoothStops(DT_MDL, 0.0, RELEASE_JERK)
    for v_ego in np.linspace(LANDING_SPEED - 0.01, 0.0, 50):
      assert math.isclose(law.bound(v_ego), landing_bound(v_ego), abs_tol=1e-9)
      assert math.isclose(law.floor(v_ego), landing_floor(v_ego), abs_tol=1e-9)

  def test_each_edge_is_requested_one_actuator_delay_ahead_of_the_car(self):
    # the request made at a speed is what a car riding the corridor carries action_t later
    law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    for edge_fn, request_fn, v0, breakpoints in ((landing_bound, law.bound, 1.8, (0.9, KISS_SPEED)),
                                                 (landing_floor, law.floor, 1.4, (CREEP_SPEED, KISS_SPEED))):
      ts, vs = corridor_ride(edge_fn, v0)
      for v_car in breakpoints:
        t_car = np.interp(-v_car, -vs, ts)
        v_request = float(np.interp(t_car - ACTION_T, ts, vs))
        assert v_request > v_car + 0.05
        assert math.isclose(request_fn(v_request), edge_fn(v_car), abs_tol=2e-3), (edge_fn.__name__, v_car)
    # the ceiling is left where it is
    assert math.isclose(law.bound(LANDING_SPEED), landing_bound(LANDING_SPEED))

  def test_a_plan_inside_the_corridor_passes_untouched(self):
    law = landing()
    assert out(law, -2.0, 3.0) == -2.0
    assert out(law, -0.5, 1.5) == -0.5
    assert out(law, -2.0, 1.5) == -edge(law, 1.5, -2.0)

  def test_the_floor_keeps_a_landing_braking_and_fades_above_creep_speed(self):
    law = landing()
    assert out(law, -0.05, 0.8) == -law.floor(0.8)
    # above the fade the floor is gone: an easing plan in a queue is not dragged to a stop
    assert out(law, -0.05, 2.0) == -0.05
    # the floor never starts a landing on its own
    assert out(SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK), -0.05, 0.8, intent=False) == -0.05

  def test_both_edges_meet_at_the_kiss_by_walking_pace(self):
    law = landing()
    for v_ego in (KISS_SPEED, 0.2, 0.12):
      for a_target in (-0.6, -0.02, 0.05):
        assert math.isclose(out(law, a_target, v_ego), -max(KISS_DECEL, law.plan_requirement(v_ego, a_target)), abs_tol=1e-9)
    # the wheels stop under the kiss; once the stop bit is up at rest, LongControl's stopping state holds the car and the plan
    # passes as it is
    assert out(law, -0.6, STANDSTILL_SPEED) == -KISS_DECEL
    for a_target in (-0.6, -0.02, 0.05):
      assert out(law, a_target, 0.0) == a_target
    assert law.landing

  def test_the_law_lives_below_landing_speed_and_starts_only_on_braking_intent(self):
    law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    assert out(law, -2.0, LANDING_SPEED + 0.1) == -2.0 and not law.landing
    assert out(law, -2.0, 1.5, intent=False) == -2.0 and not law.landing
    assert out(law, 0.5, 1.0) == 0.5 and not law.landing
    # a hover frame with intent is not a stop: the plan must brake more than the kiss to start one
    assert out(law, -KISS_DECEL / 2.0, 0.2) == -KISS_DECEL / 2.0 and not law.landing
    assert out(law, -2.0, 1.5) < -0.5 and law.landing
    # intent flickering off does not end it
    assert out(law, -2.0, 1.4, intent=False) == -edge(law, 1.4, -2.0)
    assert law.landing
    # leaving the window does
    assert out(law, -2.0, LANDING_SPEED) == -2.0 and not law.landing


class TestStopAllowance:
  def test_an_urgent_plan_keeps_the_braking_that_stops_within_the_allowance(self):
    # the stop the landing allows is never more than STOP_ALLOWANCE longer than the one the plan's own request makes
    law = landing(LANDING_SPEED - 0.01, -3.0)
    for v_ego in np.linspace(0.2, LANDING_SPEED - 0.01, 30):
      for a_target in (-0.8, -1.5, -2.5, -3.5):
        braking = -out(law, a_target, v_ego)
        assert v_ego ** 2 / (2.0 * braking) <= v_ego ** 2 / (2.0 * -a_target) + STOP_ALLOWANCE + 1e-9, (v_ego, a_target)

  def test_the_allowance_only_lets_braking_through(self):
    law = landing()
    for v_ego in (0.3, 1.0, 2.5):
      assert law.plan_requirement(v_ego, 0.3) == 0.0
      assert law.plan_requirement(v_ego, -1.0) < 1.0
    assert law.plan_requirement(0.0, -2.0) == 0.0


class TestNoLoopOnTheMeasuredDeceleration:
  def test_the_request_does_not_depend_on_a_car_braking_beyond_the_floor(self):
    for v_ego in (0.6, 1.0, 1.5, 2.0, 3.0):
      for a_target in (-0.5, -1.0, -2.5):
        outputs = set()
        for surplus in (0.0, 0.3, 1.0, 2.5):
          law = landing(v_ego, a_target)
          a_ego = -landing_floor(v_ego) - surplus
          outputs.add(tuple(out(law, a_target, v_ego, a_ego=a_ego) for _ in range(frames(1.0))))
        assert len(outputs) == 1, (v_ego, a_target, outputs)


class TestLatchAndLaunch:
  def test_a_nonfinite_speed_leaves_the_plan_alone(self):
    law = landing()
    assert step(law, -2.0, math.nan, stopping=True) == (-2.0, True) and not law.landing
    assert out(law, -2.0, math.inf) == -2.0

  def test_a_nonfinite_target_leaves_the_plan_alone_and_does_not_corrupt_the_launch_count(self):
    law = landing(0.6, -0.3)
    assert out(law, 0.1, 0.6) == -law.floor(0.6) and law.landing
    assert out(law, 0.2, 0.6) == -law.floor(0.6) and law.landing
    assert math.isnan(out(law, math.nan, 0.6)) and law.landing
    out(law, 0.3, 0.6)
    assert not law.landing
    fresh = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    assert math.isnan(out(fresh, math.nan, 1.5))
    assert not fresh.landing

  def test_a_hover_around_zero_is_held_at_the_floor_and_never_flickers(self):
    law = landing(0.3, -0.2)
    outputs = [out(law, a, 0.2) for a in (0.03, -0.2, 0.09, -0.2, 0.13, -0.05, 0.16, 0.02)]
    assert all(-edge(law, 0.2, -0.2) <= o <= -law.floor(0.2) for o in outputs), outputs
    assert max(outputs) - min(outputs) < 0.05
    assert law.landing

  def test_a_climbing_plan_ends_the_landing_after_launch_frames_only_while_rolling(self):
    law = landing(0.6, -0.3)
    for i in range(LAUNCH_FRAMES - 1):
      assert out(law, 0.1 * (i + 1), 0.6) == -law.floor(0.6) and law.landing
    out(law, 0.1 * LAUNCH_FRAMES, 0.6)
    assert not law.landing
    out(law, 0.3, 0.6)
    assert not law.landing

  def test_reset_forgets_the_landing_and_its_release(self):
    law = landing(1.0, -0.5)
    for _ in range(frames(0.5)):
      out(law, -0.1, 1.0, a_ego=-0.5)
    for p in (0.5, 0.6, 0.7):
      out(law, p, 1.0)
    assert law.releasing
    law.reset()
    assert not law.landing and not law.releasing
    assert out(law, -2.0, 1.5, intent=False) == -2.0
    assert out(law, 0.5, 1.5, intent=False) == 0.5


class TestStopBit:
  def test_the_stop_bit_is_held_back_while_the_kiss_lands_a_rolling_car(self):
    law = landing(1.0, -0.6)
    for v_ego in np.linspace(1.0, STANDSTILL_SPEED + 0.01, 20):
      assert step(law, -0.6, v_ego, stopping=True)[1] is False
    # and rises once the wheels have nearly stopped, or the car reports standstill
    assert step(law, -0.6, STANDSTILL_SPEED, stopping=True)[1] is True
    law = landing(1.0, -0.6)
    assert step(law, -0.6, 0.3, stopping=True, standstill=True)[1] is True

  def test_the_landed_stop_rises_the_bit_even_without_starpilots_own(self):
    law = landing(0.5, -0.4)
    assert step(law, -0.4, 0.05, stopping=False) == (-KISS_DECEL, True)

  def test_a_lead_nearer_than_the_kiss_can_stop_short_of_gets_the_stop_bit_at_once(self):
    law = landing(0.8, -1.0)
    assert step(law, -1.0, 0.8, leads(6.0), stopping=True)[1] is False
    near = leads(1.2)
    assert law.lead_requirement(0.8, near[0]) > law.bound(0.8)
    assert step(law, -1.0, 0.8, near, stopping=True)[1] is True

  def test_once_risen_the_bit_stays_through_wheel_speed_ticks_at_rest(self):
    law = landing(0.5, -0.4)
    assert step(law, -0.4, 0.0, stopping=True)[1] is True
    for v_ego in (0.12, 0.0, 0.2, 0.0, 0.15):
      assert step(law, -0.4, v_ego, stopping=True)[1] is True
    assert law.landing

  def test_at_rest_starpilots_own_bit_drop_ends_the_landing_at_once(self):
    # a launch from rest (a lead departing, a takeoff, a release) is never delayed: the plan passes the frame StarPilot lets go
    law = landing(0.5, -0.4)
    assert step(law, -0.4, 0.0, stopping=True) == (-KISS_DECEL, True)
    assert step(law, -0.02, 0.0, stopping=True) == (-0.02, True)
    assert step(law, 0.6, 0.0, stopping=False) == (0.6, False)
    assert not law.landing and not law.releasing
    # and the car stays StarPilot's until it rolls faster than walking pace again
    assert step(law, -0.4, 0.2, stopping=False) == (-0.4, False) and not law.landing
    assert step(law, -0.4, KISS_SPEED, stopping=False) == (-KISS_DECEL, False) and law.landing

  def test_a_creeping_car_gets_the_stop_bit_after_crawl_time(self):
    law = landing(0.6, -0.4)
    bits = [step(law, -0.4, 0.2, stopping=True)[1] for _ in range(frames(CRAWL_TIME) + 2)]
    assert not any(bits[:frames(CRAWL_TIME) - 1]) and bits[-1]

  def test_outside_a_landing_the_bit_is_starpilots(self):
    law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    assert step(law, 0.3, 0.2, stopping=True) == (0.3, True)
    assert step(law, -1.0, 5.0, stopping=False) == (-1.0, False)


class TestLead:
  def test_the_output_is_continuous_as_a_stopped_lead_closes(self):
    law = landing(2.0, -3.0)
    outputs = [out(law, -3.0, 2.0, leads(d)) for d in np.arange(12.0, 1.0, -0.001)]
    assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.01

  def test_the_braking_passed_grows_without_limit_as_the_room_left_after_the_delay_closes(self):
    law = landing(0.5, -1.0)
    outputs = [out(law, -1.0, 0.5, leads(d)) for d in np.arange(3.0, 0.05, -0.001)]
    assert all(b <= a + 1e-12 for a, b in zip(outputs, outputs[1:], strict=False))
    assert max(abs(b - a) for a, b in zip(outputs, outputs[1:], strict=False)) < 0.01
    assert outputs[0] > -0.5 and outputs[-1] == -1.0

  def test_a_lead_gets_what_stops_the_car_short_of_it_after_the_delay(self):
    for v_ego, obstacle, plan in ((0.37, (0.32,), -0.6), (1.0, (1.0,), -2.0), (0.7, (0.6,), -1.5), (1.5, (1.1,), -3.5),
                                  (3.0, (1.0, 3.0, -4.0), -3.5), (0.5, (1.2,), -1.0), (0.5, (0.8,), -1.0), (2.0, (3.0,), -2.0)):
      law = landing(v_ego, -2.0)
      distance, speed, acceleration = (*obstacle, 0.0, 0.0)[:3]
      rest = distance + (speed * speed / (-2.0 * acceleration) if acceleration < 0.0 else 0.0)
      room = rest - v_ego * ACTION_T - LEAD_MIN_GAP_BUDGET
      needs = v_ego * v_ego / (2.0 * room) if room > 0.0 else math.inf
      result = out(law, plan, v_ego, leads(*obstacle), a_ego=-0.5)
      assert -result >= min(needs, -plan) - 1e-9, (v_ego, obstacle, result)

  def test_a_plan_braking_less_than_the_leads_requirement_is_left_alone(self):
    for v_ego, distance, plan in ((1.0, 1.5, -0.8), (0.37, 0.32, -0.3), (1.5, 1.1, -0.2)):
      law = landing(v_ego, -2.0)
      assert out(law, plan, v_ego, leads(distance)) == plan

  def test_a_dropped_lead_keeps_its_requirement_for_the_absence_window_and_hands_it_back_gently(self):
    law = landing(1.0, -2.0)
    near = leads(1.0)
    held = out(law, -2.0, 1.0, near)
    assert held == -2.0
    for _ in range(LEAD_ABSENCE_FRAMES - 1):
      assert out(law, -2.0, 1.0) == held
    outputs = [held] + [out(law, -2.0, 1.0) for _ in range(frames(1.0))]
    assert outputs[-1] == -edge(law, 1.0, -2.0)
    assert max(np.diff(outputs)) <= cruise_carry(0.0, 1.0) + 1e-9


class TestAntiStall:
  def test_a_car_not_slowing_presses_both_edges_more_every_frame_up_to_the_cap(self):
    law = landing(0.3, -0.3)
    kiss = [out(law, -0.3, 0.3, a_ego=0.1) for _ in range(frames(3.0))]
    assert kiss[0] < -KISS_DECEL
    assert all(b < a for a, b in zip(kiss, kiss[1:frames(1.0)], strict=False))
    assert math.isclose(kiss[-1], -(KISS_DECEL + STALL_MAX), rel_tol=1e-9)
    # the same shortfall above the kiss speed opens the bound too, beyond what the plan's allowance already lets through
    law = landing(1.0, -3.0)
    bound = [out(law, -3.0, 1.0, a_ego=0.0) for _ in range(frames(2.0))]
    assert bound[-1] < -edge(law, 1.0, -3.0) and all(b <= a + 1e-12 for a, b in zip(bound, bound[1:], strict=False))

  def test_the_press_drains_as_the_car_slows_again(self):
    law = landing(0.3, -0.3)
    for _ in range(frames(1.0)):
      pressed = out(law, -0.3, 0.3, a_ego=0.1)
    drained = [out(law, -0.3, 0.3, a_ego=-0.6) for _ in range(frames(2.0))]
    assert drained[0] > pressed
    assert all(b >= a for a, b in zip(drained, drained[1:], strict=False))
    assert drained[-1] == -KISS_DECEL

  def test_nothing_is_pressed_at_rest(self):
    for v_ego in (0.0, 0.05, STANDSTILL_SPEED):
      law = landing(0.3, -0.3)
      for _ in range(frames(1.0)):
        out(law, -0.3, 0.3, a_ego=0.1)
      assert law.integral > 0.1
      assert out(law, -0.3, v_ego, a_ego=0.2) == -KISS_DECEL

  def test_nothing_winds_it_up_where_the_floor_has_faded_out(self):
    for a_ego in (0.0, -0.3, -1.5, 0.4, 1.8):
      law = landing(3.4, -2.5)
      for v_ego in (3.4, 2.5, CREEP_FADE_SPEED):
        for i in range(frames(2.0)):
          a_target = -2.5 if i % 2 else -0.4
          assert out(law, a_target, v_ego, a_ego=a_ego) == max(a_target, -edge(law, v_ego, a_target))
          assert law.integral == 0.0


class TestRelease:
  def test_a_plan_climbing_out_of_a_pressed_landing_is_handed_back_at_the_cruise_jerk(self):
    law = landing(0.93, -0.19)
    for _ in range(frames(1.0)):
      out(law, -0.19, 0.93, a_ego=0.0)
    outs = [law.output] + [out(law, p, 0.93, a_ego=0.0) for p in (0.04, 0.1) + (0.3,) * frames(1.5)]
    assert not law.landing and not law.releasing
    assert max(np.diff(outs)) <= cruise_carry(0.0, 0.93) + 1e-9
    assert outs[-1] == 0.3

  def test_only_what_the_landing_held_is_carried(self):
    law = landing(2.0, -0.5)
    plans = [-0.5, 0.2, 0.7, 1.2, 1.6]
    assert [out(law, p, 2.0) for p in plans] == plans
    assert not law.landing and not law.releasing

  def test_braking_during_a_release_passes_at_once(self):
    law = landing(1.0, -0.5)
    for _ in range(frames(0.5)):
      out(law, -0.1, 1.0, a_ego=-0.5)
    for p in (0.5, 0.6, 0.7):
      out(law, p, 1.0)
    assert law.releasing
    assert out(law, -1.5, 1.0, intent=False) == -1.5 and not law.releasing

  def test_the_speed_leaving_the_window_does_not_cut_a_release_short(self):
    law = landing(1.2, -0.5)
    out(law, -0.1, 1.2, a_ego=-0.5)
    v_ego, outs, beyond = 1.2, [law.output], []
    for _ in range(frames(3.0)):
      outs.append(out(law, 3.5, v_ego))
      if v_ego >= LANDING_SPEED:
        beyond.append(outs[-1])
      v_ego += outs[-1] * DT_MDL
    assert beyond[0] < 3.5 and outs[-1] == 3.5
    assert max(np.diff(outs)) <= max(RELEASE_JERK[1]) * DT_MDL + 1e-9


class TestPlannerHook:
  def sm(self, v_ego=1.0, a_ego=-0.5, standstill=False, experimental=False, model_stop=False, lead=None):
    def radar(present, d=0.0):
      return NS(present=present, dRel=d, vLeadK=0.0, aLeadK=0.0)
    return {'carState': NS(vEgo=v_ego, aEgo=a_ego, standstill=standstill),
            'radarState': NS(leadOne=radar(lead is not None, lead or 0.0), leadTwo=radar(False)),
            'modelV2': NS(action=NS(shouldStop=model_stop)), 'selfdriveState': NS(experimentalMode=experimental)}

  def test_disabled_or_inactive_returns_the_planner_values_themselves(self):
    law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    target = np.float64(-1.0)
    for enabled, active in ((False, True), (True, False), (False, False)):
      result = law.sample(self.sm(), target=target, stopping=True, enabled=enabled, active=active, plan_speeds=[0.0])
      assert result[0] is target and result[1] is True
    assert law.idle

  def test_turning_it_off_mid_landing_hands_the_plan_back(self):
    law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
    assert law.sample(self.sm(), target=-1.5, stopping=True, enabled=True, active=True, plan_speeds=[0.0])[1] is False
    assert law.landing
    assert law.sample(self.sm(), target=-1.5, stopping=True, enabled=True, active=False, plan_speeds=[0.0]) == (-1.5, True)
    assert law.idle

  def test_stop_intent(self):
    for kwargs, plan_speeds, stopping, intent in (({}, [3.0, 0.2], False, True), ({}, [3.0, 2.0], False, False),
                                                  ({}, [3.0, 2.0], True, True),
                                                  ({'experimental': True, 'model_stop': True}, [3.0, 2.0], False, True),
                                                  ({'model_stop': True}, [3.0, 2.0], False, False)):
      law = SmoothStops(DT_MDL, ACTION_T, RELEASE_JERK)
      law.sample(self.sm(**kwargs), target=-1.0, stopping=stopping, enabled=True, active=True, plan_speeds=plan_speeds)
      assert law.landing == intent, (kwargs, plan_speeds, stopping)

  def test_owner_only_for_supported_cars_with_starpilot_longitudinal(self):
    cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
    cp.openpilotLongitudinalControl, cp.pcmCruise = True, False
    assert owner_for(cp, DT_MDL, RELEASE_JERK) is not None
    cp.openpilotLongitudinalControl, cp.pcmCruise = False, True
    assert owner_for(cp, DT_MDL, RELEASE_JERK) is None
    assert owner_for(NS(), DT_MDL, RELEASE_JERK) is None


@pytest.mark.parametrize('acceleration', [0.0, -1.0])
def test_lead_from_radar_reads_finite_filtered_values(acceleration):
  lead = Lead.from_radar(NS(present=True, dRel=-0.2, vLeadK=-0.1, aLeadK=acceleration))
  assert lead == Lead(True, 0.0, 0.0, acceleration)
  assert Lead.from_radar(NS(present=True, dRel=math.nan, vLeadK=0.0, aLeadK=0.0)) == Lead()
  assert Lead.from_radar(NS(present=False)) == Lead()
  assert Lead.from_radar(None) == Lead()


def test_output_never_goes_below_accel_min():
  law = landing(3.4, -4.0)
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
  # CAN FD, the G90's own hold shaping, blended longitudinal, the Ray's pedal, and cars without openpilot longitudinal
  (CAR.HYUNDAI_IONIQ_5, False), (CAR.KIA_EV9, False), (CAR.GENESIS_G90, False), (CAR.HYUNDAI_PALISADE_2023, False),
  (CAR.KIA_RAY_EV, False), (CAR.HYUNDAI_KONA_NON_SCC, False), (CAR.GENESIS_G70, False),
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
