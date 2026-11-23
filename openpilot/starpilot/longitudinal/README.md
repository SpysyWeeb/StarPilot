# Longitudinal control

## Vehicle extensions

`LongControl` keeps the shared state machine, PID, stopping ramp and final acceleration limits. A startup-selected `LongitudinalExtension` applies the vehicle's existing transition, feedforward and output policies around those stages. Vehicle-specific behavior belongs in its policy module; adding a vehicle must not add another brand branch or argument to the upstream controller.

`LongitudinalInputs` owns the required message subscriptions and builds one `LongitudinalContext` from the current messages. Source freshness, drive identity, lead interpretation and saved profile checks remain with this owner. Missing evidence retains the selected policy's existing fallback behavior; a context does not grant CAN ownership or steering/braking permission.

Policy order matters. Target shaping precedes PID preparation; vehicle output shaping precedes launch handoff and mode-transition shaping; the shared acceleration clip remains last. `reset_start=False` resets the PID and ordinary policy state while retaining the existing launch/stop state. New policies need regressions for inactive control, driver override, missing evidence, stop/start transitions and unchanged neighboring configurations.

## Optional planner cruise ceiling

`LongitudinalPlanner.update(sm, *, cruise_ceiling=None, profile_tuning=None)` accepts an optional
`CruiseCeiling(speed_mps, authority)` in addition to the existing driver cruise
message. The live [SLC](../speed_limits/README.md) caller supplies a qualified cap.
The planner still reads `carState.vCruise` as the driver's selected ceiling and
leaves supplied messages unchanged. Accepted limits affect the planner cap, not SET.

The `CruiseCeiling` dataclass validates no source or acceptance history by
construction. Its caller must establish an accepted, normalized m/s ceiling in
the correct speed coordinate, including the fixed previous-accepted-limit fallback.
Vehicle coordinates and authority remain fresh. The planner validates its numeric
shape and the shared four-mode `Authority` contract. It applies the lower of
the supplied ceiling and the existing driver ceiling only when native system
longitudinal control is available, the supplied authority grants active system
longitudinal control, `selfdriveState.enabled` is true,
`longControlState` is active, and the driver's cruise speed is initialized and
positive. An invalid, absent, higher, or inactive ceiling leaves the original
cruise path unchanged. `forceDecel` still sets the cruise candidate to zero.
The planner records the decision in `last_cruise_ceiling_status`; it does not
publish that diagnostic or alter the longitudinal plan schema.

This optional input changes only the cruise candidate before the planner's
existing jerk limiting. SLC uses the selected profile for its capped target; a cap does
not strengthen a softer cruise profile. While the SLC cap is the final cruise target and no
braking-relevant lead is present, an overshoot is corrected over `CRUISE_ACCEL_RESPONSE_TIME`
and braking builds at 0.15 m/s³ until -0.3 m/s², so friction brakes engage gently; braking
already requested elsewhere keeps the normal jerk. Lead-relevance checks still prevent cruise comfort
tuning from weakening braking for a close, slow or braking lead; malformed lead
evidence also vetoes that tuning. The native MPC, lead, e2e, stop and acceleration
arbitration remain in place. An inactive planner may still compute a nonzero
target; the host owns actuation gating. Stock setpoint control has a different
output domain.

For the exact tagged Ioniq 6 LONG configuration while native longitudinal
control is active, the model gas-probability coast veto uses the frozen
StarPilot 0.35/0.45 thresholds with a 250 ms source-timed hold. Each accepted
model sample must be valid, alive, current and newer than the previous sample;
repeated, stale, future or unknown input revokes throttle permission without
advancing the hold. A fresh model sample at or below 2.5 m/s keeps the native
low-speed bypass. Stock Ioniq, other vehicles and inactive longitudinal control
retain the native one-frame 0.4 threshold. This comfort policy has not been
qualified on the vehicle.

Native tests use real Cap'n Proto messages and the compiled acados MPC on two
independent planner instances. They compare default/absent, lower and higher
ceilings, lead and e2e braking, force deceleration, uninitialized or zero driver
cruise, inactive control, and inconsistent authority. They verify input bytes
remain unchanged and observe the original MPC reset function. Passing these
tests establishes the bounded planner input behavior, not vehicle qualification.

## Opt-in longitudinal profiles

An exact tagged Ioniq 6 system-longitudinal session starts saved personality
settings when their validated document is present. The
`LONG_PLANNER_REPLAY_RUNTIME=1` development flag remains available for replay.
Fresh installations leave Custom Following and Jerk (`CustomPersonalities`)
off. Selected acceleration and deceleration profiles apply independently:
their fresh defaults are StarPilot Default and Comfort, respectively. Existing
saved follow and jerk values keep their frozen
names, units and defaults. The strict versioned profile document migrates
supported older versions, rejects malformed persisted values, and may choose
named or custom acceleration, braking and following curves. The local Params
cache refreshes once per second; invalid or unavailable values fade to native
behavior. A model frame uses the log timeline only under `REPLAY`; otherwise it
uses the host monotonic clock and requires fresh car, control and selfdrive
messages.

Only native system longitudinal control with active `carControl.longActive` may
apply a profile. The bounded transition changes the existing MPC follow time,
acceleration upper bound and jerk costs, plus the cruise candidate's
acceleration/braking limits. It cannot widen the native car acceleration range.
Valid saved curves above native acceleration bounds are intersected with those
bounds without dropping their other tuning or rewriting the saved document.
Lead, e2e and cruise candidates still arbitrate normally. A lower explicit
cruise or curve ceiling retains at least the native cruise braking floor even when a saved profile asks
for a softer approach; stronger configured braking may still apply.
`forceDecel` keeps the native cruise braking response. Stock ACC and lateral-only control receive
no profile effect. The acados model and cereal schemas are unchanged.

The profile runtime accepts an explicit current Traffic-mode verdict. With
Traffic known active, base follow and all five jerk values interpolate over
0–25 m/s from the Traffic value to the Relaxed low-speed value; the frozen
Traffic defaults apply even when Custom Personalities is off. Traffic also
uses the frozen seven-point acceleration ceiling and 0.42 m/s² cruise-braking
magnitude as its base defaults. Selected acceleration and braking profiles
follow the inheritance rules below. Saved Traffic following and preserved
legacy acceleration/braking curves retain their master, document and Traffic
profile gates. Historical v3
Traffic braking as low as 0.35 m/s² is admitted only for a Traffic target;
ordinary profiles retain their 0.5 m/s² lower bound. A fresh explicit Traffic OFF
immediately changes the status and fades following, jerk, acceleration and
cruise-braking values toward the ordinary native personality;
unknown or stale Traffic authority clears the applied profile immediately.
Ordinary personalities keep their existing 45/70 mph follow curve. The global
**Selected Deceleration Profile** and explicit personality braking presets use
different historical scales:

| UI selection | Global or inherited through Use global preset | Explicit personality preset |
| --- | --- | --- |
| Comfort (`eco`) | 0.6 m/s² | 0.5 m/s² |
| Normal (`standard`) | 1.2 m/s² | 1.0 m/s² |
| Sport (`sport`) | 2.4 m/s² | 2.0 m/s² |

These are positive deceleration magnitudes for ordinary cruise braking,
including a changed SLC cap. Global values preserve the historical
1.2 m/s² baseline with half/normal/double responses; explicit personality
presets retain their original physical values. The distinction preserves
existing tuning and saved-document compatibility.

Comfort is the fresh global default. A personality set to **Use global preset**
inherits the global choice; an explicit preset or Custom curve overrides it.
For example, changing from inherited global Sport to explicit personality
Sport changes the cruise braking magnitude from 2.4 to 2.0 m/s².
**StarPilot Default** uses the existing native/profile path rather than either
named scale. Existing saved selections, custom curves and migration gates
keep their original meaning.

A braking-relevant lead, force deceleration or stop context prevents the
selected response from replacing native hazard handling. A lower explicit
cruise or curve ceiling retains at least native approach braking. MPC lead and
e2e candidates still arbitrate and may demand stronger braking; the table is
not a limit on their braking. Frozen weather, Traffic hazard-context braking
and other vehicle-special-case policies remain outside this slice. The current
effective StarPilot follow floor is 0.5 s, matching saved profile admission.
Stock planner following remains independent of StarPilot preferences. The
6 m stopping distance and braking-distance guard remain unchanged. Offline native tests compare the committed planner and MPC outputs
with the new default path, then verify a selected profile changes real solver
parameters and plan outputs. No device or physical-timing qualification is
claimed.

The versioned profile document and preset curves are adapted from StarPilot
`678af78347d9656bc2f5dacd4204b012db861fa7`; its applicable MIT notice is retained
in `LICENSE`. The new runtime composes these settings with the current upstream
planner rather than retaining the older planner's unrelated policies.

## Lane-change close gap

The shared versioned Lane Change Preferences document now has an explicit,
default-off close-gap choice and a 0.75–1.0 s requested follow time. Older
documents decode with this choice off. A bounded reader accepts only a valid
saved document; the planner requires current valid model, car, radar, control
and selfdrive sources, active system longitudinal control, and a model-reported
pre/starting/finishing lane-change state with a known side. It reuses the saved
minimum lane-change speed. A blinker alone never starts the reduction.

The reduction ramps into the native MPC follow-time input and ramps out on an
ordinary lane-state exit. An occupied target-side blindspot, standstill,
strong braking by either lead, stop intent, force deceleration, driver pedal,
invalid source or lost authority restores the ordinary follow time immediately.
The MPC still arbitrates lead and stop targets and enforces its acceleration
limits. Current StarPilot MPC input admission supports follow time at or above 0.5 s;
the frozen StarPilot default was 0.6 s and could accept 0.25 s. Those smaller
saved requests are rejected, not clamped or silently rewritten. The restored
behavior and this protective hard-veto reset require vehicle evaluation.

## Ioniq 6 launch state

The separately tagged Ioniq 6 HDA-II LONG configuration can emit the starting
state consumed by its acceleration calibration. Stock Ioniq configurations and
other vehicles retain the native state machine. This policy does not grant ECU
ownership: the separate saved DEBUG opt-in must pass the live Ioniq prearm
handoff before the tagged LONG session can use it. Deprecated CarParams start
fields are not used.

Admission requires a current drive, recent valid car/planner/radar evidence,
both radar leads known absent, no pedal press, and valid CAN. At up to 0.1 m/s,
the planner target and native acceleration ceiling must both permit at least
0.75 m/s² for 35 control samples spanning at least 340 ms. A drive change,
clock discontinuity, gap above 20 ms, or lost evidence restarts that window.
Several control samples can use the same still-fresh 20 Hz source observation.
The start command cannot exceed the planner target, native ceiling, or 1 m/s².
Ordinary PID release remains available under its native conditions.

This deliberately changes the frozen soft-launch behavior: a weak lead or
profile request cannot be raised to the old launch floor. The car controller
also withdraws that floor immediately when a later target becomes softer or
negative, including between its 20 Hz calibration updates. Its existing bounded
decrease handles residual output rather than abruptly clipping acceleration.
Tests cover continuous reference traces, the host-to-decoded-SCC path for both
supported layouts, and unchanged native behavior outside this configuration.
Physical launch response and ECU ownership still require vehicle qualification.

## Smooth Stops

Default-off `SmoothStops` has one job: make the end of every stop as soft as it can on gas Hyundai, Kia and Genesis CAN cars with
radar SCC under StarPilot longitudinal control. Without it, the planner's stop bit rises at walking pace, about a second before the
wheels stop, and `LongControl`'s stopping state then ramps the request toward `CP.stopAccel` regardless of the plan: braking rises
into the stop and the body rocks forward and back as the wheels stop.

`smooth_stops.py` runs after the candidates are arbitrated and before the HKG vision stop buffer, and lands every stop the planner
makes: behind a lead, the model's in Experimental mode, cruise's, and Enhanced Stops' approach and committed stop. A landing only
ever eases what StarPilot itself asks: its plan, or, once StarPilot's own stop bit has stayed up 0.15 s, the stopping ramp
LongControl would be running. The eased request stays inside a corridor that falls with speed to a 0.15 m/s² kiss at walking pace,
requested one actuator delay ahead (the planner's effective delay, including a developer override).
More braking than StarPilot asks comes only from an anti-stall press while the car delivers less than asked (creep, a weak brake).
StarPilot's stop bit, including one Enhanced Stops or the model raises at speed, is held back until the wheels have all but
stopped (0.05 m/s, the standstill flag, or a crawl under 0.10 m/s that has lasted 1 s), so the stopping ramp and
the hold start at rest; the bit then stays up until the plan asks to move.

Where the car rests stays StarPilot's, within a bound. Enhanced Stops, committed or approaching a light it tracks, does not rest past
its stop line as moved by Stop Distance Adjustment unless today's stop already would; then, on a weak brake, the landing hands the
stop back and the car rests at most a few centimetres further past it than today (0.06 m in simulation). Today's Enhanced Stops
stops end 1.5-3.4 m short of that line, so a landed one can end that much nearer it. Any other stop rests at most 0.5 m further on than where StarPilot's stopping ramp would have ended
it, counted from the moment StarPilot's bit has stayed up 0.15 s; easing before that moment is not bounded, so the simulated worst
case is 0.73 m. It never rests past the model's own stop point, and a radar lead always keeps 2 m after the actuator delay. Where the
soft end no longer fits, the landing eases less. A demand of 2 m/s² or more passes exactly as StarPilot would send it while it lasts (its stopping state keeps a request already below
stopAccel); once it has eased under 1.8 m/s² the landing lets braking off at 1.5 m/s³ and takes the tail back, within the
model's own stop point. A landing hands the stop back when the plan no longer means to stop, when a lead is closer than the kiss can stop short of, when no room is left, or when the car keeps not slowing; the plan's
request then passes unchanged, and if StarPilot's bit is already up LongControl's ramp continues from the request last sent. A plan
asking to move ends a landing only when the plan is under way or StarPilot's bit is down; StarPilot's held bit then stays
unpublished while the plan asks to move, unless the model says stop in Experimental mode. `forceDecel`, FCW, driver override and
SafeMode are never landed.

Costs, simulated (re-driving the 18 logged stops on sixteen fitted variants of a Palisade's brake, not driven): behind a lead the
car rests a median 0.12 m and at worst 0.73 m further on and takes a median 0.5 s (at worst 2.3 s) longer; braking still on at the
wheel stop falls from 0.79 to 0.48 m/s² (median), and peak deceleration was never more than 0.02 m/s² higher. Enhanced Stops'
committed stops rest a median 0.34 m and at worst 2.2 m nearer their line and take 1.5-2.5 s longer on a nominal brake (up to
4.6 s on the weakest), with braking at the wheel stop 1.73 → 0.37 m/s² and peak deceleration 2.00 → 1.60. To pull stops back:
*Extra stopped-lead distance* on the Long Planner page (behind a lead) and Enhanced Stops' *Stop Distance Adjustment*
(`ForceStopDistanceOffset`). When a lead pulls away while the landed car still rolls at a crawl, StarPilot launches it at the
MPC's jerk limit rather than from standstill: up to 0.35 s later to 2 m/s.
`smooth_stops_preferences.supported()` excludes CAN FD, camera SCC, hybrids, plug-ins, EVs and fuel-cell cars (regenerative
braking hands over to the friction brake at kiss speeds), the Kona EV's legacy FCA11 path, the G90, blended and pedal longitudinal,
cars without openpilot longitudinal, and any car with a vehicle-owned `LongitudinalExtension`. Only one car, a `HYUNDAI_PALISADE`,
has been measured.