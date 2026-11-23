"""Persistent, default-off Smooth Stops preference and the vehicles it applies to; invalid bytes never activate it."""
from openpilot.starpilot.saved_source import read_saved

KEY = 'SmoothStops'
DESCRIPTION = ("Eases off the brakes through the last few feet of every stop so your car settles to a halt instead of lurching. " +
               "Stops can end a little further forward (in testing a few inches on average, at most about 2.5 feet) and take a " +
               "moment longer. Enhanced Stops' Stop Distance Adjustment pulls stops at lights and signs back. Gas Hyundai, Kia " +
               "and Genesis vehicles with the classic SCC radar.")


def supported(cp) -> bool:
  """Gas Hyundai, Kia and Genesis CAN cars whose stop runs through LongControl's own stopping state and the standard SCC stop
  request: the kind of car the corridor was measured on.

  CAN FD cars, hybrids, plug-ins, EVs and fuel cells (regenerative braking hands over to the friction brake at kiss speeds), the
  Kona EV's legacy FCA11 encoding, the G90's hold shaping, blended and pedal longitudinal, cars without openpilot longitudinal,
  and any car with a vehicle-owned LongControl policy (stopping rate, stopping or start policy) are left alone.
  """
  try:
    from opendbc.car.hyundai.values import CAR, HyundaiFlags, is_blended
    excluded = (HyundaiFlags.CANFD | HyundaiFlags.NON_SCC | HyundaiFlags.LEGACY | HyundaiFlags.UNSUPPORTED_LONGITUDINAL |
                HyundaiFlags.HYBRID | HyundaiFlags.EV | HyundaiFlags.FCEV)
    if (cp.brand != 'hyundai' or cp.carFingerprint not in CAR.__members__ or cp.passive or cp.dashcamOnly or cp.notCar or
        cp.flags & excluded or cp.carFingerprint == CAR.GENESIS_G90 or is_blended(cp)):
      return False
    from opendbc.car.hyundai.hyundaican import uses_legacy_fca11
    from opendbc.car.hyundai.ray_pedal import ray_pedal_enabled
    from openpilot.starpilot.longitudinal.extension import create_extension
    return not uses_legacy_fca11(cp) and not ray_pedal_enabled(cp) and create_extension(cp) is None
  except (AttributeError, IndexError, TypeError, ValueError):
    return False


class SmoothStopsPreferences:
  # the same bounded, once-a-second read as the other saved longitudinal preferences (Faster Takeoffs, Speed Nudge)
  def __init__(self, params):
    self.params = params
    self.enabled = False
    self.next_ns = 0
    self.last_ns = 0

  def sample(self, now_ns):
    if now_ns < self.last_ns:
      self.next_ns = 0
    self.last_ns = now_ns
    if now_ns >= self.next_ns:
      self.next_ns = now_ns + 1_000_000_000
      try:
        raw, readable = read_saved(self.params, KEY, 16)
        safe, safe_readable = read_saved(self.params, 'SafeMode', 16)
        self.enabled = readable and safe_readable and raw == b'1' and safe in (None, b'0')
      except (OSError, RuntimeError, TypeError, ValueError):
        self.enabled = False
    return self.enabled
