"""Persistent, default-off Smooth Stops preference and the vehicles it applies to; invalid bytes never activate it."""
from openpilot.starpilot.saved_source import read_saved

KEY = 'SmoothStops'


def supported(cp) -> bool:
  """Hyundai, Kia and Genesis CAN cars whose stop runs through LongControl's own stopping state and the SCC stop request.

  CAN FD cars, the G90's hold shaping, blended and pedal longitudinal, and any car with a vehicle-owned LongControl policy
  (stopping rate, stopping or start policy) bring the car to rest their own way and are left alone, as are cars without
  openpilot longitudinal support.
  """
  try:
    from opendbc.car.hyundai.values import CAR, HyundaiFlags, is_blended
    excluded = HyundaiFlags.CANFD | HyundaiFlags.NON_SCC | HyundaiFlags.LEGACY | HyundaiFlags.UNSUPPORTED_LONGITUDINAL
    if (cp.brand != 'hyundai' or cp.carFingerprint not in CAR.__members__ or cp.passive or cp.dashcamOnly or cp.notCar or
        cp.flags & excluded or cp.carFingerprint == CAR.GENESIS_G90 or is_blended(cp)):
      return False
    from opendbc.car.hyundai.ray_pedal import ray_pedal_enabled
    from openpilot.starpilot.longitudinal.extension import create_extension
    return not ray_pedal_enabled(cp) and create_extension(cp) is None
  except (AttributeError, IndexError, TypeError, ValueError):
    return False


class SmoothStopsPreferences:
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
