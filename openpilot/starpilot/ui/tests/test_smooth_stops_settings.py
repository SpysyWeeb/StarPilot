from pathlib import Path
import tempfile

from opendbc.car.honda.interface import CarInterface as HondaInterface
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.common.params import Params, ParamKeyType
from openpilot.starpilot.galaxy.setting_descriptions import DESCRIPTIONS
from openpilot.starpilot.ui.feature_settings_owner import FeatureSettingsOwner
from openpilot.starpilot.ui.feature_settings_state import FeatureSettingsRequest


def snapshot(directory, cp):
  owner = FeatureSettingsOwner(Params(directory), lambda group: group == 'preferences', vehicle_fingerprint=lambda: None,
                               vehicle_params=lambda: cp)
  state = owner.snapshot('profiles', parked=True, system_long=True, lateral_context=False, metric=False)
  return owner, next((row for row in state.rows if row.key == 'SmoothStops'), None)


def test_smooth_stops_is_a_default_off_persistent_long_planner_toggle_for_supported_vehicles():
  with tempfile.TemporaryDirectory() as directory:
    params = Params(directory)
    assert params.get_type('SmoothStops') == ParamKeyType.BOOL
    assert params.get_default_value('SmoothStops') is False
    cp = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
    cp.openpilotLongitudinalControl, cp.pcmCruise = True, False
    owner, row = snapshot(directory, cp)
    assert row.label == 'Smooth Stops'
    assert row.value == row.default_value == 'Off'
    assert row.available and row.reason == DESCRIPTIONS['SmoothStops']
    assert owner.apply(FeatureSettingsRequest(row.key, row.source, 'On'))
    assert Path(params.get_param_path('SmoothStops')).read_bytes() == b'1'
    assert Params(directory).get_bool('SmoothStops')
    _, row = snapshot(directory, cp)
    assert row.value == 'On'
    assert owner.apply(FeatureSettingsRequest(row.key, b'1', 'Off'))
    assert Path(params.get_param_path('SmoothStops')).read_bytes() == b'0'


def test_smooth_stops_needs_starpilot_longitudinal_and_is_hidden_elsewhere():
  with tempfile.TemporaryDirectory() as directory:
    stock = CarInterface.get_non_essential_params(CAR.HYUNDAI_PALISADE)
    stock.openpilotLongitudinalControl, stock.pcmCruise = False, True
    _, row = snapshot(directory, stock)
    assert row is not None and not row.available
    for cp in (HondaInterface.get_non_essential_params(HONDA.HONDA_CIVIC), CarInterface.get_non_essential_params(CAR.KIA_EV9),
               CarInterface.get_non_essential_params(CAR.GENESIS_G90), None):
      assert snapshot(directory, cp)[1] is None
