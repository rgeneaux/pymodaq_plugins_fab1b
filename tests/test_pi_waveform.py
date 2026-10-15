import numpy as np
from qtpy import QtWidgets

from pymodaq.control_modules.move_utility_classes import DataActuator
from pymodaq_plugins_fab1b.daq_move_plugins import daq_move_PI_Waveform as module
from pymodaq_plugins_fab1b.daq_move_plugins.daq_move_PI_Waveform import (
    DAQ_Move_PI_Waveform, DIRECTION_DOWN, DIRECTION_UP, INPUT_FREQUENCY, INPUT_SPEED, SHAPE_SINE, SHAPE_TRIANGLE,
    UM_PER_FS, corner_points, delay_to_position, frequency_from_speed, plan_table, position_range,
    position_to_delay, speed_from_frequency,
)

SERVO_DT = 20e-6  # E-754


# ---- pure functions ---------------------------------------------------------------------------------------------
def test_plan_table_gives_the_requested_frequency():
    for frequency in (0.001, 0.1, 0.5, 5., 100., 1000.):
        n_points, rate, actual = plan_table(frequency, SERVO_DT)
        assert n_points % 2 == 0 and 10 <= n_points <= module.TABLE_POINTS
        assert abs(actual - frequency) / frequency < 1e-3
        assert abs(1 / (n_points * rate * SERVO_DT) - actual) < 1e-12


def test_plan_table_slows_the_table_rate_for_slow_sweeps():
    assert plan_table(5., SERVO_DT)[1] == 1  # 10000 points at the servo rate
    assert plan_table(0.5, SERVO_DT)[1] > 1
    assert plan_table(0.001, SERVO_DT)[0] <= module.TABLE_POINTS


def test_plan_table_rejects_impossible_frequencies():
    for frequency in (0., -1., 1e6):
        try:
            plan_table(frequency, SERVO_DT)
        except ValueError:
            continue
        raise AssertionError(f"{frequency} Hz should be refused")


def test_corner_points():
    assert corner_points(0., 1000) == 0
    assert corner_points(20., 400) == 40  # 20 % of each half period (200 points)


def test_retroreflector_conversions():
    assert abs(UM_PER_FS - 0.149896229) < 1e-9  # c / 2: the light travels twice the displacement
    assert abs(delay_to_position(10., 125.) - (125. + 10 * UM_PER_FS)) < 1e-12
    assert abs(delay_to_position(10., 125., increasing=False) - (125. - 10 * UM_PER_FS)) < 1e-12
    for increasing in (True, False):
        for delay in (-35., 0., 12.5):
            position = delay_to_position(delay, 123.4, increasing)
            assert abs(position_to_delay(position, 123.4, increasing) - delay) < 1e-9


def test_position_range_is_ordered_whatever_the_direction():
    low, high = position_range(125., -20., 40., increasing=True)
    assert abs(low - (125 - 20 * UM_PER_FS)) < 1e-9 and abs(high - (125 + 40 * UM_PER_FS)) < 1e-9
    low, high = position_range(125., -20., 40., increasing=False)  # the stage moves the other way
    assert abs(low - (125 - 40 * UM_PER_FS)) < 1e-9 and abs(high - (125 + 20 * UM_PER_FS)) < 1e-9


def test_speed_and_frequency_are_two_ways_to_say_the_same_thing():
    assert frequency_from_speed(60., 60.) == 0.5  # a 60 fs span at 60 fs/s: 1 s each way
    assert speed_from_frequency(0.5, 60.) == 60.
    for speed, span in ((1., 5.), (80., 40.), (500., 120.)):
        assert abs(speed_from_frequency(frequency_from_speed(speed, span), span) - speed) < 1e-9


# ---- the plugin, with a fake controller ---------------------------------------------------------------------------
class FakeDevice:
    """Records the GCS commands the plugin sends"""
    bufstate = True

    def __init__(self):
        self.commands = []
        self.table = np.array([122., 128.])
        self.generator_running = True

    def __getattr__(self, name):
        if name.startswith(('WAV', 'WGO', 'WCL', 'WSL', 'WTR', 'WGC', 'MOV')):
            return lambda *a, **k: self.commands.append((name, a, k))
        raise AttributeError(name)

    def qGWD(self, *args):
        self.commands.append(('qGWD', args, {}))

    @property
    def bufdata(self):
        return self.table

    def IsGeneratorRunning(self, gen):
        return {gen: self.generator_running}


class FakeController:
    def __init__(self):
        self.device = FakeDevice()
        self.servo = True
        self.position = 125.

    def get_servo(self, axis):
        return self.servo

    def get_axis_position(self, axis):
        return self.position

    def move_absolute(self, axis, position):
        self.device.commands.append(('move_absolute', (axis, position), {}))

    def stop(self):
        self.device.commands.append(('stop', (), {}))


def _plugin():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    module.pitools.waitontarget = lambda *a, **k: None
    module.time.sleep = lambda *a, **k: None
    p = DAQ_Move_PI_Waveform(None, None)
    p.emit_status = lambda cmd: None
    p.controller = FakeController()
    p.axis_names = ['1']
    p._servo_dt, p._max_wave_points, p._has_wave_generator = SERVO_DT, 1048576, True
    p.settings.child('axis_infos', 'min').setValue(0.)
    p.settings.child('axis_infos', 'max').setValue(250.)
    # time zero at 125 um, sweep -20 .. +20 fs (a 5.996 um stroke), 40 fs/s: 0.5 Hz
    for name, value in (('time_zero', 125.), ('delay_min', -20.), ('delay_max', 20.), ('speed', 40.), ('cycles', 0)):
        p.settings.child('waveform', name).setValue(value)
    return p


LOW, HIGH = position_range(125., -20., 20.)


def _names(p):
    return [c[0] for c in p.controller.device.commands]


def test_triangle_is_the_default_shape():
    p = _plugin()
    assert p.settings['waveform', 'shape'] == SHAPE_TRIANGLE
    assert DAQ_Move_PI_Waveform.params[-1]['children'][0]['value'] == SHAPE_TRIANGLE


def test_start_sequence_of_a_triangle():
    p = _plugin()
    p.start_waveform()
    names = _names(p)
    assert names[:3] == ['WGO', 'WCL', 'WAV_RAMP'], names  # stop, clear, define
    assert names == ['WGO', 'WCL', 'WAV_RAMP', 'qGWD', 'WSL', 'WTR', 'WGC', 'MOV', 'WGO'], names
    assert p.controller.device.commands[names.index('MOV')][1] == ('1', LOW)  # to the lowest position first
    cmd = {c[0]: c for c in p.controller.device.commands}
    n_points, rate, _ = plan_table(0.5, SERVO_DT)
    assert cmd['WAV_RAMP'][2] == dict(table=1, firstpoint=0, numpoints=n_points, append='X', center=n_points // 2,
                                      speedupdown=0, amplitude=HIGH - LOW, offset=LOW, seglength=n_points)
    assert cmd['WTR'][1] == (0, rate, 1) and cmd['WGC'][1] == (1, 0)
    assert p.controller.device.commands[-1][1] == (1, 1)  # WGO(1, 1) is the very last command: start
    assert p.settings['waveform', 'running'] and p._waveform_running


def test_sine_uses_the_sine_table_and_corner_smoothing_the_ramp_option():
    p = _plugin()
    p.settings.child('waveform', 'shape').setValue(SHAPE_SINE)
    p.start_waveform()
    assert 'WAV_SIN_P' in _names(p) and 'WAV_RAMP' not in _names(p)

    p = _plugin()
    p.settings.child('waveform', 'corner_smoothing').setValue(20.)
    p.start_waveform()
    n_points = plan_table(0.5, SERVO_DT)[0]
    ramp = next(c for c in p.controller.device.commands if c[0] == 'WAV_RAMP')
    assert ramp[2]['speedupdown'] == corner_points(20., n_points) > 0


def test_a_waveform_leaving_the_travel_range_is_refused_before_anything_is_sent():
    p = _plugin()
    p.settings.child('waveform', 'delay_max').setValue(2000.)  # 2000 fs = 300 um of stage
    try:
        p.start_waveform()
    except ValueError as exc:
        assert 'travel range' in str(exc)
    else:
        raise AssertionError('should have been refused')
    assert p.controller.device.commands == [] and not p._waveform_running


def test_module_bounds_are_respected():
    p = _plugin()
    p.settings.child('bounds', 'is_bounds').setValue(True)
    p.settings.child('bounds', 'min_bound').setValue(124.)
    p.settings.child('bounds', 'max_bound').setValue(126.)
    try:
        p.start_waveform()
    except ValueError as exc:
        assert 'bounds' in str(exc)
    else:
        raise AssertionError('should have been refused')
    assert p.controller.device.commands == []


def test_servo_off_is_refused():
    p = _plugin()
    p.controller.servo = False
    try:
        p.start_waveform()
    except RuntimeError as exc:
        assert 'closed loop' in str(exc)
    else:
        raise AssertionError('should have been refused')
    assert p.controller.device.commands == []


def test_a_table_outside_the_requested_range_is_not_started():
    p = _plugin()
    p.controller.device.table = np.array([100., 128.])  # the controller did not build what was asked
    try:
        p.start_waveform()
    except RuntimeError as exc:
        assert 'not within the requested range' in str(exc)
    else:
        raise AssertionError('should have been refused')
    assert not p._waveform_running
    last_go = [c for c in p.controller.device.commands if c[0] == 'WGO'][-1]
    assert last_go[1] == (1, 0)  # the generator is left stopped


def test_the_rate_is_set_by_the_speed_or_by_the_frequency():
    p = _plugin()  # speed typed: 40 fs/s over 40 fs
    assert p.settings['waveform', 'rate_input'] == INPUT_SPEED
    assert abs(p._requested_frequency() - 0.5) < 1e-12
    p._update_plan_info()
    assert abs(p.settings['waveform', 'frequency'] - 0.5) < 1e-12  # the computed field follows
    assert abs(p.settings['waveform', 'actual_speed'] - 40.) < 0.01

    p.settings.child('waveform', 'rate_input').setValue(INPUT_FREQUENCY)  # now the frequency is typed
    p.settings.child('waveform', 'frequency').setValue(2.)
    assert abs(p._requested_frequency() - 2.) < 1e-12
    p._update_plan_info()
    assert abs(p.settings['waveform', 'speed'] - 160.) < 1e-9  # 2 * 40 fs * 2 Hz
    assert abs(p.settings['waveform', 'actual_speed'] - 160.) < 0.1


def test_the_table_is_built_for_the_requested_speed():
    p = _plugin()
    p.settings.child('waveform', 'speed').setValue(100.)  # 100 fs/s over 40 fs: 1.25 Hz
    p.start_waveform()
    n_points, rate, _ = plan_table(1.25, SERVO_DT)
    wtr = next(c for c in p.controller.device.commands if c[0] == 'WTR')
    assert wtr[1] == (0, rate, 1)
    assert n_points * rate * SERVO_DT > 0 and abs(1 / (n_points * rate * SERVO_DT) - 1.25) < 1e-3
    assert abs(p.settings['waveform', 'actual_speed'] - 100.) < 0.1
    assert '100' in p.settings['waveform', 'status'] and 'fs/s' in p.settings['waveform', 'status']


def test_delay_max_must_exceed_delay_min():
    p = _plugin()
    p.settings.child('waveform', 'delay_max').setValue(-30.)
    try:
        p.start_waveform()
    except ValueError as exc:
        assert 'greater' in str(exc)
    else:
        raise AssertionError('should have been refused')
    assert p.controller.device.commands == []


def test_a_decreasing_direction_moves_the_sweep_to_the_other_side_of_time_zero():
    p = _plugin()
    p.settings.child('waveform', 'delay_min').setValue(-20.)
    p.settings.child('waveform', 'delay_max').setValue(60.)
    p.settings.child('waveform', 'delay_direction').setValue(DIRECTION_DOWN)
    p.start_waveform()
    low, high = position_range(125., -20., 60., increasing=False)
    assert low < 125. < high and abs((125. - low) - 60 * UM_PER_FS) < 1e-9  # the +60 fs end is below time zero
    move = next(c for c in p.controller.device.commands if c[0] == 'MOV')
    assert abs(move[1][1] - low) < 1e-9
    assert 'from 60 fs' in p.settings['waveform', 'status']  # it starts at the delay of the lowest position


def test_the_current_delay_is_shown():
    p = _plugin()
    p.controller.position = 125. + 10 * UM_PER_FS
    p.get_actuator_value()
    assert abs(p.settings['waveform', 'current_delay'] - 10.) < 1e-3


def test_a_move_command_stops_the_waveform_first():
    p = _plugin()
    p.start_waveform()
    p.controller.device.commands.clear()
    p.move_abs(DataActuator(p.axis_name, data=125., units=p.axis_unit))
    names = _names(p)
    assert names[0] == 'WGO' and p.controller.device.commands[0][1] == (1, 0)
    assert 'move_absolute' in names and not p._waveform_running


def test_end_of_a_finite_waveform_is_noticed():
    p = _plugin()
    p.settings.child('waveform', 'cycles').setValue(3)
    p.start_waveform()
    p.get_actuator_value()
    assert p._waveform_running  # still running
    p.controller.device.generator_running = False
    p.get_actuator_value()
    assert not p._waveform_running and p.settings['waveform', 'status'] == 'Finished'


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:
                failed += 1
                print("FAIL", name, type(exc).__name__, exc)
    sys.exit(failed)
