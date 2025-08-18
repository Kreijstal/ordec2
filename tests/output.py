# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
import subprocess
import shutil
import io
import time
import select
import termios
import tty
import math
from ordec.lib.base import SinusoidalVoltageSource, Res, Gnd
from ordec.sim2.ngspice import Ngspice, NgspiceBackend
try:
    from ordec.sim2.ngspice import _FFIBackend
    FFI_CHECK_AVAILABLE = True
except ImportError:
    FFI_CHECK_AVAILABLE = False
from ordec.sim2.sim_hierarchy import HighlevelSim, SimHierarchy
from ordec.core import *
from ordec import Rational as R
from ordec import helpers


class SineWaveTestbench(Cell):
    frequency = Parameter(R, optional=True)
    amplitude = Parameter(R, optional=True)
    offset = Parameter(R, optional=True)
    delay = Parameter(R, optional=True)
    load_resistance = Parameter(R, optional=True)

    @generate
    def schematic(self):
        s = Schematic(cell=self)
        s.input_node = Net()
        s.gnd = Net()
        freq = self.params.get('frequency', R(1000))
        amp = self.params.get('amplitude', R(1))
        off = self.params.get('offset') or R(0)
        del_val = self.params.get('delay') or R(0)
        load_r = self.params.get('load_resistance') or R(1000)
        sine_source = SinusoidalVoltageSource(
            offset=off, amplitude=amp, frequency=freq, delay=del_val, damping_factor=R(0)
        ).symbol
        s.sine_source = SchemInstance(sine_source.portmap(p=s.input_node, m=s.gnd), pos=Vec2R(2, 5))
        load_resistor = Res(r=load_r).symbol
        s.load = SchemInstance(load_resistor.portmap(p=s.input_node, m=s.gnd), pos=Vec2R(8, 5))
        s.default_ground = s.gnd
        gnd_inst = Gnd().symbol
        s.gnd_inst = SchemInstance(gnd_inst.portmap(p=s.gnd), pos=Vec2R(5, 0))
        s.outline = Rect4R(lx=0, ly=0, ux=12, uy=10)
        helpers.schem_check(s, add_conn_points=True, add_terminal_taps=True)
        return s

    def sim_tran(self, tstep="1u", tstop="10m"):
        s = SimHierarchy(cell=self)
        sim = HighlevelSim(self.schematic, s)
        backend_to_use = NgspiceBackend.SUBPROCESS
        if FFI_CHECK_AVAILABLE:
            try:
                _FFIBackend.find_library()
                backend_to_use = NgspiceBackend.FFI
                print("[adf.py] FFI backend available, selecting.")
            except (OSError, Exception):
                print("[adf.py] FFI backend not found, falling back to subprocess.")
        with Ngspice.launch(backend=backend_to_use, debug=False) as ngspice_sim:
            backend_type = type(ngspice_sim._backend_impl).__name__
            print(f"Using {backend_type} backend")
            netlist = sim.netlister.out()
            ngspice_sim.load_netlist(netlist)
            result = ngspice_sim.tran(tstep, tstop)
            return result

def _read_terminal_response_bytes(timeout=0.5):
    response = b''
    start_time = time.time()
    while time.time() - start_time < timeout:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            char = sys.stdin.read(1)
            if char: response += char.encode('utf-8')
            else: break
        else:
            if response: break
    return response

def _query_sixel_support_from_terminal():
    if not sys.stdout.isatty() or not sys.stdin.isatty(): return False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        sys.stdout.write('\x1b[c')
        sys.stdout.flush()
        da1_response = _read_terminal_response_bytes(timeout=0.2)
        if da1_response and b'4' in da1_response: return True
        sys.stdout.write('\x1b[>0c')
        sys.stdout.flush()
        da2_response = _read_terminal_response_bytes(timeout=0.2)
        if da2_response and (b'4' in da2_response or b'sixel' in da2_response.lower()): return True
    except Exception: return False
    finally:
        if old_settings: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return False

def detect_terminal_capabilities():
    sixel_support = _query_sixel_support_from_terminal()
    display = os.environ.get('DISPLAY')
    return {'sixel': sixel_support, 'x11': bool(display), 'ascii': True}



def plot_sixel(time_data, voltage_data, title="Sine Wave"):
    """
    Generate and display a Sixel plot by using the matplotlib-backend-sixel module.
    This function will directly print the plot to the terminal.
    Returns True on success, False on failure.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        # Set the backend right before plotting. If this fails, the backend is not installed correctly.
        matplotlib.use('module://matplotlib-backend-sixel')

        # Standard matplotlib plotting commands
        plt.figure(figsize=(8, 4.5), dpi=100)
        plt.plot(time_data, voltage_data, color='cyan')
        plt.title(title, color='white')
        plt.xlabel("Time (s)", color='white')
        plt.ylabel("Voltage (V)", color='white')
        ax = plt.gca()
        ax.set_facecolor('#202020')
        plt.gcf().set_facecolor('#202020')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # This call now renders the plot directly to the terminal as Sixel
        plt.show()
        plt.close() # Clean up the figure from memory
        return True # Indicate success
        
    except Exception as e:
        print(f"\n--- Sixel plotting failed ---", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        print(f"  Please ensure 'matplotlib', 'matplotlib-backend-sixel', and 'imagemagick' are correctly installed.", file=sys.stderr)
        print(f"---------------------------------\n", file=sys.stderr)
        return False

def plot_ascii(time_data, voltage_data, width=80, height=20, title="Sine Wave"):
    # ... (function is unchanged) ...
    if not time_data or not voltage_data or len(time_data) != len(voltage_data): return "No data to plot"
    min_v, max_v = min(voltage_data), max(voltage_data)
    v_range = max_v - min_v if max_v != min_v else 1.0
    min_t, max_t = min(time_data), max(time_data)
    t_range = max_t - min_t if max_t != min_t else 1.0
    plot_height = height - 1
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    for i in range(len(time_data) - 1):
        t1, v1 = time_data[i], voltage_data[i]
        t2, v2 = time_data[i+1], voltage_data[i+1]
        x1 = int((width - 1) * (t1 - min_t) / t_range)
        x2 = int((width - 1) * (t2 - min_t) / t_range)
        y1_normalized = (v1 - min_v) / v_range
        y1 = 1 + int((plot_height - 2) * (1 - y1_normalized))
        y2_normalized = (v2 - min_v) / v_range
        y2 = 1 + int((plot_height - 2) * (1 - y2_normalized))
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(plot_height - 1, y1)), max(0, min(plot_height - 1, y2))
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            if 0 <= y1 < plot_height and 0 <= x1 < width: grid[y1][x1] = '*'
            continue
        x_inc, y_inc = dx / steps, dy / steps
        x, y = float(x1), float(y1)
        for _ in range(steps + 1):
            px, py = int(round(x)), int(round(y))
            if 0 <= py < plot_height and 0 <= px < width: grid[py][px] = '*'
            x += x_inc
            y += y_inc
    for y in range(plot_height): grid[y][0] = '|'
    for x in range(width): grid[height-1][x] = '-'
    grid[height-1][0] = '+'
    lines = [''.join(row) for row in grid]
    result = [f"{title:^{width}}"]
    result.extend(lines)
    result.append(f"Time: {min_t:.2e} to {max_t:.2e} s")
    result.append(f"Voltage: {min_v:.3f} to {max_v:.3f} V (range: {v_range:.3f})")
    return '\n'.join(result)

# In the plotting section of the script

def plot_x11(time_data, voltage_data, title="Sine Wave"):
    """Generate and display a plot in a separate X11 window."""
    try:
        import matplotlib
        # A common, often built-in backend is TkAgg
        matplotlib.use('TkAgg') 
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(time_data, voltage_data)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.tight_layout()
        
        print("Displaying plot in new window. Close the window to continue...")
        plt.show() # This call blocks until the window is closed
        plt.close()
        return True

    except ImportError:
        print("\n--- X11 plotting failed ---", file=sys.stderr)
        print("  Could not import a GUI backend (like tkinter).", file=sys.stderr)
        print("---------------------------------\n", file=sys.stderr)
        return False
    except Exception as e:
        # This catches errors like "no display name and no $DISPLAY environment variable"
        print(f"\n--- X11 plotting failed ---", file=sys.stderr)
        print(f"  Error: {e}", file=sys.stderr)
        print("---------------------------------\n", file=sys.stderr)
        return False


def display_sine_wave(time_data, voltage_data, title="Sine Wave"):
    """Display sine wave using best available method."""
    capabilities = detect_terminal_capabilities()

    print(f"\nTerminal capabilities detected:")
    print(f"  Sixel support: {capabilities['sixel']}")
    print(f"  X11 available: {capabilities['x11']}\n")

    if capabilities['sixel']:
        print("Attempting to use Sixel graphics backend...")
        if plot_sixel(time_data, voltage_data, title=title):
            return # Success

    # Add the X11 check here
    if capabilities['x11']:
        print("Attempting to use X11 backend...")
        if plot_x11(time_data, voltage_data, title=title):
            return # Success

    print("Falling back to ASCII art plotting:")
    print(plot_ascii(time_data, voltage_data, title=title))

def test_sine_wave_testbench_creation():
    tb = SineWaveTestbench(frequency=R(1000), amplitude=R(2), offset=R(0))
    assert tb is not None
def test_sine_wave_netlist_generation():
    tb = SineWaveTestbench(frequency=R(1000), amplitude=R(1), offset=R(0))
    from ordec.sim2.ngspice import Netlister
    netlister = Netlister()
    netlister.netlist_hier(tb.schematic)
    netlist = netlister.out()
    print("\nGenerated netlist:")
    print(netlist)
    assert 'SIN(' in netlist
def test_sine_wave_transient_simulation():
    tb = SineWaveTestbench(frequency=R(1000), amplitude=R(1), offset=R(0), load_resistance=R(1000))
    s = SimHierarchy(cell=tb)
    sim = HighlevelSim(tb.schematic, s)
    netlist = sim.netlister.out()
    assert 'SIN(' in netlist
    print("✓ Netlist generation and validation successful")
    result = tb.sim_tran(tstep="10u", tstop="2m")
    assert result is not None
    if result.time and len(result.time) > 1:
        print(f"Simulation successful! {len(result.time)} time points")
        if 'input_node' in result.signals:
            signal_data = result.signals['input_node']
            display_sine_wave(result.time, signal_data, title=f"Sine Wave {tb.params.get('frequency', R(1000)).compat_str()}Hz")
            print("✓ Full simulation test passed!")
        else: pytest.fail("Input node signal not found")
    else: pytest.fail("No time data in results")
def test_terminal_capabilities():
    caps = detect_terminal_capabilities()
    assert isinstance(caps, dict)
def test_ascii_plotting():
    time_points = [i * 0.001 for i in range(100)]
    voltage_points = [math.sin(2 * math.pi * 10 * t) for t in time_points]
    plot = plot_ascii(time_points, voltage_points, width=60, height=15, title="Test Sine")
    assert isinstance(plot, str)
def test_voltage_range_plotting():
    time_points = [i * 0.01 for i in range(50)]
    voltage_points = [2 * math.sin(2 * math.pi * t) - 1 for t in time_points]
    plot = plot_ascii(time_points, voltage_points, width=40, height=10, title="Range Test")
    min_v, max_v = min(voltage_points), max(voltage_points)
    lines = plot.split('\n')
    voltage_line = [line for line in lines if 'Voltage:' in line][0]
    assert f"{min_v:.3f}" in voltage_line
def test_sixel_detection_functionality():
    sixel_support = _query_sixel_support_from_terminal()
    assert isinstance(sixel_support, bool)


if __name__ == "__main__":
    test_sine_wave_transient_simulation()

