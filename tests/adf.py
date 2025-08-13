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
from ordec.lib.base import (
    SinusoidalVoltageSource, Res, Cap, Gnd,
    PulseVoltageSource, PieceWiseLinearVoltageSource
)
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

# --- Plotting & Terminal Detection Logic (utility functions) ---
def plot_sixel(time_data, voltage_in, voltage_out, title="RC Circuit Response"):
    """
    Generate and display a Sixel plot showing both input and output voltages.
    Returns True on success, False on failure.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('module://matplotlib-backend-sixel')

        plt.figure(figsize=(9, 5), dpi=100)
        plt.plot(time_data, voltage_in, color='#40E0D0', label='Vin (Input)')
        plt.plot(time_data, voltage_out, color='#FFD700', label='Vout (Capacitor)')
        plt.title(title, color='white')
        plt.xlabel("Time (s)", color='white')
        plt.ylabel("Voltage (V)", color='white')
        plt.legend()
        
        ax = plt.gca()
        ax.set_facecolor('#202020')
        plt.gcf().set_facecolor('#202020')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        plt.close()
        return True
    except Exception as e:
        print(f"\n--- Sixel plotting failed ---\n  Error: {e}\n  Ensure 'matplotlib-backend-sixel' and 'imagemagick' are installed.\n--------------------------\n", file=sys.stderr)
        return False

def plot_ascii(time_data, voltage_in, voltage_out, width=80, height=20, title="RC Circuit Response"):
    """Create ASCII art plot showing both input (I) and output (O) waveforms."""
    if not time_data or not voltage_in or not voltage_out: return "No data to plot"

    min_v, max_v = min(min(voltage_in), min(voltage_out)), max(max(voltage_in), max(voltage_out))
    v_range = max_v - min_v if max_v != min_v else 1.0
    min_t, max_t = min(time_data), max(time_data)
    t_range = max_t - min_t if max_t != min_t else 1.0
    plot_height = height - 1
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw both lines
    for i in range(len(time_data) - 1):
        for j, (v_data, char) in enumerate([(voltage_in, 'I'), (voltage_out, 'O')]):
            t1, v1 = time_data[i], v_data[i]
            t2, v2 = time_data[i+1], v_data[i+1]
            x1, x2 = int((width-1)*(t1-min_t)/t_range), int((width-1)*(t2-min_t)/t_range)
            y1, y2 = 1+int((plot_height-2)*(1-((v1-min_v)/v_range))), 1+int((plot_height-2)*(1-((v2-min_v)/v_range)))
            x1, x2, y1, y2 = max(0,min(width-1,x1)), max(0,min(width-1,x2)), max(0,min(plot_height-1,y1)), max(0,min(plot_height-1,y2))
            dx, dy = x2 - x1, y2 - y1
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                if 0 <= y1 < plot_height and 0 <= x1 < width: grid[y1][x1] = char
                continue
            x_inc, y_inc = dx / steps, dy / steps
            x, y = float(x1), float(y1)
            for _ in range(steps + 1):
                px, py = int(round(x)), int(round(y))
                if 0 <= py < plot_height and 0 <= px < width: grid[py][px] = char
                x += x_inc
                y += y_inc

    for y in range(plot_height): grid[y][0] = '|'
    for x in range(width): grid[height-1][x] = '-'
    grid[height-1][0] = '+'
    lines = [''.join(row) for row in grid]
    result = [f"{title:^{width}}", *lines, f"Time: {min_t:.2e} to {max_t:.2e} s", f"Voltage: {min_v:.3f} to {max_v:.3f} V | (I)nput, (O)utput"]
    return '\n'.join(result)

def _query_sixel_support_from_terminal():
    if not sys.stdout.isatty() or not sys.stdin.isatty(): return False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        sys.stdout.write('\x1b[c'); sys.stdout.flush()
        if b'4' in sys.stdin.read(10).encode('utf-8'): return True
        sys.stdout.write('\x1b[>0c'); sys.stdout.flush()
        if b'4' in sys.stdin.read(10).encode('utf-8'): return True
    except Exception: return False
    finally:
        if old_settings: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return False

def detect_terminal_capabilities():
    return {'sixel': _query_sixel_support_from_terminal(), 'ascii': True}

# --- Reusable Testbench Cell ---
# --- Reusable Testbench Cell ---
class RCTestbench(Cell):
    """
    A generic RC circuit testbench.
    The `source_cell` parameter determines the voltage source to use.
    """
    source_cell = Parameter(Cell)

    @generate
    def schematic(self):
        s = Schematic(cell=self)
        s.vin = Net()
        s.vout = Net()
        s.gnd = Net()

        # Instantiate the provided voltage source
        s.source = SchemInstance(
            self.source_cell.symbol.portmap(p=s.vin, m=s.gnd),
            pos=Vec2R(2, 5)
        )
        s.R1 = SchemInstance(Res(r=R(1000)).symbol.portmap(p=s.vin, m=s.vout), pos=Vec2R(8, 5))
        
        # --- CORRECTED LINE: Removed 'F' from the Rational string ---
        s.C1 = SchemInstance(Cap(c=R('1u')).symbol.portmap(p=s.vout, m=s.gnd), pos=Vec2R(14, 5))
        
        s.gnd_inst = SchemInstance(Gnd().symbol.portmap(p=s.gnd), pos=Vec2R(8, 0))

        s.outline = Rect4R(lx=0, ly=0, ux=18, uy=10)
        helpers.schem_check(s, add_conn_points=True, add_terminal_taps=True)
        return s
# --- Reusable Simulation and Plotting Function ---
def run_and_plot_simulation(tb_cell, title, tstop="5m"):
    """
    Takes a testbench cell, simulates it, and plots the result.
    """
    print(f"\n--- {title} ---")
    s = SimHierarchy(cell=tb_cell)
    sim = HighlevelSim(tb_cell.schematic, s)

    backend_to_use = NgspiceBackend.SUBPROCESS
    if FFI_CHECK_AVAILABLE:
        try:
            _FFIBackend.find_library()
            backend_to_use = NgspiceBackend.FFI
        except (OSError, Exception): pass

    with Ngspice.launch(backend=backend_to_use, debug=False) as ngspice_sim:
        netlist = sim.netlister.out()
        ngspice_sim.load_netlist(netlist)
        # Simulate for 5 periods of a 1kHz signal for good resolution
        result = ngspice_sim.tran(tstep="10u", tstop=tstop)

    if result and result.time:
        vin = result.get_signal('vin')
        vout = result.get_signal('vout')
        if not vin or not vout:
            print("Error: Could not find 'vin' or 'vout' signals in simulation result.")
            return

        capabilities = detect_terminal_capabilities()
        if capabilities['sixel']:
            success = plot_sixel(result.time, vin, vout, title=title)
            if success:
                return

        plot_ascii(result.time, vin, vout, title=title)
    else:
        print("Simulation failed to produce results.")

if __name__ == "__main__":
    # 1. Define the Sine Wave Source and run the simulation
    sine_source = SinusoidalVoltageSource(amplitude=R(1), frequency=R(1000))
    sine_tb = RCTestbench(source_cell=sine_source)
    run_and_plot_simulation(sine_tb, "Sine Wave RC Circuit (1kHz)", tstop="5m")

    # 2. Define the Square Wave (Pulse) Source and run the simulation
    square_wave_source = PulseVoltageSource(
        initial_value=R(0),
        pulsed_value=R(1),
        pulse_width=R("0.5m"), # 50% duty cycle for 1kHz
        period=R("1m"),
        rise_time=R("1n"), # Ideal rise/fall
        fall_time=R("1n")
    )
    square_tb = RCTestbench(source_cell=square_wave_source)
    run_and_plot_simulation(square_tb, "Square Wave RC Circuit (1kHz)", tstop="5m")

    # 3. Define the Sawtooth Wave (PWL) Source and run the simulation
    sawtooth_wave_source = PieceWiseLinearVoltageSource(
        V=[(0, 0), ("1m", 1), ("1.001m", 0), ("2m", 1), ("2.001m", 0), ("3m", 1), ("3.001m", 0), ("4m", 1), ("4.001m", 0), ("5m", 1)]
    )
    sawtooth_tb = RCTestbench(source_cell=sawtooth_wave_source)
    run_and_plot_simulation(sawtooth_tb, "Sawtooth Wave RC Circuit (1kHz)", tstop="5m")
