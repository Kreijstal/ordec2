# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import termios
import tty
import select
import time
from ordec.core import *
from ordec import Rational as R
from ordec.lib.base import PulseVoltageSource, Res, Cap, Gnd
from ordec.sim2.ngspice import Ngspice, NgspiceBackend
from ordec.sim2.sim_hierarchy import SimHierarchy, HighlevelSim

# --- Sixel Plotting Function ---
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

# --- ASCII Plotting Fallback ---
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

# --- Terminal Sixel Support Detection ---
def _query_sixel_support_from_terminal():
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        return False
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        # First query with timeout
        sys.stdout.write('\x1b[c')
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)  # 200ms timeout
        if ready:
            # Read with timeout to prevent hanging
            response = b''
            start_time = time.time()
            while time.time() - start_time < 0.2 and len(response) < 10:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    chunk = sys.stdin.read(1)
                    response += chunk.encode('utf-8')
                    if len(response) >= 10:
                        break
            if b'4' in response:
                return True

        # Second query with timeout
        sys.stdout.write('\x1b[>0c')
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)  # 200ms timeout
        if ready:
            # Read with timeout to prevent hanging
            response = b''
            start_time = time.time()
            while time.time() - start_time < 0.2 and len(response) < 10:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    chunk = sys.stdin.read(1)
                    response += chunk.encode('utf-8')
                    if len(response) >= 10:
                        break
            if b'4' in response:
                return True

    except Exception:
        return False
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return False

def detect_terminal_capabilities():
    sixel_support = _query_sixel_support_from_terminal()
    return {'sixel': sixel_support, 'ascii': True}

# --- RC Testbench with Square Wave Source ---
class RCSquareWaveTb(Cell):
    """
    RC circuit testbench with square wave voltage source.
    Shows the classic RC integration curve.
    """

    @generate
    def schematic(self):
        s = Schematic(cell=self)
        s.vin = Net()
        s.vout = Net()
        s.gnd = Net()

        # Square wave source: 1kHz, 0-1V, 50% duty cycle
        s.source = SchemInstance(
            PulseVoltageSource(
                initial_value=R(0),
                pulsed_value=R(1),
                pulse_width=R("0.5m"),
                period=R("1m"),
                rise_time=R("1n"),
                fall_time=R("1n")
            ).symbol.portmap(p=s.vin, m=s.gnd),
            pos=Vec2R(2, 5)
        )

        # Resistor and capacitor
        s.R1 = SchemInstance(Res(r=R(1000)).symbol.portmap(p=s.vin, m=s.vout), pos=Vec2R(8, 5))
        s.C1 = SchemInstance(Cap(c=R("1u")).symbol.portmap(p=s.vout, m=s.gnd), pos=Vec2R(14, 5))

        s.gnd_inst = SchemInstance(Gnd().symbol.portmap(p=s.gnd), pos=Vec2R(8, 0))

        s.outline = Rect4R(lx=0, ly=0, ux=18, uy=10)
        return s

# --- VCD Generation Function ---
def generate_vcd_file(time_data, vin_data, vout_data, filename="rc_simulation.vcd"):
    """
    Generate a VCD file from simulation data.

    VCD format supports analog signals using real-valued variables.
    This creates a VCD file that can be viewed in waveform viewers.
    """
    try:
        with open(filename, 'w') as vcd_file:
            # VCD header
            vcd_file.write("$date\n")
            vcd_file.write(f"   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            vcd_file.write("$end\n")
            vcd_file.write("$version\n")
            vcd_file.write("   ORDeC VCD Generator\n")
            vcd_file.write("$end\n")
            vcd_file.write("$timescale 1us $end\n")

            # Variable definitions
            vcd_file.write("$scope module top $end\n")
            vcd_file.write("$var real 64 ! vin $end\n")  # 64-bit real for input voltage
            vcd_file.write("$var real 64 \" vout $end\n") # 64-bit real for output voltage
            vcd_file.write("$upscope $end\n")
            vcd_file.write("$enddefinitions $end\n")

            # Initial values
            vcd_file.write("#0\n")
            vcd_file.write(f"r{vin_data[0]} !\n")
            vcd_file.write(f"r{vout_data[0]} \"\n")

            # Value changes
            for i in range(1, len(time_data)):
                time_units = int(time_data[i] * 1e6)  # Convert to microseconds
                vcd_file.write(f"#{time_units}\n")
                vcd_file.write(f"r{vin_data[i]} !\n")
                vcd_file.write(f"r{vout_data[i]} \"\n")

        print(f"VCD file generated: {filename}")
        return True

    except Exception as e:
        print(f"Error generating VCD file: {e}")
        return False

# --- Main Simulation Function ---
def run_rc_square_wave_simulation():
    """Run RC circuit simulation with square wave input and plot results."""

    print("=== ORDeC RC Circuit with Square Wave Input ===")
    print("Creating RC circuit with R=1kΩ, C=1μF...")
    print("Square wave: 1kHz, 0-1V, 50% duty cycle")

    # Create testbench
    tb = RCSquareWaveTb()
    s = SimHierarchy(cell=tb)
    sim = HighlevelSim(tb.schematic, s)

    # Try to use FFI backend if available, otherwise subprocess
    backend_to_use = NgspiceBackend.SUBPROCESS
    try:
        from ordec.sim2.ngspice import _FFIBackend
        _FFIBackend.find_library()
        backend_to_use = NgspiceBackend.FFI
        print("Using FFI backend for faster simulation")
    except (ImportError, OSError):
        print("Using subprocess backend")

    with Ngspice.launch(backend=backend_to_use, debug=False) as ngspice_sim:
        netlist = sim.netlister.out()
        ngspice_sim.load_netlist(netlist)

        # Simulate for 5 periods to see the integration curve clearly
        print("Running transient simulation for 5ms...")
        result = ngspice_sim.tran("10u", "5m")

    if result and result.time:
        vin = result.get_signal('vin')
        vout = result.get_signal('vout')

        if not vin or not vout:
            print("Error: Could not find 'vin' or 'vout' signals")
            return False

        print(f"\nSimulation completed successfully!")
        print(f"Time points: {len(result.time)}")
        print(f"Final time: {result.time[-1]:.6f} s")

        # Detect terminal capabilities
        capabilities = detect_terminal_capabilities()
        print(f"Terminal capabilities: {capabilities}")

        # Plot results
        if capabilities['sixel']:
            print("\nDisplaying Sixel plot...")
            success = plot_sixel(result.time, vin, vout,
                               title="RC Circuit - Square Wave Response")
            if success:
                # Generate VCD file after successful plotting
                vcd_success = generate_vcd_file(result.time, vin, vout, "rc_simulation.vcd")
                if vcd_success:
                    print("\nVCD file also generated: rc_simulation.vcd")
                return True

        # Fallback to ASCII plot
        print("\nDisplaying ASCII plot (Sixel not available)...")
        ascii_plot = plot_ascii(result.time, vin, vout,
                               title="RC Circuit - Square Wave Response",
                               width=80, height=20)
        print(ascii_plot)

        # Generate VCD file
        vcd_success = generate_vcd_file(result.time, vin, vout, "rc_simulation.vcd")
        if vcd_success:
            print("\nVCD file generated: rc_simulation.vcd")
            print("You can view it with: gtkwave rc_simulation.vcd")

        return True
    else:
        print("Simulation failed to produce results")
        return False

if __name__ == "__main__":
    try:
        success = run_rc_square_wave_simulation()
        if success:
            print("\n✅ Simulation and plotting completed successfully!")
        else:
            print("\n❌ Simulation failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
