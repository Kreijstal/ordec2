#!/usr/bin/env python3
"""
Detailed demonstration of chunked simulation implementation.

This script shows the internals of how the subprocess backend uses
ngspice's stop after/step commands to implement async simulation.
"""

import sys
from ordec.sim2.ngspice_subprocess import NgspiceSubprocess

def demonstrate_chunked_commands():
    """Demonstrate the raw ngspice commands used for chunked simulation."""
    
    print("=" * 80)
    print("Chunked Simulation - Raw Ngspice Commands Demonstration")
    print("=" * 80)
    print()
    
    # Create a simple voltage divider netlist
    netlist = """Voltage Divider Test Circuit

V1 in 0 DC 3
R1 in a 1k
R2 a 0 2k

.end
"""
    
    print("1. Starting ngspice subprocess...")
    with NgspiceSubprocess.launch(debug=False) as ngspice:
        print("   ✓ ngspice started")
        print()
        
        print("2. Loading netlist...")
        ngspice.load_netlist(netlist)
        print("   ✓ Netlist loaded")
        print()
        
        print("3. Setting up chunked simulation:")
        print("   Command: 'stop after 5' - pause after 5 steps")
        ngspice.command("stop after 5")
        print("   ✓ Stop condition set")
        print()
        
        print("4. Starting transient analysis:")
        print("   Command: 'tran 1n 1p' - tstep=1ns, tstop=1ps")
        result = ngspice.command("tran 1n 1p")
        print("   ✓ Initial simulation started")
        if "stop" in result.lower() or "pause" in result.lower():
            print("   ✓ Simulation paused as expected after 5 steps")
        print()
        
        print("5. Retrieving first chunk of data:")
        print("   Command: 'print all'")
        data_lines = list(ngspice.print_all())
        print(f"   ✓ Retrieved {len(data_lines)} lines of data")
        print()
        
        # Show a sample of the data
        print("   Sample output (first 15 lines):")
        for i, line in enumerate(data_lines[:15]):
            if line.strip():
                print(f"   {line.rstrip()}")
            if i >= 14:
                break
        print()
        
        print("6. Continuing simulation:")
        print("   Command: 'step 5' - continue for 5 more steps")
        result = ngspice.command("step 5")
        print("   ✓ Simulation continued")
        if "stop" in result.lower() or "pause" in result.lower():
            print("   ✓ Simulation paused again after 5 more steps")
        print()
        
        print("7. Retrieving second chunk of data:")
        print("   Command: 'print all'")
        data_lines = list(ngspice.print_all())
        print(f"   ✓ Retrieved {len(data_lines)} lines of data")
        print()
        
        print("8. Continuing again:")
        print("   Command: 'step 5'")
        result = ngspice.command("step 5")
        print("   ✓ Simulation continued")
        print()
        
        print("9. Final data retrieval:")
        print("   Command: 'print all'")
        data_lines = list(ngspice.print_all())
        print(f"   ✓ Retrieved {len(data_lines)} lines of data")
        print()
        
        # Show final sample
        print("   Sample output (last 10 lines):")
        for line in data_lines[-10:]:
            if line.strip():
                print(f"   {line.rstrip()}")
        print()
    
    print("=" * 80)
    print("Demonstration complete!")
    print()
    print("Key takeaways:")
    print("1. 'stop after N' pauses the simulation after N steps")
    print("2. 'step N' continues the simulation for N more steps")
    print("3. 'print all' retrieves the current simulation data")
    print("4. Each call to 'print all' returns ALL data up to current time")
    print("5. The implementation must filter out duplicate time points")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_chunked_commands()
