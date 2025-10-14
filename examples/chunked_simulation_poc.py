#!/usr/bin/env python3
"""
Proof of concept for chunked simulation using subprocess backend.

This demonstrates the use of ngspice commands:
- stop after N: Stop simulation after N steps
- step N: Continue simulation for N more steps
- print all: Print all simulation vectors

Based on the ngspice documentation examples for .step emulation.
"""

from ordec.lib import test as lib_test

def main():
    print("=" * 80)
    print("Chunked Simulation Proof of Concept")
    print("=" * 80)
    print()
    
    # Create a simple test circuit
    print("Creating ResdivFlatTb test circuit...")
    h = lib_test.ResdivFlatTb(backend="subprocess")
    
    # Run async simulation with chunking
    print("Starting chunked async simulation...")
    print("Using commands: 'stop after N', 'step N', and 'print all'")
    print()
    
    data_points = []
    for i, result in enumerate(h.sim_tran_async("0.1u", "5u")):
        data_points.append(result)
        
        if i < 10 or i % 20 == 0:
            print(f"Point {i:4d}: time={result.time.value:12.6e} s, "
                  f"a={result.a.value:10.6f} V, progress={result.progress:6.2%}")
        
        # Stop after collecting enough samples for demo
        if i >= 50:
            print("...")
            break
    
    print()
    print(f"Collected {len(data_points)} data points")
    print()
    
    # Verify time monotonicity
    times = [dp.time.value for dp in data_points]
    is_monotonic = all(times[i] <= times[i+1] for i in range(len(times)-1))
    print(f"Time values are monotonically increasing: {is_monotonic}")
    
    if is_monotonic:
        print("✓ Chunked simulation working correctly!")
    else:
        print("✗ Time values are not monotonic - there may be an issue")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
