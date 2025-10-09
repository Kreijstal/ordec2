# ORDeC2 Simulation Benchmarks

This file contains benchmarks for testing the performance of the ORDeC2 simulation framework across different backends.

## Usage

### Basic Usage

Run all benchmarks on all backends:
```bash
python benchmarks.py
```

### List Available Testbenches

```bash
python benchmarks.py --list-tests
```

### Run Specific Testbenches

Run only testbenches matching specific patterns:
```bash
python benchmarks.py --match ResdivFlatTb InvTb
```

### Select Backends

Run benchmarks on specific backends only:
```bash
python benchmarks.py --backend subprocess ffi
```

Available backends:
- `subprocess` - Ngspice subprocess backend (async not supported)
- `ffi` - Ngspice FFI backend (supports async)
- `mp` - Ngspice multiprocessing backend (supports async)

### Combined Options

Run specific testbenches on specific backends:
```bash
python benchmarks.py --match ResdivFlatTb LargeRingOsc --backend ffi mp
```

## Testbench Descriptions

### Basic Testbenches

- **ResdivFlatTb** - Simple resistor divider (flat hierarchy)
- **ResdivHierTb** - Resistor divider with hierarchical design
- **NmosSourceFollowerTb** - NMOS source follower circuit
- **InvTb** - Generic inverter testbench
- **RcFilterTb** - RC low-pass filter (includes AC analysis)
- **SimpleRCFilter** - Simple RC filter (AC analysis only)

### Technology-Specific Testbenches

- **InvSkyTb** - Sky130 inverter testbench
- **InvIhpTb** - IHP130 inverter testbench
- **Sky130Inverter** - Sky130 inverter with pulse input
- **IHP130Inverter** - IHP130 inverter with pulse input

### Large Circuit Testbenches

- **LargeRingOsc** - 21-stage ring oscillator (stress test)

## Simulation Types

Each testbench runs the following simulations:

1. **DC OP** - DC operating point analysis
2. **Transient** - Time-domain transient analysis
3. **Async Transient** - Asynchronous streaming transient analysis (ffi/mp backends only)
4. **AC** - AC frequency analysis (selected testbenches only)

## Notes

- The `subprocess` backend does not support async transient simulations
- Async simulations provide progress updates and streaming data
- Simulation times have been optimized for reasonable benchmark duration
- The timeout for each simulation is set to 300 seconds by default

## Performance Tips

For faster benchmarking:
- Use the `--match` option to run only needed testbenches
- Use the `--backend` option to test only specific backends
- Consider running on multiple cores if testing multiple backends

## Interpreting Results

The benchmark summary shows:
- **DC OP/Transient/AC** - Time taken in seconds
- **Async Transient** - Time taken and number of samples collected
- **Skipped** - Test not supported for the backend
- **Failed** - Test encountered an error

## Example Output

```
==================================================
               BENCHMARK SUMMARY
==================================================

Backend: ffi
-------------
  Testbench: ResdivFlatTb
    - Async Transient          : 2.0964 seconds (50 samples)
    - DC OP                    : 0.0151 seconds
    - Transient                : 0.0156 seconds
```
