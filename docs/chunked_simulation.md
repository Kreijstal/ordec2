# Chunked Simulation Implementation for Subprocess Backend

## Overview

This document describes the implementation of chunked simulation for the subprocess backend in ORDeC2. The implementation enables asynchronous transient simulations using ngspice's `stop after` and `step` commands to break long simulations into manageable chunks.

## Background

The ngspice manual provides examples of using control flow commands to create nested loops and simulate circuits iteratively. This implementation adapts these concepts for asynchronous streaming simulation.

### Key Ngspice Commands Used

1. **`stop after N`**: Pauses simulation after N steps
2. **`step N`**: Continues simulation for N more steps  
3. **`print all`**: Outputs all current vector values

## Implementation Details

### Architecture

The chunked simulation is implemented in `ordec/sim2/ngspice_subprocess.py` with the following key components:

#### 1. Main Entry Point: `tran_async()`

```python
def tran_async(self, tstep, tstop=None, *extra_args, 
               throttle_interval: float = 0.1,
               buffer_size: int = 10,
               disable_buffering: bool = False,
               disable_throttling: bool = False,
               fallback_sampling_ratio: int = 100) -> "queue.Queue[dict]":
```

This method:
- Parses time parameters (tstep, tstop)
- Initializes async state (queue, thread, locks)
- Starts the chunked simulation in a separate thread
- Returns a queue for consuming simulation data points

#### 2. Chunked Simulation Loop: `_run_chunked_simulation()`

```python
def _run_chunked_simulation(self, tstep: float, tstop: float | None, 
                           tstep_str: str, throttle_interval: float):
```

This method:
1. Calculates optimal chunk size (aims for ~100 chunks per simulation)
2. Starts simulation with `stop after N` command
3. Runs `tran` command to begin transient analysis
4. Enters main loop:
   - Calls `print all` to get current simulation data
   - Parses and enqueues new data points
   - Checks if simulation is complete
   - Calls `step N` to continue simulation
5. Sends completion status when done

#### 3. Data Parsing: `_parse_and_enqueue_from_lines()`

```python
def _parse_and_enqueue_from_lines(self, lines: list, current_time: float, 
                                  chunk_end: float, tstop: float | None):
```

This method:
- Parses the output of `print all` command
- Extracts signal names and values from tabular data
- Filters out duplicate time points (already sent)
- Creates data point dictionaries with:
  - Timestamp
  - Signal data
  - Signal types (voltage, current, time)
  - Progress indicator
- Enqueues data points to the async queue

### Deduplication Strategy

A critical aspect of the implementation is avoiding duplicate data points. Since `print all` returns the entire simulation history up to the current point, we track `_async_current_time` to filter out previously sent data:

```python
# Only send data points that are newer than what we've already sent
if time_val <= self._async_current_time and self._data_points_sent > 0:
    continue
```

### Thread Safety

The implementation uses threading primitives for safe concurrent access:
- `_async_lock`: Protects shared state
- `_async_halt_requested`: Flag for graceful shutdown
- `_async_queue`: Thread-safe queue for data points

## Usage Example

```python
from ordec.lib import test as lib_test

# Create a test circuit
h = lib_test.ResdivFlatTb(backend="subprocess")

# Run async simulation
for result in h.sim_tran_async("0.1u", "5u"):
    print(f"Time: {result.time.value}, Signal: {result.a.value}")
    # Process data points as they arrive...
```

## Performance Characteristics

- **Chunk Size**: Automatically calculated based on simulation duration
  - For simulations with known tstop: `chunk_steps = max(5, total_steps // 100)`
  - Aims for ~100 chunks across the entire simulation
- **Throttling**: Configurable throttle interval (default 0.1s) prevents overwhelming the consumer
- **Memory**: Only current chunk data is held in memory, enabling large simulations

## Testing

The implementation includes comprehensive tests in `tests/test_sim2_async.py`:

1. **`test_highlevel_async_tran_basic[subprocess]`**: Basic functionality
2. **`test_highlevel_async_early_termination[subprocess]`**: Early exit handling
3. **`test_highlevel_async_tran_with_callback[subprocess]`**: Callback support

All tests verify:
- Monotonic time progression
- Correct signal values
- Progress tracking
- Early termination support

## Comparison with FFI/MP Backends

| Feature | Subprocess | FFI | MP |
|---------|-----------|-----|-----|
| Real-time data | ✓ (chunked) | ✓ (callbacks) | ✓ (callbacks) |
| Process isolation | ✓ | ✗ | ✓ |
| Overhead | Medium | Low | Medium |
| Complexity | Medium | High | High |

## Limitations

1. **Granularity**: Chunk size may be larger than individual time steps
2. **Latency**: Small delay between chunks due to `print all` overhead
3. **No alter support**: Unlike FFI backend, cannot modify circuit during simulation

## Future Enhancements

Potential improvements for the chunked simulation:

1. **Adaptive chunk sizing**: Adjust chunk size based on simulation progress
2. **Selective signal printing**: Only print requested signals instead of `print all`
3. **Resume support**: Implement proper pause/resume functionality
4. **Error recovery**: Better handling of simulation failures mid-stream

## References

- ngspice Manual: Chapter 13.6 (Control Structures)
- ngspice Manual: Section 13.5.52 (mrdump command)
- Example netlists in ngspice manual showing `.step` emulation

## See Also

- `examples/chunked_simulation_poc.py`: Proof of concept demonstration
- `ordec/sim2/ngspice_ffi.py`: FFI backend implementation for comparison
- `ordec/sim2/ngspice_mp.py`: Multiprocessing backend implementation
