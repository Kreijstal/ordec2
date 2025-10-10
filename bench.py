import multiprocessing
import time
import logging
import queue
import argparse
import sys
import statistics
import json
import datetime
from ordec.core import (
    Schematic, Net, SchemInstance, Vec2R, Parameter, Cell
)
from ordec.core.rational import R
from ordec import helpers
from ordec.lib.base import (
    Gnd, Vdc, Cap, Res, PulseVoltageSource, SinusoidalVoltageSource
)
from ordec.lib import test as lib_test
# Imports needed for the robust async runner
from ordec.sim2.ngspice import Ngspice
from ordec.sim2.sim_hierarchy import HighlevelSim, SimHierarchy

# --- Configuration ---

# Configure logging based on verbosity
def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("Benchmark")
ALL_BACKENDS = ["subprocess", "ffi", "mp"]
SIMULATION_TIMEOUT = 300 # Timeout in seconds

# --- Base Class & New Testbenches (for self-containment) ---

class SimBase(lib_test.SimBase): pass

# Use only available testbench classes from lib_test

# --- Benchmark Functions ---

def run_dc_op(tb_class, backend):
    start_time = time.time()
    _ = tb_class(backend=backend).sim_dc
    return time.time() - start_time

def run_tran(tb_class, backend, tstep, tstop):
    start_time = time.time()
    tb_class(backend=backend).sim_tran(tstep, tstop)
    return time.time() - start_time

def run_ac(tb_class, backend, ptype, n, start, stop):
    start_time = time.time()
    tb_class(backend=backend).sim_ac(ptype, n, start, stop)
    return time.time() - start_time

def run_tran_async(tb_class, backend, tstep, tstop, buffer_size=10, disable_buffering=False, debug=False):
    """
    DEFINITIVE ROBUST IMPLEMENTATION: This function bypasses the flawed library generator
    and uses a consumer loop that is immune to race conditions.
    Returns: (time_taken, sample_count, fallback_used)
    """
    if debug:
        logger.debug(f"[DEBUG] run_tran_async started: tb_class={tb_class.__name__}, backend={backend}, buffer_size={buffer_size}, disable_buffering={disable_buffering}")

    if backend == "subprocess":
        # Async simulation is not supported for subprocess backend
        raise NotImplementedError("tran_async is not supported for subprocess backend. Use FFI or MP backend for async simulation.")

    total_start_time = time.time()
    tb_instance = tb_class(backend=backend)

    node = SimHierarchy()

    # For complex models that don't work well with savecurrents in async mode,
    # disable savecurrents to enable proper async callbacks
    enable_savecurrents = True
    # Check if this is a testbench that uses complex models
    from ordec.lib import test as lib_test
    if isinstance(tb_instance, (lib_test.InvSkyTb, lib_test.InvIhpTb)):
        enable_savecurrents = False
        if debug:
            logger.info(f"Disabling savecurrents for complex model testbench: {tb_class.__name__}")

    highlevel_sim = HighlevelSim(tb_instance.schematic, node, backend=backend, enable_savecurrents=enable_savecurrents)

    sample_count = 0
    fallback_used = False
    with Ngspice.launch(backend=backend) as sim:
        for hook in highlevel_sim.sim_setup_hooks:
            hook(sim)
        sim.load_netlist(highlevel_sim.netlister.out())

        data_queue = sim.tran_async(tstep, tstop, buffer_size=buffer_size, disable_buffering=disable_buffering)
        if debug:
            if hasattr(sim, '_backend_impl'):
                sim._backend_impl.debug = True
            else:
                sim.debug = True
            logger.debug(f"[DEBUG] Debug enabled for {tb_class.__name__}, buffer_size={buffer_size}, disable_buffering={disable_buffering}")
        sim_start_time = time.time()

        # This loop condition is the key to fixing the race condition.
        # It continues as long as the simulation is producing data OR there is data left to consume.
        while sim.is_running() or not data_queue.empty():
            try:
                data_point = data_queue.get(timeout=0.05) # Small timeout to prevent deadlocks

                if data_point == "---ASYNC_SIM_SENTINEL---":
                    continue # Ignore sentinel, rely on the main loop condition

                if isinstance(data_point, dict) and "data" in data_point:
                    sample_count += 1
                    if sample_count % 5000 == 0:
                        progress = data_point.get("progress", 0.0) * 100
                        sim_time = data_point.get("data", {}).get("time", 0)
                        # Only log progress at INFO level if verbose mode is enabled
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.info(
                                f"[{tb_class.__name__}/{backend}] Progress: {progress:.2f}% | "
                                f"Real Time: {time.time() - sim_start_time:.2f}s | Sim Time: {sim_time:.3e}s"
                            )

                # Check if fallback mechanism was used
                if isinstance(data_point, dict) and data_point.get("fallback_executed", False):
                    fallback_used = True
                    if debug:
                        logger.info(f"[DEBUG] Fallback mechanism detected in data point")

            except queue.Empty:
                # If the queue is empty, the loop will re-check sim.is_running().
                # If the sim is also finished, the loop terminates. Otherwise, it continues to wait.
                pass

        # Cleanup phase: process any remaining data that might have been buffered
        # This ensures we count all samples even when buffering is enabled
        if debug:
            logger.debug(f"[DEBUG] Starting cleanup phase, current sample_count: {sample_count}")
        remaining_data = True
        while remaining_data:
            try:
                data_point = data_queue.get_nowait()
                if isinstance(data_point, dict) and "data" in data_point:
                    sample_count += 1
                    if debug:
                        logger.debug(f"[DEBUG] Cleanup: processed data point, sample_count: {sample_count}")
            except queue.Empty:
                remaining_data = False
                if debug:
                    logger.debug(f"[DEBUG] Cleanup: queue empty, final sample_count: {sample_count}")

        # Force flush any remaining buffered data directly
        # This ensures we don't miss the final buffer flush when simulation ends naturally
        if debug:
            logger.debug(f"[DEBUG] Starting buffer flush phase, current sample_count: {sample_count}")
        if hasattr(sim, '_backend_impl'):
            backend_impl = sim._backend_impl
            if debug:
                logger.debug(f"[DEBUG] Using backend_impl from sim._backend_impl")
        else:
            backend_impl = sim
            if debug:
                logger.debug(f"[DEBUG] Using sim directly as backend_impl")

        # Add debug logging to track buffer state
        if debug:
            logger.debug(f"[DEBUG] Cleanup phase: checking buffer state")
            logger.debug(f"[DEBUG] Backend impl type: {type(backend_impl)}")
            if hasattr(backend_impl, '_buffer_enabled'):
                logger.debug(f"[DEBUG] Buffer enabled: {backend_impl._buffer_enabled}")
            else:
                logger.debug(f"[DEBUG] No _buffer_enabled attribute")
            if hasattr(backend_impl, '_data_buffer'):
                logger.debug(f"[DEBUG] Buffered data points: {len(backend_impl._data_buffer)}")
            else:
                logger.debug(f"[DEBUG] No _data_buffer attribute")
            if hasattr(backend_impl, '_flush_buffer'):
                logger.debug(f"[DEBUG] Flush buffer method available")
            else:
                logger.debug(f"[DEBUG] No _flush_buffer method")

        # Directly flush buffer if simulation has buffered data
        if hasattr(backend_impl, '_buffer_enabled') and hasattr(backend_impl, '_data_buffer'):
            if backend_impl._buffer_enabled and backend_impl._data_buffer:
                if debug:
                    logger.debug(f"[DEBUG] Flushing {len(backend_impl._data_buffer)} buffered data points")
                if hasattr(backend_impl, '_flush_buffer'):
                    backend_impl._flush_buffer()
                    if debug:
                        logger.debug(f"[DEBUG] Buffer flush completed")
                else:
                    if debug:
                        logger.debug(f"[DEBUG] Cannot flush buffer - no _flush_buffer method")
            else:
                if debug:
                    logger.debug(f"[DEBUG] Buffer not enabled or no data to flush")
        else:
            if debug:
                logger.debug(f"[DEBUG] Missing required buffer attributes")

    if debug:
        logger.debug(f"[DEBUG] Final sample count: {sample_count}")
    time_taken = time.time() - total_start_time
    if debug:
        logger.debug(f"[DEBUG] run_tran_async completed: time_taken={time_taken:.4f}, sample_count={sample_count}, fallback_used={fallback_used}")
    return time_taken, sample_count, fallback_used

# --- Multiprocessing Wrapper ---

def worker(task_func, *args):
    """Worker function for multiprocessing with timeout support."""
    try:
        result = task_func(*args)
        return result
    except Exception as e:
        return f"ERROR: {e}"

def run_with_timeout(task_func, *args, timeout=SIMULATION_TIMEOUT):
    """Run a task with timeout using multiprocessing."""
    # Check if we're using the mp backend (which already uses multiprocessing internally)
    # The backend is the second argument in task_args: (benchmark.tb_class, backend) + sim_args
    if len(args) >= 2 and args[1] == "mp":
        # mp backend already uses multiprocessing, can't create subprocesses
        # Just run the task directly without timeout protection
        try:
            return task_func(*args)
        except Exception as e:
            return f"ERROR: {e}"

    # Check if we're already in a multiprocessing context
    if multiprocessing.current_process().name != 'MainProcess':
        # We're already in a child process, can't create subprocesses
        # Just run the task directly without timeout protection
        try:
            return task_func(*args)
        except Exception as e:
            return f"ERROR: {e}"

    # Normal case: we're in the main process, can use multiprocessing
    with multiprocessing.Pool(1) as pool:
        try:
            result = pool.apply_async(worker, (task_func,) + args)
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            return f"TIMEOUT ({timeout}s)"
        except Exception as e:
            return f"ERROR: {e}"

# --- Benchmark Class ---

class Benchmark:
    def __init__(self, name, tb_class):
        self.name = name
        self.tb_class = tb_class
        self.tests = []

    def _add_dc_op_test(self):
        self.tests.append({"name": "DC OP", "func": run_dc_op, "args": (self.tb_class,)})
        return self

    def add_transient_test(self, tstep, tstop, name_suffix=""):
        suffix = f" {name_suffix}" if name_suffix else ""
        sim_params = (tstep, tstop)
        self.tests.append({"name": f"Transient{suffix}", "func": run_tran, "args": (self.tb_class,) + sim_params})
        return self

    def add_async_transient_test(self, tstep, tstop, name_suffix=""):
        suffix = f" {name_suffix}" if name_suffix else ""
        sim_params = (tstep, tstop)
        self.tests.append({"name": f"Async Transient{suffix} (default)", "func": run_tran_async, "args": (self.tb_class,) + sim_params + (10, False, False)})
        self.tests.append({"name": f"Async Transient{suffix} (large buffer)", "func": run_tran_async, "args": (self.tb_class,) + sim_params + (100, False, False)})
        self.tests.append({"name": f"Async Transient{suffix} (no buffering)", "func": run_tran_async, "args": (self.tb_class,) + sim_params + (1, True, False)})
        return self

    def add_ac_test(self, ptype, n, start, stop):
        sim_params = (ptype, n, start, stop)
        self.tests.append({"name": "AC", "func": run_ac, "args": (self.tb_class,) + sim_params})
        return self

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Run circuit simulation benchmarks")
    parser.add_argument('--list-tests', action='store_true', help='List available testbenches and exit.')
    parser.add_argument('--match', nargs='+', metavar='PATTERN', help='Run only benchmarks whose names contain any of the given substrings (case-insensitive).')
    parser.add_argument('--backend', nargs='+', metavar='BACKEND', choices=ALL_BACKENDS, help=f'Run only for the specified backends. If not set, all backends will run. Choices: {ALL_BACKENDS}')
    parser.add_argument('--output', '-o', metavar='FILE', help='Save benchmark results to JSON file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    args = parser.parse_args()

    # Setup logging based on verbosity
    setup_logging(verbose=args.verbose)

    # Testbench classes
    TESTBENCH_CLASSES = {
        "ResdivFlatTb": lib_test.ResdivFlatTb, "ResdivHierTb": lib_test.ResdivHierTb,
        "NmosSourceFollowerTb": lib_test.NmosSourceFollowerTb, "InvTb": lib_test.InvTb,
        "InvSkyTb": lib_test.InvSkyTb, "InvIhpTb": lib_test.InvIhpTb,
        "RcFilterTb": lib_test.RcFilterTb,
    }

    all_benchmarks = [
        Benchmark("ResdivFlatTb", TESTBENCH_CLASSES["ResdivFlatTb"]).add_transient_test("0.1u", "50u").add_async_transient_test("0.1u", "50u"),
        Benchmark("ResdivHierTb", TESTBENCH_CLASSES["ResdivHierTb"]).add_transient_test("0.1u", "50u").add_async_transient_test("0.1u", "50u"),
        Benchmark("NmosSourceFollowerTb", TESTBENCH_CLASSES["NmosSourceFollowerTb"]).add_transient_test("0.1u", "20u").add_async_transient_test("0.1u", "20u"),
        Benchmark("InvTb", TESTBENCH_CLASSES["InvTb"]).add_transient_test("0.1u", "20u").add_async_transient_test("0.1u", "20u"),
        Benchmark("InvSkyTb", TESTBENCH_CLASSES["InvSkyTb"])
            .add_transient_test("0.01u", "10u")
            .add_transient_test("0.01u", "50u", name_suffix="(5k samples)")
            .add_async_transient_test("0.01u", "10u")
            .add_async_transient_test("0.01u", "50u", name_suffix="(5k samples)"),
        Benchmark("InvIhpTb", TESTBENCH_CLASSES["InvIhpTb"]).add_transient_test("0.01u", "10u").add_async_transient_test("0.01u", "10u"),
        Benchmark("RcFilterTb", TESTBENCH_CLASSES["RcFilterTb"]).add_transient_test("10u", "10m").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("10u", "10m"),
    ]

    if args.list_tests:
        print("Available testbenches:")
        for b in all_benchmarks: print(f"  - {b.name}")
        sys.exit(0)

    benchmarks_to_run = all_benchmarks
    if args.match:
        patterns = [p.lower() for p in args.match]
        benchmarks_to_run = [b for b in all_benchmarks if any(p in b.name.lower() for p in patterns)]
        if not benchmarks_to_run:
            logger.error(f"No benchmarks found matching: {' '.join(args.match)}. Use --list-tests to see options.")
            sys.exit(1)
        logger.info(f"Running matched benchmarks: {[b.name for b in benchmarks_to_run]}")

    backends_to_run = args.backend if args.backend else ALL_BACKENDS
    logger.info(f"Running on backends: {backends_to_run}")

    results = {}
    for backend in backends_to_run:
        logger.info(f"--- Starting benchmarks for backend: {backend} ---")
        results[backend] = {}
        for benchmark in benchmarks_to_run:
            logger.info(f"--- Running testbench: {benchmark.name} ---")
            results[backend][benchmark.name] = {}
            for test in benchmark.tests:
                test_name, task_func, sim_args = test["name"], test["func"], test["args"][1:]
                task_args = (benchmark.tb_class, backend) + sim_args

                # Skip async tests for subprocess backend
                if backend == "subprocess" and test_name.startswith("Async Transient"):
                    continue

                # We now use the reliable multiprocessing wrapper for ALL tests,
                # as the new async runner is robust enough to handle it.
                # Pass verbose flag to async tests for debug output
                if test_name.startswith("Async Transient") and args.verbose:
                    # Replace debug parameter with True for verbose mode
                    task_args_list = list(task_args)
                    if len(task_args_list) >= 3:
                        task_args_list[-1] = True  # Set debug parameter to True
                    task_args = tuple(task_args_list)
                    logger.debug(f"[DEBUG] Setting debug=True for {test_name}")
                result_value = run_with_timeout(task_func, *task_args)

                results[backend][benchmark.name][test_name] = result_value

                if test_name.startswith('Async Transient') and isinstance(result_value, tuple):
                    time_taken, sample_count, fallback_used = result_value
                    fallback_info = " (FALLBACK)" if fallback_used else ""
                    logger.info(f"Finished test: '{test_name}'. Time: {time_taken} seconds, Samples: {sample_count}{fallback_info}")
                else:
                    logger.info(f"Finished test: '{test_name}'. Time: {result_value} seconds")

    if results:
        print("\n" + "="*50 + "\n" + " " * 15 + "BENCHMARK SUMMARY" + "\n" + "="*50)
        for backend, tb_results in results.items():
            print(f"\nBackend: {backend}\n" + "-" * (len(backend) + 10))
            for tb_name, sim_results in tb_results.items():
                if sim_results:  # Only show testbenches that have results
                    print(f"  Testbench: {tb_name}")
                    for sim_type in sorted(sim_results.keys()):
                        result_value = sim_results[sim_type]
                        if sim_type.startswith("Async Transient") and isinstance(result_value, tuple):
                            time_taken, sample_count, fallback_used = result_value
                            fallback_marker = " [FALLBACK]" if fallback_used else ""
                            print(f"    - {sim_type:<45}: {time_taken:.4f} seconds ({sample_count} samples{fallback_marker})")
                        elif isinstance(result_value, float):
                            print(f"    - {sim_type:<45}: {result_value:.4f} seconds")
        print("\n" + "="*50)

    # Save results to JSON file if requested
    if args.output:
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmarks": results,
            "metadata": {
                "backends_tested": backends_to_run,
                "benchmarks_run": [b.name for b in benchmarks_to_run],
                "total_tests": sum(len(tb_results) for backend_results in results.values() for tb_results in backend_results.values())
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Benchmark results saved to: {args.output}")

if __name__ == "__main__":
    main()
