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
from ordec.lib.generic_mos import Inv
# Imports needed for the robust async runner
from ordec.sim2.ngspice import Ngspice
from ordec.sim2.sim_hierarchy import HighlevelSim, SimHierarchy

# --- Configuration ---
def setup_logging(verbose=False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("Benchmark")
ALL_BACKENDS = ["subprocess", "ffi", "mp"]
SIMULATION_TIMEOUT = 300 # Timeout in seconds

# --- Base Classes & Custom Testbench Definitions ---

class SimBase(lib_test.SimBase): pass

# Simple Inverter definition, needed by LargeRingOsc
# This assumes it inherits from a class that sets up a basic inverter.
class Inv(lib_test.InvTb): pass

# The LargeRingOsc class, now included in the script.
class LargeRingOsc(SimBase):
    """Large ring oscillator for stress testing."""
    stages = Parameter(int, default=51)

    @helpers.generate
    def schematic(self):
        s = Schematic(cell=self)
        s.vdd = Net()
        s.vss = Net()

        for i in range(self.stages):
            s[f'net_{i}'] = Net()

        s.i_vdd = SchemInstance(Vdc(dc=R('1.8')).symbol.portmap(p=s.vdd, m=s.vss),
                                pos=Vec2R(0, (self.stages * 10)))
        s.i_gnd = SchemInstance(Gnd().symbol.portmap(p=s.vss), pos=Vec2R(0, -5))

        inv_sym = Inv().symbol
        for i in range(self.stages):
            input_net = s[f'net_{i-1}'] if i > 0 else s[f'net_{self.stages-1}']
            output_net = s[f'net_{i}']
            s[f'inv_{i}'] = SchemInstance(
                inv_sym.portmap(a=input_net, y=output_net, vdd=s.vdd, vss=s.vss),
                pos=Vec2R((i * 12), 8)
            )

        s.out_cap = SchemInstance(Cap(c=R('1p')).symbol.portmap(
            p=s[f'net_{self.stages-1}'], m=s.vss), pos=Vec2R((self.stages * 12), 8))

        helpers.schem_check(s, add_terminal_taps=True)
        return s

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
    if backend == "subprocess":
        raise NotImplementedError("tran_async is not supported for subprocess backend.")
    total_start_time = time.time()
    tb_instance = tb_class(backend=backend)
    node = SimHierarchy()
    enable_savecurrents = not isinstance(tb_instance, (lib_test.InvSkyTb, lib_test.InvIhpTb))
    highlevel_sim = HighlevelSim(tb_instance.schematic, node, backend=backend, enable_savecurrents=enable_savecurrents)
    sample_count = 0
    fallback_used = False
    with Ngspice.launch(backend=backend) as sim:
        for hook in highlevel_sim.sim_setup_hooks:
            hook(sim)
        sim.load_netlist(highlevel_sim.netlister.out())
        data_queue = sim.tran_async(tstep, tstop, buffer_size=buffer_size, disable_buffering=disable_buffering)
        while sim.is_running() or not data_queue.empty():
            try:
                data_point = data_queue.get(timeout=0.05)
                if isinstance(data_point, dict):
                    if "data" in data_point:
                        sample_count += 1
                    if data_point.get("fallback_executed", False):
                        fallback_used = True
            except queue.Empty:
                pass
    time_taken = time.time() - total_start_time
    return time_taken, sample_count, fallback_used

# --- Multiprocessing Wrapper ---

def worker(task_func, *args):
    try:
        return task_func(*args)
    except Exception as e:
        return f"ERROR: {e}"

def run_with_timeout(task_func, *args, timeout=SIMULATION_TIMEOUT):
    if (len(args) >= 2 and args[1] == "mp") or multiprocessing.current_process().name != 'MainProcess':
        try:
            return task_func(*args)
        except Exception as e:
            return f"ERROR: {e}"
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
        self.tests.append({"name": f"Transient{suffix}", "func": run_tran, "args": (self.tb_class, tstep, tstop)})
        return self
    def add_async_transient_test(self, tstep, tstop, name_suffix=""):
        suffix = f" {name_suffix}" if name_suffix else ""
        sim_params = (self.tb_class, tstep, tstop)
        self.tests.append({"name": f"Async Transient{suffix} (default)", "func": run_tran_async, "args": sim_params + (10, False, False), "buffer_config": {"buffer_size": 10, "disable_buffering": False}})
        self.tests.append({"name": f"Async Transient{suffix} (large buffer)", "func": run_tran_async, "args": sim_params + (100, False, False), "buffer_config": {"buffer_size": 100, "disable_buffering": False}})
        self.tests.append({"name": f"Async Transient{suffix} (no buffering)", "func": run_tran_async, "args": sim_params + (1, True, False), "buffer_config": {"buffer_size": 1, "disable_buffering": True}})
        return self
    def add_ac_test(self, ptype, n, start, stop):
        self.tests.append({"name": "AC", "func": run_ac, "args": (self.tb_class, ptype, n, start, stop)})
        return self

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Run circuit simulation benchmarks")
    parser.add_argument('-n', '--repetitions', type=int, default=1, metavar='N', help='Number of times to repeat each test for statistical analysis. Default: 1')
    parser.add_argument('--list-tests', action='store_true', help='List available testbenches and exit.')
    parser.add_argument('--match', nargs='+', metavar='PATTERN', help='Run only benchmarks whose names contain any of the given substrings (case-insensitive).')
    parser.add_argument('--backend', nargs='+', metavar='BACKEND', choices=ALL_BACKENDS, help=f'Run only for the specified backends. If not set, all backends will run. Choices: {ALL_BACKENDS}')
    parser.add_argument('--output', '-o', metavar='FILE', default='bench.json', help='Save benchmark results to JSON file. Default: bench.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging (DEBUG level)')
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    if args.repetitions > 1:
        logger.info(f"Running each test {args.repetitions} times.")

    TESTBENCH_CLASSES = {
        "ResdivFlatTb": lib_test.ResdivFlatTb, "ResdivHierTb": lib_test.ResdivHierTb,
        "NmosSourceFollowerTb": lib_test.NmosSourceFollowerTb, "InvTb": lib_test.InvTb,
        "InvSkyTb": lib_test.InvSkyTb, "InvIhpTb": lib_test.InvIhpTb,
        "RcFilterTb": lib_test.RcFilterTb,
        "LargeRingOsc": LargeRingOsc,
    }

    DISPLAY_NAME_MAP = {
        ("ResdivFlatTb", "DC OP"): "Resistor Divider (Flat, DC)",
        ("ResdivHierTb", "DC OP"): "Resistor Divider (Hier, DC)",
        ("NmosSourceFollowerTb", "DC OP"): "NMOS Follower (DC)",
        ("InvTb", "DC OP"): "Inverter (DC)",
        ("InvSkyTb", "DC OP"): "Sky130 Inverter (DC)",
        ("InvIhpTb", "DC OP"): "IHP130 Inverter (DC)",
        ("RcFilterTb", "DC OP"): "RC Filter (DC)",
        ("LargeRingOsc", "DC OP"): "51-stage Ring Oscillator (DC)",
        ("ResdivFlatTb", "Transient"): "Resistor Divider (Flat, Tran)",
        ("ResdivHierTb", "Transient"): "Resistor Divider (Hier, Tran)",
        ("NmosSourceFollowerTb", "Transient"): "NMOS Follower (Tran)",
        ("InvTb", "Transient"): "Inverter (Tran)",
        ("InvSkyTb", "Transient"): "Sky130 Inverter (Tran)",
        ("InvSkyTb", "Transient (5k samples)"): "Sky130 Inverter (5k)",
        ("InvIhpTb", "Transient"): "IHP130 Inverter (Tran)",
        ("RcFilterTb", "Transient"): "RC Filter (Tran)",
        ("LargeRingOsc", "Transient"): "51-stage Ring Oscillator (Tran)",
        ("ResdivFlatTb", "Async Transient (default)"): "Resistor Divider (Flat, Async)",
        ("ResdivHierTb", "Async Transient (default)"): "Resistor Divider (Hier, Async)",
        ("NmosSourceFollowerTb", "Async Transient (default)"): "NMOS Follower (Async)",
        ("InvTb", "Async Transient (default)"): "Inverter (Async)",
        ("InvSkyTb", "Async Transient (default)"): "Sky130 Inverter (Async)",
        ("InvIhpTb", "Async Transient (default)"): "IHP130 Inverter (Async)",
        ("RcFilterTb", "Async Transient (default)"): "RC Filter (Async)",
        ("LargeRingOsc", "Async Transient (default)"): "51-stage Ring Oscillator (Async)",
        ("ResdivFlatTb", "AC"): "Resistor Divider (Flat, AC)",
        ("ResdivHierTb", "AC"): "Resistor Divider (Hier, AC)",
        ("NmosSourceFollowerTb", "AC"): "NMOS Follower (AC)",
        ("InvTb", "AC"): "Inverter (AC)",
        ("InvSkyTb", "AC"): "Sky130 Inverter (AC)",
        ("InvIhpTb", "AC"): "IHP130 Inverter (AC)",
        ("RcFilterTb", "AC"): "RC Filter (AC)",
        ("LargeRingOsc", "AC"): "51-stage Ring Oscillator (AC)",
    }

    all_benchmarks = [
        Benchmark("ResdivFlatTb", TESTBENCH_CLASSES["ResdivFlatTb"])._add_dc_op_test().add_transient_test("0.1u", "50u").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.1u", "50u"),
        Benchmark("ResdivHierTb", TESTBENCH_CLASSES["ResdivHierTb"])._add_dc_op_test().add_transient_test("0.1u", "50u").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.1u", "50u"),
        Benchmark("NmosSourceFollowerTb", TESTBENCH_CLASSES["NmosSourceFollowerTb"])._add_dc_op_test().add_transient_test("0.1u", "20u").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.1u", "20u"),
        Benchmark("InvTb", TESTBENCH_CLASSES["InvTb"])._add_dc_op_test().add_transient_test("0.1u", "20u").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.1u", "20u"),
        Benchmark("InvSkyTb", TESTBENCH_CLASSES["InvSkyTb"])._add_dc_op_test().add_transient_test("0.01u", "10u").add_transient_test("0.01u", "50u", name_suffix="(5k samples)").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.01u", "10u").add_async_transient_test("0.01u", "50u", name_suffix="(5k samples)"),
        Benchmark("InvIhpTb", TESTBENCH_CLASSES["InvIhpTb"])._add_dc_op_test().add_transient_test("0.01u", "10u").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("0.01u", "10u"),
        Benchmark("RcFilterTb", TESTBENCH_CLASSES["RcFilterTb"])._add_dc_op_test().add_transient_test("10u", "10m").add_ac_test('dec', '10', '1', '1G').add_async_transient_test("10u", "10m"),

        Benchmark("LargeRingOsc", TESTBENCH_CLASSES["LargeRingOsc"])._add_dc_op_test().add_transient_test("0.1u", "50u").add_ac_test('dec', '10', '1', '1G'),
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

    backends_to_run = args.backend if args.backend else ALL_BACKENDS
    logger.info(f"Running on backends: {backends_to_run}")

    flat_results = []
    for backend in backends_to_run:
        logger.info(f"--- Starting benchmarks for backend: {backend} ---")
        for benchmark in benchmarks_to_run:
            logger.info(f"--- Running testbench: {benchmark.name} ---")
            for test in benchmark.tests:
                test_name, task_func, sim_args = test["name"], test["func"], test["args"]

                if backend == "subprocess" and "Async" in test_name:
                    logger.info(f"Skipping test: '{test_name}' for subprocess backend.")
                    continue

                logger.info(f"Running test: '{test_name}' ({args.repetitions} repetition(s))...")

                repetition_results = []
                for i in range(args.repetitions):
                    task_args_with_backend = (sim_args[0], backend) + sim_args[1:]
                    result_value = run_with_timeout(task_func, *task_args_with_backend)
                    repetition_results.append(result_value)

                default_display_name = f"{benchmark.name} - {test_name}"
                display_name = DISPLAY_NAME_MAP.get((benchmark.name, test_name), default_display_name)

                result_entry = {
                    "backend": backend, "testbench": benchmark.name, "test_name": test_name,
                    "display_name": display_name, "repetitions": args.repetitions
                }

                valid_results = [r for r in repetition_results if not isinstance(r, str)]
                if not valid_results:
                    result_entry["status"] = repetition_results[0] if repetition_results else "No valid results"
                else:
                    if isinstance(valid_results[0], tuple):
                        times = [r[0] for r in valid_results]
                        result_entry.update({
                            "time_mean": statistics.mean(times) if times else 0,
                            "time_median": statistics.median(times) if times else 0,
                            "time_stdev": statistics.stdev(times) if len(times) > 1 else 0,
                            "sample_count": valid_results[0][1],
                            "fallbacks": sum(1 for r in valid_results if r[2]),
                            "successful_runs": len(valid_results),
                            "all_times": times,
                            "buffer_config": test.get("buffer_config", {})
                        })
                    else:
                        times = valid_results
                        result_entry.update({
                            "time_mean": statistics.mean(times) if times else 0,
                            "time_median": statistics.median(times) if times else 0,
                            "time_stdev": statistics.stdev(times) if len(times) > 1 else 0,
                            "successful_runs": len(valid_results), "all_times": times
                        })
                flat_results.append(result_entry)
                logger.info(f"Finished test: '{test_name}'.")

    if flat_results:
        print("\n" + "="*50 + "\n" + " " * 15 + "BENCHMARK SUMMARY" + "\n" + "="*50)
        from collections import defaultdict
        grouped_results = defaultdict(lambda: defaultdict(list))
        for res in flat_results:
            grouped_results[res['backend']][res['testbench']].append(res)
        for backend, tb_results in grouped_results.items():
            print(f"\nBackend: {backend}\n" + "-" * (len(backend) + 10))
            for tb_name, sim_results in tb_results.items():
                print(f"  Testbench: {tb_name}")
                for result in sorted(sim_results, key=lambda x: x['test_name']):
                    sim_type = result['test_name']
                    if "status" in result:
                        print(f"    - {sim_type:<45}: {result['status']}")
                    else:
                        mean_time = result.get("time_mean", 0.0)
                        stdev = result.get("time_stdev", 0.0)
                        stdev_str = f" (± {stdev:.4f})" if result.get('successful_runs', 0) > 1 else ""
                        if "Async" in sim_type:
                            samples = result.get("sample_count", 0)
                            print(f"    - {sim_type:<45}: {mean_time:.4f}s{stdev_str} ({samples} samples)")
                        else:
                            print(f"    - {sim_type:<45}: {mean_time:.4f}s{stdev_str}")
        print("\n" + "="*50)

    if args.output:
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmarks": flat_results,
            "metadata": {
                "backends_tested": backends_to_run,
                "benchmarks_run": [b.name for b in benchmarks_to_run],
                "repetitions_per_test": args.repetitions,
            }
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Benchmark results saved to: {args.output}")

if __name__ == "__main__":
    main()
