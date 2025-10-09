import multiprocessing
import time
import logging
import queue
import argparse
import sys
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")
ALL_BACKENDS = ["subprocess", "ffi", "mp"]
SIMULATION_TIMEOUT = 300 # Timeout in seconds

# --- Base Class & New Testbenches (for self-containment) ---
class SimBase(lib_test.SimBase): pass
class LargeRingOsc(SimBase):
    stages = Parameter(int, default=51)
    @helpers.generate
    def schematic(self):
        from ordec.lib.generic_mos import Inv
        s = Schematic(cell=self)
        s.vdd, s.vss = Net(), Net()
        stage_nets = [Net(name=f"net_{i}") for i in range(self.stages)]
        s.add(stage_nets)
        s.i_vdd = SchemInstance(Vdc(dc=R('1.8')).symbol.portmap(p=s.vdd, m=s.vss), pos=Vec2R(0, (self.stages * 10)))
        s.i_gnd = SchemInstance(Gnd().symbol.portmap(p=s.vss), pos=Vec2R(0, -5))
        inv_sym = Inv().symbol
        for i in range(self.stages):
            s[f'inv_{i}'] = SchemInstance(inv_sym.portmap(a=stage_nets[i-1], y=stage_nets[i], vdd=s.vdd, vss=s.vss), pos=Vec2R((i*12), 8))
        s.out_cap = SchemInstance(Cap(c=R('1p')).symbol.portmap(p=stage_nets[-1], m=s.vss), pos=Vec2R((self.stages*12), 8))
        helpers.schem_check(s, add_terminal_taps=True, add_conn_points=True)
        return s

class Sky130Inverter(SimBase):
    @helpers.generate
    def schematic(self):
        from ordec.lib import sky130
        s = Schematic(cell=self)
        s.vin, s.vout, s.vdd, s.vss = Net(), Net(), Net(), Net()
        s.vdd_src = SchemInstance(Vdc(dc=R('1.8')).symbol.portmap(p=s.vdd, m=s.vss), pos=Vec2R(-20, 20))
        s.gnd = SchemInstance(Gnd().symbol.portmap(p=s.vss), pos=Vec2R(-20, -10))
        s.vin_src = SchemInstance(PulseVoltageSource(initial_value=R('0'), pulsed_value=R('1.8'), delay_time=R('1n'), rise_time=R('10p'), fall_time=R('10p'), pulse_width=R('5n'), period=R('10n')).symbol.portmap(p=s.vin, m=s.vss), pos=Vec2R(-20, 0))
        s.inv = SchemInstance(sky130.Inv().symbol.portmap(a=s.vin, y=s.vout, vdd=s.vdd, vss=s.vss), pos=Vec2R(10, 5))
        s.load_cap = SchemInstance(Cap(c=R('10f')).symbol.portmap(p=s.vout, m=s.vss), pos=Vec2R(25, 5))
        helpers.schem_check(s, add_terminal_taps=True, add_conn_points=True)
        return s

class IHP130Inverter(SimBase):
    @helpers.generate
    def schematic(self):
        from ordec.lib import ihp130
        s = Schematic(cell=self)
        s.vin, s.vout, s.vdd, s.vss = Net(), Net(), Net(), Net()
        s.vdd_src = SchemInstance(Vdc(dc=R('1.8')).symbol.portmap(p=s.vdd, m=s.vss), pos=Vec2R(-20, 20))
        s.gnd = SchemInstance(Gnd().symbol.portmap(p=s.vss), pos=Vec2R(-20, -10))
        s.vin_src = SchemInstance(PulseVoltageSource(initial_value=R('0'), pulsed_value=R('1.8'), delay_time=R('1n'), rise_time=R('10p'), fall_time=R('10p'), pulse_width=R('5n'), period=R('10n')).symbol.portmap(p=s.vin, m=s.vss), pos=Vec2R(-20, 0))
        s.inv = SchemInstance(ihp130.Inv().symbol.portmap(a=s.vin, y=s.vout, vdd=s.vdd, vss=s.vss), pos=Vec2R(10, 5))
        s.load_cap = SchemInstance(Cap(c=R('10f')).symbol.portmap(p=s.vout, m=s.vss), pos=Vec2R(25, 5))
        helpers.schem_check(s, add_terminal_taps=True, add_conn_points=True)
        return s

class SimpleRCFilter(SimBase):
    @helpers.generate
    def schematic(self):
        s = Schematic(cell=self)
        s.vin, s.vout, s.gnd = Net(), Net(), Net()
        s.vsrc = SchemInstance(SinusoidalVoltageSource(amplitude=R('1'), frequency=R('1M')).symbol.portmap(p=s.vin, m=s.gnd), pos=Vec2R(0, 10))
        s.r1 = SchemInstance(Res(r=R('1k')).symbol.portmap(p=s.vin, m=s.vout), pos=Vec2R(10, 10))
        s.c1 = SchemInstance(Cap(c=R('1n')).symbol.portmap(p=s.vout, m=s.gnd), pos=Vec2R(20, 10))
        s.gnd_inst = SchemInstance(Gnd().symbol.portmap(p=s.gnd), pos=Vec2R(20, 0))
        helpers.schem_check(s, add_terminal_taps=True, add_conn_points=True)
        return s

TESTBENCH_CLASSES = {
    "ResdivFlatTb": lib_test.ResdivFlatTb, "ResdivHierTb": lib_test.ResdivHierTb,
    "NmosSourceFollowerTb": lib_test.NmosSourceFollowerTb, "InvTb": lib_test.InvTb,
    "InvSkyTb": lib_test.InvSkyTb, "InvIhpTb": lib_test.InvIhpTb,
    "RcFilterTb": lib_test.RcFilterTb, "LargeRingOsc": LargeRingOsc,
    "Sky130Inverter": Sky130Inverter, "IHP130Inverter": IHP130Inverter,
    "SimpleRCFilter": SimpleRCFilter,
}

# --- Simulation Runner Functions ---
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

def run_tran_async(tb_class, backend, tstep, tstop):
    """
    DEFINITIVE ROBUST IMPLEMENTATION: This function bypasses the flawed library generator
    and uses a consumer loop that is immune to race conditions.
    """
    total_start_time = time.time()
    tb_instance = tb_class(backend=backend)

    node = SimHierarchy()
    highlevel_sim = HighlevelSim(tb_instance.schematic, node, backend=backend)

    sample_count = 0
    with Ngspice.launch(backend=backend) as sim:
        for hook in highlevel_sim.sim_setup_hooks:
            hook(sim)
        sim.load_netlist(highlevel_sim.netlister.out())

        data_queue = sim.tran_async(tstep, tstop)
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
                    if sample_count % 1000 == 0:
                        progress = data_point.get("progress", 0.0) * 100
                        sim_time = data_point.get("data", {}).get("time", 0)
                        logger.info(
                            f"[{tb_class.__name__}/{backend}] Progress: {progress:.2f}% | "
                            f"Real Time: {time.time() - sim_start_time:.2f}s | Sim Time: {sim_time:.3e}s"
                        )
            except queue.Empty:
                # If the queue is empty, the loop will re-check sim.is_running().
                # If the sim is also finished, the loop terminates. Otherwise, it continues to wait.
                pass

    time_taken = time.time() - total_start_time
    return time_taken, sample_count

# --- Multiprocessing and Benchmark Helpers ---
# We keep the multiprocessing wrapper for the simple, blocking calls (DC, AC, tran)
# as it is a reliable way to enforce timeouts for them. Async is now handled in-process.

def worker(task_func, result_queue, *args):
    try:
        result = task_func(*args)
        result_queue.put(result)
    except Exception as e:
        logger.error(f"Exception in worker process for {task_func.__name__}: {e}", exc_info=True)
        result_queue.put(f"Failed in Worker: {e}")

def run_with_timeout(task_func, *args):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(task_func, result_queue, *args))
    process.start()
    process.join(timeout=SIMULATION_TIMEOUT)
    if process.is_alive():
        logger.warning(f"Timeout occurred for {task_func.__name__}. Terminating process.")
        process.terminate()
        process.join()
        return "Timeout"
    if not result_queue.empty():
        return result_queue.get()
    return "Failed (No Result)"

class Benchmark:
    def __init__(self, name, tb_class):
        self.name, self.tb_class = name, tb_class
        self.tests = []
        self._add_dc_op_test()

    def _add_dc_op_test(self):
        self.tests.append({"name": "DC OP", "func": run_dc_op, "args": (self.tb_class,)})
        return self

    def add_transient_test(self, tstep, tstop, name_suffix=""):
        suffix = f" {name_suffix}" if name_suffix else ""
        sim_params = (tstep, tstop)
        self.tests.append({"name": f"Transient{suffix}", "func": run_tran, "args": (self.tb_class,) + sim_params})
        self.tests.append({"name": f"Async Transient{suffix}", "func": run_tran_async, "args": (self.tb_class,) + sim_params})
        return self

    def add_ac_test(self, ptype, n, start, stop):
        sim_params = (ptype, n, start, stop)
        self.tests.append({"name": "AC", "func": run_ac, "args": (self.tb_class,) + sim_params})
        return self

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation benchmarks for different backends.")
    parser.add_argument('--list-tests', action='store_true', help='List all available benchmark names and exit.')
    parser.add_argument('--match', nargs='+', metavar='PATTERN', help='Run only benchmarks whose names contain any of the given substrings (case-insensitive).')
    parser.add_argument('--backend', nargs='+', metavar='BACKEND', choices=ALL_BACKENDS, help=f'Run only for the specified backends. If not set, all backends will run. Choices: {ALL_BACKENDS}')
    args = parser.parse_args()

    all_benchmarks = [
        Benchmark("ResdivFlatTb", TESTBENCH_CLASSES["ResdivFlatTb"]).add_transient_test("0.1u", "5u"),
        Benchmark("ResdivHierTb", TESTBENCH_CLASSES["ResdivHierTb"]).add_transient_test("0.1u", "5u"),
        Benchmark("NmosSourceFollowerTb", TESTBENCH_CLASSES["NmosSourceFollowerTb"]).add_transient_test("0.1u", "2u"),
        Benchmark("InvTb", TESTBENCH_CLASSES["InvTb"]).add_transient_test("0.1u", "2u"),
        Benchmark("InvSkyTb", TESTBENCH_CLASSES["InvSkyTb"])
            .add_transient_test("0.01u", "1u")
            .add_transient_test("0.01u", "50u", name_suffix="(5k samples)"),
        Benchmark("InvIhpTb", TESTBENCH_CLASSES["InvIhpTb"]).add_transient_test("0.01u", "1u"),
        Benchmark("RcFilterTb", TESTBENCH_CLASSES["RcFilterTb"]).add_transient_test("10u", "1m").add_ac_test('dec', '10', '1', '1G'),
        Benchmark("LargeRingOsc", TESTBENCH_CLASSES["LargeRingOsc"]).add_transient_test("100p", "50n"),
        Benchmark("Sky130Inverter", TESTBENCH_CLASSES["Sky130Inverter"]).add_transient_test("100p", "30n"),
        Benchmark("IHP130Inverter", TESTBENCH_CLASSES["IHP130Inverter"]).add_transient_test("100p", "30n"),
        Benchmark("SimpleRCFilter", TESTBENCH_CLASSES["SimpleRCFilter"]).add_ac_test('dec', '20', '1', '1G'),
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
                logger.info(f"Starting test: '{test_name}'...")

                # We now use the reliable multiprocessing wrapper for ALL tests,
                # as the new async runner is robust enough to handle it.
                result_value = run_with_timeout(task_func, *task_args)

                results[backend][benchmark.name][test_name] = result_value

                if test_name.startswith('Async Transient') and isinstance(result_value, tuple):
                    logger.info(f"Finished test: '{test_name}'. Time: {result_value[0]} seconds, Samples: {result_value[1]}")
                else:
                    logger.info(f"Finished test: '{test_name}'. Time: {result_value} seconds")

    if results:
        print("\n" + "="*50 + "\n" + " " * 15 + "BENCHMARK SUMMARY" + "\n" + "="*50)
        for backend, tb_results in results.items():
            print(f"\nBackend: {backend}\n" + "-" * (len(backend) + 10))
            for tb_name, sim_results in tb_results.items():
                print(f"  Testbench: {tb_name}")
                for sim_type in sorted(sim_results.keys()):
                    result_value = sim_results[sim_type]
                    if sim_type.startswith("Async Transient") and isinstance(result_value, tuple):
                        time_taken, sample_count = result_value
                        print(f"    - {sim_type:<25}: {time_taken:.4f} seconds ({sample_count} samples)")
                    elif isinstance(result_value, float):
                        print(f"    - {sim_type:<25}: {result_value:.4f} seconds")
                    else:
                        print(f"    - {sim_type:<25}: {result_value}")
        print("\n" + "="*50)
