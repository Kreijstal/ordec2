# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import re
from collections import namedtuple
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Optional, Callable, Generator

import numpy as np

from ..core import *
from .ngspice_ffi import _FFIBackend
from .ngspice_subprocess import _SubprocessBackend
from .ngspice_mp import IsolatedFFIBackend

class NgspiceBackend(Enum):
    """Available NgSpice backend types."""
    SUBPROCESS = "subprocess"
    FFI = "ffi"
    MP = "mp"

class Ngspice:
    @staticmethod
    @contextmanager
    def launch(debug=False, backend : NgspiceBackend = NgspiceBackend.SUBPROCESS):
        if isinstance(backend, str):
            backend = NgspiceBackend(backend.lower())

        if debug:
            print(f"[Ngspice] Using backend: {backend.value}")

        backend_class = {
            NgspiceBackend.FFI: _FFIBackend,
            NgspiceBackend.SUBPROCESS: _SubprocessBackend,
            NgspiceBackend.MP: IsolatedFFIBackend,
        }[backend]

        with backend_class.launch(debug=debug) as backend_instance:
            yield Ngspice(backend_instance, debug=debug)

    def __init__(self, backend_impl, debug: bool = False):
        self._backend_impl = backend_impl
        self.debug = debug

    def command(self, command: str) -> str:
        """Executes ngspice command and returns string output from ngspice process."""
        return self._backend_impl.command(command)

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        return self._backend_impl.load_netlist(netlist, no_auto_gnd=no_auto_gnd)

    def op(self) -> Iterator['NgspiceValue']:
        return self._backend_impl.op()

    def tran(self, *args) -> 'NgspiceTransientResult':
        return self._backend_impl.tran(*args)

    def ac(self, *args, **kwargs) -> 'NgspiceAcResult':
        return self._backend_impl.ac(*args, **kwargs)

    def tran_async(self, *args, throttle_interval: float = 0.1) -> 'queue.Queue':
        """
        Start asynchronous transient analysis and return data queue.

        This replaces the problematic generator-based approach with a queue-based system
        that allows for better control over halt/alter operations.

        Args:
            *args: tran arguments (tstep, tstop, etc.)
            throttle_interval: Minimum time between data updates

        Returns:
            queue.Queue object containing simulation data points

        Raises:
            NotImplementedError: If backend doesn't support async
        """
        if hasattr(self._backend_impl, 'tran_async'):
            return self._backend_impl.tran_async(*args, throttle_interval=throttle_interval)
        else:
            raise NotImplementedError("Async transient analysis is only available with FFI or MP backends")

    def op_async(self, callback: Optional[Callable] = None) -> Generator:
        if hasattr(self._backend_impl, 'op_async'):
            yield from self._backend_impl.op_async(callback=callback)
        else:
            raise NotImplementedError("Async operating point analysis is only available with FFI backend")

    def is_running(self) -> bool:
        if hasattr(self._backend_impl, 'is_running'):
            return self._backend_impl.is_running()
        return False

    def stop_simulation(self):
        """Stop/halt running background simulation"""
        if hasattr(self._backend_impl, 'stop_simulation'):
            self._backend_impl.stop_simulation()

    def safe_halt_simulation(self, max_attempts: int = 3, wait_time: float = 0.2) -> bool:
        """
        Safely halt simulation with retries and verification.

        This method addresses the critical timing issues with ngspice background simulations:
        - bg_halt is not instantaneous and can fail silently
        - Multiple attempts may be needed
        - Must verify halt succeeded before proceeding with alter commands

        Args:
            max_attempts: Maximum number of halt attempts
            wait_time: Time to wait between attempts and for verification

        Returns:
            True if halt succeeded, False otherwise
        """
        import time

        if not self.is_running():
            return True

        for attempt in range(max_attempts):
            try:
                # Send halt command
                if hasattr(self._backend_impl, 'command'):
                    self._backend_impl.command("bg_halt")
                else:
                    self.stop_simulation()

                # CRITICAL: Must yield after halt command before checking state
                # Use race condition approach instead of fixed sleep
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

                    def check_halt_status():
                        """Check if simulation has halted"""
                        return not self.is_running()

                    timeout = time.time() + wait_time
                    while time.time() < timeout:
                        halt_future = executor.submit(check_halt_status)
                        try:
                            if halt_future.result(timeout=min(0.01, wait_time)):
                                return True
                        except concurrent.futures.TimeoutError:
                            pass
                        finally:
                            if not halt_future.done():
                                halt_future.cancel()

            except Exception:
                pass

            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    def wait_before_retry():
                        return True

                    wait_future = executor.submit(wait_before_retry)
                    try:
                        wait_future.result(timeout=wait_time)
                    except concurrent.futures.TimeoutError:
                        pass
                    finally:
                        if not wait_future.done():
                            wait_future.cancel()

        # Fallback attempt with stop_simulation
        try:
            self.stop_simulation()
            # Use race condition approach for final check
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

                def check_final_halt():
                    """Final check if simulation has stopped"""
                    return not self.is_running()

                timeout = time.time() + wait_time
                while time.time() < timeout:
                    final_future = executor.submit(check_final_halt)
                    try:
                        return final_future.result(timeout=min(0.01, wait_time))
                    except concurrent.futures.TimeoutError:
                        pass
                    finally:
                        if not final_future.done():
                            final_future.cancel()

            return not self.is_running()
        except:
            return False

    def safe_resume_simulation(self, max_attempts: int = 3, wait_time: float = 0.2) -> bool:
        """
        Safely resume a halted simulation with retries and verification.

        This method addresses the critical timing issues with ngspice background simulations:
        - bg_resume is not instantaneous and can fail silently
        - Multiple attempts may be needed
        - Must verify resume succeeded before proceeding

        Args:
            max_attempts: Maximum number of resume attempts
            wait_time: Time to wait between attempts and for verification

        Returns:
            True if resume succeeded, False otherwise
        """
        import time

        if self.is_running():
            return True

        for attempt in range(max_attempts):
            try:
                # Send resume command
                if hasattr(self._backend_impl, 'command'):
                    self._backend_impl.command("bg_resume")
                elif hasattr(self._backend_impl, 'resume_simulation'):
                    self._backend_impl.resume_simulation()

                # CRITICAL: Must yield after resume command before checking state
                # Use race condition approach instead of fixed sleep
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

                    def check_resume_status():
                        """Check if simulation has resumed"""
                        return self.is_running()

                    timeout = time.time() + wait_time
                    while time.time() < timeout:
                        resume_future = executor.submit(check_resume_status)
                        try:
                            if resume_future.result(timeout=min(0.01, wait_time)):
                                return True
                        except concurrent.futures.TimeoutError:
                            pass
                        finally:
                            if not resume_future.done():
                                resume_future.cancel()

            except Exception:
                pass

            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    def wait_before_retry():
                        return True

                    wait_future = executor.submit(wait_before_retry)
                    try:
                        wait_future.result(timeout=wait_time)
                    except concurrent.futures.TimeoutError:
                        pass
                    finally:
                        if not wait_future.done():
                            wait_future.cancel()

        return False

RawVariable = namedtuple('RawVariable', ['name', 'unit'])

def parse_raw(fn):
    info = {}
    info_vars = []

    with open(fn, 'rb') as f:
        for i in range(100):
            l = f.readline()[:-1].decode('ascii')

            if l.startswith('\t'):
                _, var_idx, var_name, var_unit = l.split('\t')
                assert int(var_idx) == len(info_vars)
                info_vars.append(RawVariable(var_name, var_unit))
            else:
                lhs, rhs = l.split(':', 1)
                info[lhs] = rhs.strip()
                if lhs == "Binary":
                    break
        assert len(info_vars) == int(info['No. Variables'])
        no_points = int(info['No. Points'])

        dtype = np.dtype({
            "names": [v.name for v in info_vars],
            "formats": [np.float64]*len(info_vars)
            })

        np.set_printoptions(precision=5)

        data=np.fromfile(f, dtype=dtype, count=no_points)
    return data


def basename_escape(obj):
    if isinstance(obj, Cell):
        basename = f"{type(obj).__name__}_{'_'.join(obj.params_list())}"
    else:
        basename = "_".join(obj.full_path_list())
    return re.sub(r'[^a-zA-Z0-9]', '_', basename).lower()

class Netlister:
    def __init__(self, enable_savecurrents: bool = True):
        self.obj_of_name = {}
        self.name_of_obj = {}
        self.spice_cards = []
        self.cur_line = 0
        self.indent = 0
        self.setup_funcs = set()
        self.enable_savecurrents = enable_savecurrents
        self._sim_setup_hooks = []

    def require_setup(self, setup_func):
        self.setup_funcs.add(setup_func)

    def require_sim_setup(self, sim_setup_func):
        """Register a function to be called during simulation setup.

        The function should accept a single argument: the Ngspice instance.
        This is useful for PDK-specific setup commands that need to be
        executed on the simulator instance rather than in the netlist.
        """
        self._sim_setup_hooks.append(sim_setup_func)

    def out(self):
        return "\n".join(self.spice_cards)+"\n.end\n"

    def add(self, *args):
        args_flat = []
        for arg in args:
            if isinstance(arg, list):
                args_flat += arg
            else:
                args_flat.append(arg)
        self.spice_cards.insert(self.cur_line, " "*self.indent + " ".join(args_flat))
        self.cur_line += 1

    def name_obj(self, obj, domain=None, prefix=""):
        try:
            return self.name_of_obj[obj]
        except KeyError:
            basename = prefix + basename_escape(obj)
            name = basename
            suffix = 0
            while (domain, name) in self.obj_of_name:
                name = f"{basename}{suffix}"
                suffix += 1
            self.obj_of_name[domain, name] = obj
            self.name_of_obj[obj] = name
            return name

    def name_hier_simobj(self, sn):
        c = sn
        if not isinstance(c, SimInstance):
            ret = [self.name_obj(sn.eref, sn.eref.subgraph)]
        else:
            ret = []

        while not isinstance(c, SimHierarchy):
            if isinstance(c, SimInstance):
                ret.insert(0, self.name_obj(c.eref, c.eref.subgraph))
            c = c.parent
        return ".".join(ret)

    def pinlist(self, sym: Symbol):
        return list(sym.all(Pin))

    def portmap(self, inst, pins):
        ret = []
        for pin in pins:
            conn = inst.subgraph.one(SchemInstanceConn.ref_pin_idx.query((inst.nid, pin.nid)))
            ret.append(self.name_of_obj[conn.here])
        return ret

    def netlist_schematic(self, s: Schematic):
        for net in s.all(Net):
            self.name_obj(net, s)

        subckt_dep = set()
        for inst in s.all(SchemInstance):
            try:
                f = inst.symbol.cell.netlist_ngspice
            except AttributeError: # subckt
                pins = self.pinlist(inst.symbol)
                subckt_dep.add(inst.symbol)
                self.add(self.name_obj(inst, s, prefix="x"), self.portmap(inst, pins), self.name_obj(inst.symbol.cell))
            else:
                f(self, inst, s)
        return subckt_dep

    def netlist_hier(self, top: Schematic):
        self.add('.title', self.name_obj(top.cell))
        if self.enable_savecurrents:
            self.add('.option', 'savecurrents')

        subckt_dep = self.netlist_schematic(top)
        subckt_done = set()
        while len(subckt_dep - subckt_done) > 0:
            symbol = next(iter(subckt_dep - subckt_done))
            schematic = symbol.cell.schematic
            self.add('.subckt', self.name_obj(symbol.cell), [self.name_obj(pin, symbol) for pin in self.pinlist(symbol)])
            self.indent += 4
            subckt_dep |= self.netlist_schematic(schematic)
            self.indent -= 4
            self.add(".ends", self.name_obj(symbol.cell))
            subckt_done.add(symbol)

        self.cur_line = 1
        for setup_func in self.setup_funcs:
            setup_func(self)
