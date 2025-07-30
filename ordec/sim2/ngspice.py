# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import re
import signal
import sys
import tempfile
import warnings
from collections import namedtuple
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import Iterator, List, Optional

import numpy as np

from ..core import *

NgspiceVector = namedtuple('NgspiceVector', ['name', 'quantity', 'dtype', 'rest'])
NgspiceValue = namedtuple('NgspiceValue', ['type', 'name', 'subname', 'value'])

class NgspiceError(Exception):
    pass

class NgspiceFatalError(NgspiceError):
    pass


class NgSpiceConfigError(NgspiceError):
    """Raised when backend configuration fails."""
    pass

class NgSpiceBackend(Enum):
    """Available NgSpice backend types."""
    SUBPROCESS = "subprocess"
    FFI = "ffi"
    AUTO = "auto"


def get_default_backend() -> NgSpiceBackend:
    backend_str = os.environ.get('NGSPICE_BACKEND', 'auto').lower()
    try:
        return NgSpiceBackend(backend_str)
    except ValueError:
        warnings.warn(f"Invalid NGSPICE_BACKEND '{backend_str}', falling back to auto.")
        return NgSpiceBackend.AUTO


def _detect_ffi_availability() -> bool:
    try:
        _FFIBackend.find_library()
        return True
    except NgSpiceConfigError:
        return False


def _detect_subprocess_availability() -> bool:
    # We assume 'ngspice' is in the PATH. A more thorough check could be added.
    return True


def select_backend(preferred: NgSpiceBackend) -> NgSpiceBackend:
    """Selects the best available backend."""
    is_ffi_available = _detect_ffi_availability()
    is_subprocess_available = _detect_subprocess_availability()

    if preferred == NgSpiceBackend.AUTO:
        if is_subprocess_available:
            return NgSpiceBackend.SUBPROCESS
        if is_ffi_available:
            return NgSpiceBackend.FFI
        raise NgSpiceConfigError("No suitable NgSpice backend found. Please install NgSpice.")
    elif preferred == NgSpiceBackend.FFI:
        if is_ffi_available:
            return NgSpiceBackend.FFI
        raise NgSpiceConfigError("NgSpice FFI backend selected but the shared library could not be found.")
    elif preferred == NgSpiceBackend.SUBPROCESS:
        if is_subprocess_available:
            return NgSpiceBackend.SUBPROCESS
        raise NgSpiceConfigError("NgSpice subprocess backend selected but the executable is not available.")
    raise NgSpiceConfigError(f"Unknown backend '{preferred}' requested.")


# --- Data Structures ---

class NgspiceTable:
    def __init__(self, name):
        self.name = name
        self.headers = []
        self.data = []

class NgspiceTransientResult:
    def __init__(self):
        self.time = []
        self.signals = {}
        self.tables = []
        self.voltages = {}
        self.currents = {}
        self.branches = {}

    def add_table(self, table):
        """Add a table and extract signals into the signals dictionary."""
        self.tables.append(table)

        if not table.headers or not table.data:
            return

        # Find time column (usually index 1, but could be elsewhere)
        time_idx = None
        for i, header in enumerate(table.headers):
            if header.lower() == 'time':
                time_idx = i
                break

        if time_idx is None:
            return

        # Extract time data if we don't have it yet
        if not self.time and table.data:
            self.time = [float(row[time_idx]) for row in table.data if len(row) > time_idx]

        # Extract signal data
        for i, header in enumerate(table.headers):
            if header.lower() in ['index', 'time']:
                continue

            signal_name = header
            signal_data = []

            for row in table.data:
                if len(row) > i:
                    try:
                        signal_data.append(float(row[i]))
                    except (ValueError, IndexError):
                        signal_data.append(0.0)

            self.signals[signal_name] = signal_data

            # Categorize signals for easier access
            self._categorize_signal(signal_name, signal_data)

    def _categorize_signal(self, signal_name, signal_data):
        """Categorize signals into voltages, currents, and branches."""
        if signal_name.startswith('@') and '[' in signal_name:
            # Device current like "@m.xi0.mpd[id]"
            device_part = signal_name.split('[')[0][1:]  # Remove @ and get device part
            current_type = signal_name.split('[')[1].rstrip(']')  # Get current type
            if device_part not in self.currents:
                self.currents[device_part] = {}
            self.currents[device_part][current_type] = signal_data
        elif signal_name.endswith('#branch'):
            # Branch current like "vi3#branch"
            branch_name = signal_name.replace('#branch', '')
            self.branches[branch_name] = signal_data
        else:
            # Regular node voltage
            self.voltages[signal_name] = signal_data

    def __getitem__(self, key):
        """Allow table indexing or signal access."""
        if isinstance(key, int):
            return self.tables[key]
        else:
            return self.get_signal(key)

    def __len__(self):
        return len(self.tables)

    def __iter__(self):
        return iter(self.tables)

    def get_signal(self, signal_name):
        return self.signals.get(signal_name, [])

    def get_voltage(self, node_name):
        return self.voltages.get(node_name, [])

    def get_current(self, device_name, current_type='id'):
        device_currents = self.currents.get(device_name, {})
        return device_currents.get(current_type, [])

    def get_branch_current(self, branch_name):
        return self.branches.get(branch_name, [])

    def list_signals(self):
        return list(self.signals.keys())

    def list_voltages(self):
        return list(self.voltages.keys())

    def list_currents(self):
        return list(self.currents.keys())

    def list_branches(self):
        return list(self.branches.keys())

    def plot_signals(self, *signal_names):
        result = {'time': self.time}
        for name in signal_names:
            result[name] = self.get_signal(name)
        return result


# --- FFI Backend Implementation ---

class _FFIBackend:
    # --- C-compatible Data Types and Structures ---
    class NgComplex(ctypes.Structure):
        _fields_ = [("cx_real", ctypes.c_double), ("cx_imag", ctypes.c_double)]

    class VectorInfo(ctypes.Structure):
        pass
    PVectorInfo = ctypes.POINTER(VectorInfo)
    VectorInfo._fields_ = [
        ("v_name", ctypes.c_char_p), ("v_type", ctypes.c_int),
        ("v_flags", ctypes.c_short), ("v_realdata", ctypes.POINTER(ctypes.c_double)),
        ("v_compdata", ctypes.POINTER(NgComplex)), ("v_length", ctypes.c_int),
    ]

    def _send_char_handler(self, output, id):
        output_str = ctypes.string_at(output).decode('utf-8')
        if self.debug:
            print(f"[ngspice-ffi-out] {output_str}")
        if "Error: unknown subckt:" in output_str:
            raise NgspiceError(output_str.strip())
        self._output_lines.append(output_str)

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.lib = self.find_library()
        self._setup_library_functions()
        self._output_lines = []

        # Keep references to callbacks
        self._send_char_cb = self._SendChar(self._send_char_handler)
        self._send_stat_cb = self._SendStat(self._send_stat_handler)
        self._exit_cb = self._ControlledExit(self._exit_handler)

        init_result = self.lib.ngSpice_Init(self._send_char_cb, self._send_stat_cb, self._exit_cb, None, None, None, None)
        if init_result != 0:
            raise NgSpiceConfigError(f"Failed to initialize NgSpice FFI library (error code: {init_result}).")

    @staticmethod
    @contextmanager
    def launch(debug=False):
        backend = None
        try:
            backend = _FFIBackend(debug=debug)
            yield backend
        except NgspiceError as e:
            raise e
        finally:
            if backend:
                try:
                    backend.cleanup()
                except:
                    pass  # Ignore cleanup errors to prevent segfaults

    def cleanup(self):
        try:
            if hasattr(self, 'lib') and self.lib:
                self.lib.ngSpice_Command(b"quit")
        except:
            pass  # Ignore cleanup errors to prevent segfaults

    def _send_char_handler(self, message: bytes, ident: int, user_data) -> int:
        if message:
            msg_str = message.decode('utf-8', errors='ignore').strip()
            self._output_lines.append(msg_str)
            if self.debug:
                print(f"[ngspice-ffi-out] {msg_str}")
        return 0

    def _send_stat_handler(self, status: bytes, ident: int, user_data) -> int:
        if self.debug and status:
            print(f"[ngspice-ffi-stat] {status.decode('utf-8', errors='ignore').strip()}")
        return 0

    def _exit_handler(self, status: int, unload: bool, quit_upon_exit: bool, ident: int, user_data) -> int:
        if status != 0 and self.debug:
            print(f"[ngspice-ffi-exit] code {status}")
        return status

    def command(self, command: str) -> str:
        self._output_lines.clear()
        ret = self.lib.ngSpice_Command(command.encode('utf-8'))
        output = "\n".join(self._output_lines)
        check_errors(output)
        return output

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        # FFI backend loads circuit from an array of strings
        if no_auto_gnd:
            self.command("set no_auto_gnd")

        circuit_lines = [line.encode('utf-8') for line in netlist.split('\n') if line.strip()]
        c_circuit = (ctypes.c_char_p * len(circuit_lines))()
        c_circuit[:] = circuit_lines

        circ_result = self.lib.ngSpice_Circ(c_circuit)
        if circ_result != 0:
            # Check the output for a more descriptive error message
            output = "\n".join(self._output_lines)
            check_errors(output)
            raise NgspiceFatalError(f"Failed to load circuit into FFI backend. Full output:\n{output}")

    def op(self) -> Iterator[NgspiceValue]:
        self.command("op")
        all_vectors = self._get_all_vectors()

        for vec_name in all_vectors:
            vec_info = self._get_vector_info(vec_name)
            if not vec_info or vec_info.v_length == 0:
                continue

            value = vec_info.v_realdata[0]

            # Match naming conventions from subprocess backend
            if vec_name.startswith('@') and '[' in vec_name:
                match = re.match(r"@([a-zA-Z]\.)?([0-9a-zA-Z_.#]+)\[([0-9a-zA-Z_]+)\]", vec_name)
                if match:
                    yield NgspiceValue('current', match.group(2), match.group(3), value)
            elif vec_name.endswith('#branch'):
                yield NgspiceValue('current', vec_name.replace('#branch', ''), 'branch', value)
            else:
                yield NgspiceValue('voltage', vec_name, None, value)

    def tran(self, *args) -> NgspiceTransientResult:
        self.command(f"tran {' '.join(args)}")
        result = NgspiceTransientResult()
        table = NgspiceTable("transient_analysis")

        all_vectors = self._get_all_vectors()
        table.headers = all_vectors

        num_points = 0

        # Get all vector data and structure it by column
        vector_data_map = {}
        for vec_name in all_vectors:
            vec_info = self._get_vector_info(vec_name)
            if vec_info:
                num_points = max(num_points, vec_info.v_length)
                data_list = [vec_info.v_realdata[i] for i in range(vec_info.v_length)]
                vector_data_map[vec_name] = data_list

        # Transpose columns into rows
        for i in range(num_points):
            row = [vector_data_map.get(h, [None])[i] for h in table.headers]
            table.data.append(row)

        result.add_table(table)
        return result

    def _get_all_vectors(self) -> List[str]:
        plot_name = self.lib.ngSpice_CurPlot()
        if not plot_name:
            return []

        vecs_ptr = self.lib.ngSpice_AllVecs(plot_name)
        vectors = []
        i = 0
        while vecs_ptr and vecs_ptr[i]:
            vectors.append(vecs_ptr[i].decode('utf-8'))
            i += 1
        return vectors

    def _get_vector_info(self, vector_name: str) -> Optional[VectorInfo]:
        vec_info_ptr = self.lib.ngGet_Vec_Info(vector_name.encode('utf-8'))
        return vec_info_ptr.contents if vec_info_ptr else None

    @staticmethod
    def find_library() -> ctypes.CDLL:
        lib_names = {
            'win32': ['ngspice.dll', 'libngspice-0.dll'],
            'darwin': ['libngspice.dylib', 'libngspice.0.dylib'],
        }.get(sys.platform, ['libngspice.so', 'libngspice.so.0'])

        # Check standard system paths first
        for lib_name in lib_names:
            try:
                return ctypes.CDLL(lib_name)
            except OSError:
                continue

        raise NgSpiceConfigError(
            "Could not find the ngspice shared library. Please ensure ngspice is installed "
            "and the library is in the system's search path."
        )

    def _setup_library_functions(self):
        # Define callback function prototypes
        self._SendChar = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        self._SendStat = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        self._ControlledExit = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_void_p)

        # Core functions
        self.lib.ngSpice_Init.restype = ctypes.c_int
        self.lib.ngSpice_Init.argtypes = [self._SendChar, self._SendStat, self._ControlledExit, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.lib.ngSpice_Command.restype = ctypes.c_int
        self.lib.ngSpice_Command.argtypes = [ctypes.c_char_p]

        self.lib.ngSpice_Circ.restype = ctypes.c_int
        self.lib.ngSpice_Circ.argtypes = [ctypes.POINTER(ctypes.c_char_p)]

        self.lib.ngGet_Vec_Info.restype = self.PVectorInfo
        self.lib.ngGet_Vec_Info.argtypes = [ctypes.c_char_p]

        self.lib.ngSpice_CurPlot.restype = ctypes.c_char_p
        self.lib.ngSpice_AllVecs.restype = ctypes.POINTER(ctypes.c_char_p)
        self.lib.ngSpice_AllVecs.argtypes = [ctypes.c_char_p]


# --- Subprocess Backend Implementation ---

class _SubprocessBackend:
    @staticmethod
    @contextmanager
    def launch(debug=False):
        with tempfile.TemporaryDirectory() as cwd_str:
            p = Popen(['ngspice', '-p'], stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=cwd_str)
            try:
                yield _SubprocessBackend(p, debug=debug, cwd=Path(cwd_str))
            finally:
                try:
                    p.send_signal(signal.SIGTERM)
                    if p.stdin:
                        p.stdin.close()
                    if p.stdout:
                        p.stdout.read()
                    p.wait(timeout=1.0)
                except (ProcessLookupError, BrokenPipeError, TimeoutError):
                    pass  # Process may have already terminated

    def __init__(self, p: Popen, debug: bool, cwd: Path):
        self.p = p
        self.debug = debug
        self.cwd = cwd

    def command(self, command: str) -> str:
        """Executes ngspice command and returns string output from ngspice process."""
        if self.p.poll() is not None:
            raise NgspiceFatalError("ngspice process has terminated unexpectedly.")
        if self.debug:
            print(f"[debug] sending command to ngspice ({self.p.pid}): {command}")

        if self.p.stdin:
            self.p.stdin.write(f"{command}; echo FINISHED\n".encode("ascii"))
            self.p.stdin.flush()

        out = []
        while True:
            l=self.p.stdout.readline()
            #print(f"[debug] received line from ngspice: {l}")

            # Ignore echo in case of ngspice build with libreadline:
            if re.match(rb"ngspice [0-9]+ -> .*; echo FINISHED\n", l):
                continue

            # Strip "ngspice 123 -> " from line in case of ngspice build with neither libreadline nor libedit:
            m = re.match(rb"ngspice [0-9]+ -> (.*\n)", l)
            if m:
                l = m.group(1)

            if l == b'FINISHED\n':
                break
            elif l == b'': # readline() returns the empty byte string only on EOF.
                out_flat = "".join(out)
                raise NgspiceFatalError(f"ngspice terminated abnormally:\n{out_flat}")
            out.append(l.decode('ascii'))
        out_flat = "".join(out)
        if self.debug:
            print(f"[debug] received result from ngspice ({self.p.pid}): {repr(out_flat)}")

        check_errors(out_flat)
        return out_flat

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        netlist_fn = self.cwd / 'netlist.sp'
        netlist_fn.write_text(netlist)
        if self.debug:
            print(f"Written netlist: \n {netlist}")
        if no_auto_gnd:
            self.command("set no_auto_gnd")
        check_errors(self.command(f"source {netlist_fn}"))

    def print_all(self) -> Iterator[str]:
        """
        Tries "print all" first. If it fails due to zero-length vectors, emulate
        "print all" using display and print but skip zero-length vectors.
        """

        print_all_res = self.command("print all")
        # Check if the result contains the warning about zero-length vectors
        if "is not available or has zero length" in print_all_res:
            # Fallback: get list of available vectors and print only valid ones
            display_output = self.command("display")

            # Parse vector list and print only vectors with length > 0
            for line in display_output.split('\n'):
                # Look for vector definitions like "name: type, real, N long"
                vector_match = re.match(r'\s*([^:]+):\s*[^,]+,\s*[^,]+,\s*([0-9]+)\s+long', line)
                if vector_match:
                    vector_name = vector_match.group(1).strip()
                    vector_length = int(vector_match.group(2))

                    # Only print vectors that have data (length > 0)
                    if vector_length > 0:
                        yield self.command(f"print {vector_name}")
        else:
            yield from print_all_res.split('\n')

    def op(self) -> Iterator[NgspiceValue]:
        self.command("op")

        for line in self.print_all():
            if len(line) == 0:
                continue

            # Voltage result - updated regex to handle device names with special chars:
            res = re.match(r"([0-9a-zA-Z_.#]+)\s*=\s*([0-9.\-+e]+)\s*", line)
            if res:
                yield NgspiceValue(type='voltage', name=res.group(1), subname=None, value=float(res.group(2)))

            # Current result like "vgnd#branch":
            res = re.match(r"([0-9a-zA-Z_.#]+)#branch\s*=\s*([0-9.\-+e]+)\s*", line)
            if res:
                yield NgspiceValue(type='current', name=res.group(1), subname='branch', value=float(res.group(2)))

            # Current result like "@m.xdut.mm2[is]" from savecurrents:
            res = re.match(r"@([a-zA-Z]\.)?([0-9a-zA-Z_.#]+)\[([0-9a-zA-Z_]+)\]\s*=\s*([0-9.\-+e]+)\s*", line)
            if res:
                yield NgspiceValue(type='current', name=res.group(2), subname=res.group(3), value=float(res.group(4)))

    def tran(self, *args) -> NgspiceTransientResult:
        self.command(f"tran {' '.join(args)}")
        print_all_res = self.command("print all")
        lines = print_all_res.split('\n')

        result = NgspiceTransientResult()
        i = 0

        while i < len(lines):
            line = lines[i]

            if len(line) == 0:
                i += 1
                continue

            # Look for NGspice table pattern: (only tested with a single version of ngspice!)

            # Check if this could be start of a table
            if (not re.match(r"^-+$", line.strip()) and  # Not a separator line
                line.strip() and  # Not empty
                i + 4 < len(lines)):  # Enough lines ahead

                # Look for the pattern ahead
                desc_offset = 1

                # Check if next line is description or separator
                if re.match(r"^-+$", lines[i + 1].strip()):
                    # Next line is separator, no description
                    desc_offset = 0
                elif (i + 2 < len(lines) and
                      re.match(r"^-+$", lines[i + 2].strip())):
                    # Line after next is separator, so next line is description
                    desc_offset = 1
                else:
                    # Not a table pattern
                    i += 1
                    continue

                separator1_idx = i + 1 + desc_offset
                headers_idx = separator1_idx + 1
                separator2_idx = headers_idx + 1

                if (separator2_idx < len(lines) and
                    re.match(r"^-+$", lines[separator1_idx].strip()) and
                    re.match(r"^-+$", lines[separator2_idx].strip())):

                    table = NgspiceTable(line.strip())
                    table.headers = lines[headers_idx].split()

                    # Skip to data section
                    i = separator2_idx + 1

                    # Read data rows
                    while i < len(lines):
                        data_line = lines[i]
                        i += 1

                        # Check for table end
                        if (re.match(r"^-+$", data_line.strip()) or
                            data_line == '\x0c' or
                            not data_line.strip()):
                            break

                        # Add data row
                        table.data.append(data_line.split())

                    result.add_table(table)
                    continue

            # Not a table, move to next line
            i += 1

        return result


def check_errors(ngspice_out):
    """Helper function to raise NgspiceError in Python from "Error: ..."
    messages in Ngspice's output."""
    for line in ngspice_out.split('\n'):
        m = re.match(r"Error:\s*(.*)", line)
        if m:
            raise NgspiceError(m.group(1))


class Ngspice:
    @staticmethod
    @contextmanager
    def launch(debug=False, backend=None):
        """
        Launch NgSpice using the best available or specified backend.

        Args:
            debug: Enable detailed logging from the backend.
            backend: "ffi", "subprocess", or "auto". If None, defaults to "auto".
        """
        if backend is None:
            backend_type = get_default_backend()
        elif isinstance(backend, str):
            backend_type = NgSpiceBackend(backend.lower())
        else:
            backend_type = backend

        chosen_backend = select_backend(backend_type)

        if debug:
            print(f"[Ngspice] Using backend: {chosen_backend.value}")

        backend_class = {
            NgSpiceBackend.FFI: _FFIBackend,
            NgSpiceBackend.SUBPROCESS: _SubprocessBackend,
        }[chosen_backend]

        with backend_class.launch(debug=debug) as backend_instance:
            yield Ngspice(backend_instance, debug=debug)

    def __init__(self, backend_impl, debug: bool = False):
        self._backend_impl = backend_impl
        self.debug = debug

    def command(self, command: str) -> str:
        """Executes ngspice command and returns string output from ngspice process."""
        return self._backend_impl.command(command)

    def vector_info(self) -> Iterator[NgspiceVector]:
        """Wrapper for ngspice's "display" command."""
        for line in self.command("display").split("\n\n")[2].split('\n'):
            if len(line) == 0:
                continue
            res = re.match(r"\s*([0-9a-zA-Z_.#]*)\s*:\s*([a-zA-Z]+),\s*([a-zA-Z]+),\s*(.*)", line)
            if res:
                yield NgspiceVector(*res.groups())

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        return self._backend_impl.load_netlist(netlist, no_auto_gnd=no_auto_gnd)



    def op(self) -> Iterator[NgspiceValue]:
        return self._backend_impl.op()

    def tran(self, *args) -> NgspiceTransientResult:
        return self._backend_impl.tran(*args)


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
        #print(info)
        #print(info_vars)
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
    def __init__(self):
        self.obj_of_name = {}
        self.name_of_obj = {}
        self.spice_cards = []
        self.cur_line = 0
        self.indent = 0
        self.setup_funcs = set()

    def require_setup(self, setup_func):
        self.setup_funcs.add(setup_func)

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
        #self.add('.probe', 'alli') # This seems to be needed to see currents of subcircuits
        self.add('.option', 'savecurrents') # This seems to be needed to see currents of devices (R, M)

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

        # Add model setup lines at top:
        self.cur_line = 1
        for setup_func in self.setup_funcs:
            setup_func(self)
