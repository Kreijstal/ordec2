# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import re
import signal
import sys
import tempfile
import shutil
import queue
import threading
import time
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import Iterator, Optional

import numpy as np
from ..core.rational import Rational as R

from .ngspice_common import (
    NgspiceValue,
    NgspiceError,
    NgspiceFatalError,
    NgspiceTransientResult,
    NgspiceAcResult,
    NgspiceResultBase,
    check_errors,
    NgspiceTable,
    SignalKind,
    SignalArray,
    NgspiceBase,
)

NgspiceVector = namedtuple(
    "NgspiceVector", ["name", "quantity", "dtype", "length", "rest"]
)


class NgspiceSubprocess(NgspiceBase):
    # Class-level setting for restart threshold
    # Restart ngspice after this many simulations to prevent state accumulation
    # Set conservatively to 5 to ensure restart happens before issues occur
    # This is especially important for tests that run many simulations with batching
    RESTART_AFTER_N_SIMULATIONS = 2
    
    @classmethod
    @contextmanager
    def launch(cls, debug: bool):
        # Choose the correct ngspice executable for the platform
        if sys.platform == "win32":
            # On Windows, prefer ngspice_con if available, fall back to ngspice
            ngspice_exe = "ngspice_con" if shutil.which("ngspice_con") else "ngspice"
        else:
            ngspice_exe = "ngspice"

        if debug:
            print(f"[debug] Using ngspice executable: {ngspice_exe}")
            print(f"[debug] Platform: {sys.platform}")

        with tempfile.TemporaryDirectory() as cwd_str:
            if debug:
                print(f"[debug] Starting ngspice with command: {[ngspice_exe, '-p']}")
                print(f"[debug] Working directory: {cwd_str}")

            p: Popen[bytes] = Popen(
                [ngspice_exe, "-p"], stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd=cwd_str
            )
            if debug:
                print(f"[debug] Process started with PID: {p.pid}")

            try:
                instance = cls(p, debug=debug, cwd=Path(cwd_str), ngspice_exe=ngspice_exe)
                yield instance
            finally:
                if debug:
                    print(f"[debug] Cleaning up process {p.pid}")
                try:
                    p.send_signal(signal.SIGTERM)
                    if p.stdin:
                        p.stdin.close()
                    if p.stdout:
                        p.stdout.read()
                    p.wait(timeout=1.0)
                except (ProcessLookupError, BrokenPipeError, TimeoutError):
                    pass  # Process may have already terminated

    def __init__(self, p: Popen, debug: bool, cwd: Path, ngspice_exe: str = "ngspice"):
        self.p: Popen[bytes] = p
        self.debug = debug
        self.cwd = cwd
        self.ngspice_exe = ngspice_exe
        self._async_queue: Optional[queue.Queue] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_lock = threading.Lock()
        self._async_halt_requested = False
        self._async_resume_event = threading.Event()
        self._async_current_time = 0.0
        self._data_points_sent = 0
        self._last_vector_length = 0
        self._is_running = False
        self._print_commands_count = 0  # Track number of print commands executed
        self._simulation_count = 0  # Track number of simulations run
        self._netlist_content = None  # Cache netlist for restart
        self._no_auto_gnd = True  # Cache netlist settings

    def command(self, command: str) -> str:
        """Executes ngspice command and returns string output from ngspice process."""
        if self.p.poll() is not None:
            raise NgspiceFatalError("ngspice process has terminated unexpectedly.")
        if self.debug:
            print(f"[debug] sending command to ngspice ({self.p.pid}): {command}")

        if self.p.stdin:
            # Send the command followed by echo marker on separate lines
            full_input = f"{command}\necho FINISHED\n"
            if self.debug:
                print(f"[debug] Writing to stdin: {repr(full_input)}")
            self.p.stdin.write(full_input.encode("ascii"))
            self.p.stdin.flush()
            if self.debug:
                print(f"[debug] Stdin flushed")

        out = []
        line_count = 0
        while True:
            if self.debug:
                print(f"[debug] Waiting for line {line_count}...")
            l = self.p.stdout.readline()
            line_count += 1
            if self.debug:
                print(f"[debug] received line {line_count} from ngspice: {repr(l)}")

            # Check for EOF first
            if l == b"":  # readline() returns the empty byte string only on EOF.
                out_flat = "".join(out)
                if self.debug:
                    print(f"[debug] EOF detected, ngspice terminated")
                raise NgspiceFatalError(f"ngspice terminated abnormally:\n{out_flat}")

            # Strip ALL occurrences of "ngspice 123 -> " from the line on all platforms
            # Preserve newlines when stripping prompts
            while True:
                m = re.match(rb"ngspice [0-9]+ -> (.*)", l)
                if not m:
                    break
                if self.debug:
                    print(
                        f"[debug] Stripping prompt from line: {repr(l)} -> {repr(m.group(1))}"
                    )
                stripped_content = m.group(1)
                # Preserve the newline if the original line had one
                if l.endswith(b"\n") and not stripped_content.endswith(b"\n"):
                    l = stripped_content + b"\n"
                else:
                    l = stripped_content

            # Check for our finish marker
            if l.rstrip() == b"FINISHED":
                if self.debug:
                    print(f"[debug] Found FINISHED marker, breaking")
                break

            # Skip empty lines that are just prompts
            if l.strip() == b"":
                continue

            out.append(l.decode("ascii"))
            if self.debug:
                print(f"[debug] Added to output: {repr(l.decode('ascii'))}")

        out_flat = "".join(out)
        if self.debug:
            print(
                f"[debug] received result from ngspice ({self.p.pid}): {repr(out_flat)}"
            )

        check_errors(out_flat)
        return out_flat

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        # Cache netlist for potential restart
        self._netlist_content = netlist
        self._no_auto_gnd = no_auto_gnd
        
        netlist_fn = self.cwd / "netlist.sp"
        netlist_fn.write_text(netlist)
        if self.debug:
            print(f"Written netlist: \n {netlist}")
        if no_auto_gnd:
            self.command("set no_auto_gnd")
        # Set output width to avoid header wrapping in vector slicing output
        self.command("set width 200")
        check_errors(self.command(f"source {netlist_fn}"))

    def _restart_ngspice_process(self):
        """Restart the ngspice subprocess to clear accumulated state."""
        if self.debug:
            print(f"[debug] Restarting ngspice process (old PID: {self.p.pid})")
        
        # Terminate the old process
        try:
            self.p.send_signal(signal.SIGTERM)
            if self.p.stdin:
                self.p.stdin.close()
            if self.p.stdout:
                # Drain stdout to avoid blocking
                try:
                    self.p.stdout.read()
                except:
                    pass
            self.p.wait(timeout=1.0)
        except (ProcessLookupError, BrokenPipeError, TimeoutError):
            # Process may have already terminated
            pass
        
        # Start a new process
        new_p: Popen[bytes] = Popen(
            [self.ngspice_exe, "-p"], 
            stdin=PIPE, 
            stdout=PIPE, 
            stderr=STDOUT, 
            cwd=str(self.cwd)
        )
        
        if self.debug:
            print(f"[debug] New ngspice process started with PID: {new_p.pid}")
        
        # Update the process handle
        self.p = new_p
        
        # Reset counters
        self._print_commands_count = 0
        self._simulation_count = 0
        
        # Reload the netlist if we have one cached
        if self._netlist_content:
            if self.debug:
                print(f"[debug] Reloading netlist after restart")
            self.load_netlist(self._netlist_content, self._no_auto_gnd)

    def print_all(self) -> Iterator[str]:
        """
        Tries "print all" first. If it fails due to zero-length vectors, emulate
        "print all" using display and print but skip zero-length vectors.
        """

        print_all_res = self.command("print all")
        # Check if the result contains the warning about zero-length vectors
        if "is not available or has zero length" in print_all_res:
            # get list of available vectors and print only valid ones
            display_output = self.command("display")

            # Parse vector list and print only vectors with length > 0
            for line in display_output.split("\n"):
                # Look for vector definitions like "name: type, real, N long"
                vector_match = re.match(
                    r"\s*([^:]+):\s*[^,]+,\s*[^,]+,\s*([0-9]+)\s+long", line
                )
                if vector_match:
                    vector_name = vector_match.group(1).strip()
                    vector_length = int(vector_match.group(2))

                    # Only print vectors that have data (length > 0)
                    if vector_length > 0:
                        yield self.command(f"print {vector_name}")
        else:
            yield from print_all_res.split("\n")

    def _parse_op_results(self) -> Iterator[str]:
        """
        Parse operating point results, extracting only the result lines
        from command output and skipping command echoes and FINISHED markers.
        """
        print_all_res = self.command("print all")
        # Check if the result contains the warning about zero-length vectors
        if "is not available or has zero length" in print_all_res:
            # Fallback: get list of available vectors and print only valid ones
            display_output = self.command("display")

            # Parse vector list and print only vectors with length > 0
            for line in display_output.split("\n"):
                # Look for vector definitions like "name: type, real, N long"
                vector_match = re.match(
                    r"\s*([^:]+):\s*[^,]+,\s*[^,]+,\s*([0-9]+)\s+long", line
                )
                if vector_match:
                    vector_name = vector_match.group(1).strip()
                    vector_length = int(vector_match.group(2))

                    # Only print vectors that have data (length > 0)
                    if vector_length > 0:
                        cmd_output = self.command(f"print {vector_name}")
                        # Extract just the result lines from command output
                        for output_line in cmd_output.split("\n"):
                            if re.match(
                                r"([0-9a-zA-Z_.#]+)\s*=\s*([0-9.\-+e]+)\s*", output_line
                            ):
                                yield output_line
        else:
            # Extract just the result lines from the print all output
            for line in print_all_res.split("\n"):
                if re.match(r"([0-9a-zA-Z_.#]+)\s*=\s*([0-9.\-+e]+)\s*", line):
                    yield line

    def op(self) -> Iterator[NgspiceValue]:
        self.command("op")

        for line in self._parse_op_results():
            if len(line) == 0:
                continue

            # Voltage result - updated regex to handle device names with special chars:
            res = re.match(r"([0-9a-zA-Z_.#]+)\s*=\s*([0-9.\-+e]+)\s*", line)
            if res:
                yield NgspiceValue(
                    type="voltage",
                    name=res.group(1),
                    subname=None,
                    value=float(res.group(2)),
                )

            # Current result like "vgnd#branch":
            res = re.match(r"([0-9a-zA-Z_.#]+)#branch\s*=\s*([0-9.\-+e]+)\s*", line)
            if res:
                yield NgspiceValue(
                    type="current",
                    name=res.group(1),
                    subname="branch",
                    value=float(res.group(2)),
                )

            # Current result like "@m.xdut.mm2[is]" from savecurrents:
            res = re.match(
                r"@([a-zA-Z]\.)?([0-9a-zA-Z_.#]+)\[([0-9a-zA-Z_]+)\]\s*=\s*([0-9.\-+e]+)\s*",
                line,
            )
            if res:
                yield NgspiceValue(
                    type="current",
                    name=res.group(2),
                    subname=res.group(3),
                    value=float(res.group(4)),
                )

    def tran(self, *args) -> NgspiceTransientResult:
        self.command(f"tran {' '.join(args)}")
        print_all_res = "\n".join(self.print_all())
        lines = print_all_res.split("\n")

        result = NgspiceTransientResult()
        tables = {}  # map from header tuple to list of data rows
        current_headers = None

        for line in lines:
            line = line.strip()
            if not line or re.match(r"^-+$", line) or "Transient Analysis" in line:
                continue

            potential_headers = line.split()
            is_header = any(
                h.lower() in ("time", "index") for h in potential_headers
            ) and not self._is_numeric_row(potential_headers)

            if is_header:
                current_headers = tuple(potential_headers)
                if current_headers not in tables:
                    tables[current_headers] = []
            elif current_headers:
                row_data = line.split()
                if self._is_numeric_row(row_data):
                    # Ensure data row has a compatible number of columns, pad if necessary
                    if len(row_data) <= len(current_headers):
                        tables[current_headers].append(row_data)

        for headers, data in tables.items():
            if data:
                table = NgspiceTable("transient")
                table.headers = list(headers)
                table.data = data
                result.add_table(table)

        self._update_signal_kinds_from_vector_info(result)

        return result

    def _update_signal_kinds_from_vector_info(self, result):
        vectors_info = self.vector_info()
        for vec_info in vectors_info:
            if vec_info.name in result.signals:
                if hasattr(vec_info, "quantity") and vec_info.quantity:
                    if vec_info.quantity.lower() in ("time", "index"):
                        result.signals[vec_info.name].kind = SignalKind.TIME
                    elif vec_info.quantity.lower() in ("voltage", "v"):
                        result.signals[vec_info.name].kind = SignalKind.VOLTAGE
                    elif vec_info.quantity.lower() in ("current", "i"):
                        result.signals[vec_info.name].kind = SignalKind.CURRENT

    def tran_async(
        self,
        tstep,
        tstop=None,
        *extra_args,
        throttle_interval: float = 0.1,
        buffer_size: int = 10,
        disable_buffering: bool = False,
        disable_throttling: bool = False,
        fallback_sampling_ratio: int = 100,
    ) -> "queue.Queue[dict]":
        """Run async transient simulation using chunked approach with stop after and step commands."""

        # Check if we need to restart the ngspice process
        # This prevents state accumulation after many simulations  
        if self._simulation_count >= self.RESTART_AFTER_N_SIMULATIONS:
            print(f"[RESTART] Restarting ngspice after {self._simulation_count} simulations (threshold={self.RESTART_AFTER_N_SIMULATIONS})")
            if self.debug:
                print(f"[debug] Restarting ngspice after {self._simulation_count} simulations")
            try:
                self._restart_ngspice_process()
            except Exception as e:
                if self.debug:
                    print(f"[debug] Warning: Could not restart ngspice process: {e}")
                # Continue anyway - the simulation might still work

        tstep_r = R(tstep)
        tstop_r = R(tstop) if tstop is not None else None

        tstep_val = float(tstep_r)
        tstop_val = float(tstop_r) if tstop_r is not None else None
        tstep_str = str(tstep_r)

        self._async_queue = queue.Queue()
        self._async_halt_requested = False
        self._async_resume_event = threading.Event()
        self._async_resume_event.set()
        self._async_current_time = 0.0
        self._data_points_sent = 0
        self._last_vector_length = 0
        self._is_running = False
        self._print_commands_count = 0  # Reset for new simulation
        self._simulation_count += 1  # Increment simulation counter
        
        print(f"[TRAN_ASYNC] Starting simulation #{self._simulation_count}, PID={self.p.pid if hasattr(self, 'p') and self.p else 'none'}")
        
        # Try to reset ngspice state before starting new simulation
        try:
            # Destroy any existing plots to free memory
            self.command("destroy all")
            # Reset simulation state
            self.command("reset")
        except (NgspiceError, NgspiceFatalError) as e:
            if self.debug:
                print(f"DEBUG: Could not reset ngspice state: {e}")

        self._async_thread = threading.Thread(
            target=self._run_chunked_simulation,
            args=(tstep_val, tstop_val, tstep_str, throttle_interval),
            daemon=True,
        )
        self._async_thread.start()

        return self._async_queue

    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running and self._async_thread is not None and self._async_thread.is_alive()

    def safe_halt_simulation(
        self, max_attempts: int = 3, wait_time: float = 0.2
    ) -> bool:
        """Halt simulation by setting halt flag and clearing resume event."""
        with self._async_lock:
            self._async_halt_requested = True
            self._async_resume_event.clear()
            self._is_running = False

        time.sleep(wait_time)
        return True

    def resume_simulation(self, timeout: float = 3.0) -> bool:
        """Resume simulation by clearing halt flag and setting resume event."""
        with self._async_lock:
            self._async_halt_requested = False
            self._async_resume_event.set()  # Signal resume
            self._is_running = True

        if self.debug:
            print("DEBUG: Resume requested")

        return True

    def safe_resume_simulation(
        self, max_attempts: int = 3, wait_time: float = 2.0
    ) -> bool:
        """Resume simulation safely with retry logic."""
        for attempt in range(max_attempts):
            result = self.resume_simulation(timeout=wait_time)
            if result:
                return True
            time.sleep(wait_time)
        return False

    def _is_header_line(self, line, expected_headers):
        """Check if a line looks like a header line."""
        if not line.strip():
            return False
        line_lower = line.lower()
        header_matches = 0
        for header in expected_headers:
            if header.lower() in line_lower:
                header_matches += 1
        return header_matches >= len(expected_headers) * 0.6

    def _print_new_vectors_only(self) -> Iterator[str]:
        """
        Print only new vector values using ngspice vector slicing.
        This avoids the need to parse and filter duplicate data points.
        """
        try:
            len_result = self.command("let current_len = length(time)")
            len_print = self.command("print current_len")


            match = re.search(r'current_len\s*=\s*([\d.e+-]+)', len_print)

            self.command("unlet current_len")

            if not match:
                if self.debug:
                    print(f"DEBUG: Could not extract vector length from: {len_print}")
                yield from self.print_all()
                return

            current_len = int(float(match.group(1)))

            if self.debug:
                print(f"DEBUG: Vector length: old={self._last_vector_length}, current={current_len}")

            if self._last_vector_length == 0:
                if self.debug:
                    print(f"DEBUG: First chunk, printing all {current_len} points")
                yield from self.print_all()
                self._last_vector_length = current_len
                return

            if current_len <= self._last_vector_length:
                # No new data
                if self.debug:
                    print(f"DEBUG: No new data (current_len={current_len}, last={self._last_vector_length})")
                return

            # Get vector names from display command, excluding temporary variables
            vectors_to_print = []
            display_output = self.command("display")
            for line in display_output.split("\n"):
                vector_match = re.match(
                    r"\s*([^:]+):\s*[^,]+,\s*[^,]+,\s*([0-9]+)\s+long", line
                )
                if vector_match:
                    vector_name = vector_match.group(1).strip()
                    if vector_name not in ["current_len", "old_len", "new_len", "start_idx", "end_idx"]:
                        vectors_to_print.append(vector_name)

            if not vectors_to_print:
                if self.debug:
                    print(f"DEBUG: No vectors found, falling back to print all")
                yield from self.print_all()
                self._last_vector_length = current_len
                return

            # Ngspice can slice vectors with brackets like @r1[i] using syntax @r1[i][5,9]
            # The parser correctly handles this sliced output format
            # No need to fall back to print_all for bracketed vectors
            
            # Build print command with slicing for only new values
            start_idx = self._last_vector_length
            end_idx = current_len - 1

            if self.debug:
                print(f"DEBUG: Printing slice [{start_idx}, {end_idx}] of {len(vectors_to_print)} vectors")

            # Check if total header length would exceed ngspice limit
            # If so, fall back to print_all with filtering
            # Separate time from other vectors
            time_vectors = [vec for vec in vectors_to_print if vec.lower() == 'time']
            other_vectors = [vec for vec in vectors_to_print if vec.lower() != 'time']
            
            # Split vectors into batches to avoid header truncation
            # Target: keep header line under 50 characters to be very safe
            # ngspice truncates at ~80 chars but we need margin for spacing
            # Be more conservative after empirical testing shows issues at 60
            max_header_len = 50
            base_len = len("Index   time            ")  # ~24 chars
            
            batches = []
            current_batch = []
            current_header_len = base_len
            
            # Account for time vector with slice notation
            time_slice_str = f"[{start_idx},{end_idx}]"
            time_with_slice_len = len("time") + len(time_slice_str) + 2  # +2 for spacing
            current_header_len += time_with_slice_len
            
            for vec in other_vectors:
                vec_with_slice_len = len(vec) + len(time_slice_str) + 2
                if current_header_len + vec_with_slice_len > max_header_len and current_batch:
                    # Batch is full, save it and start new one
                    batches.append(current_batch)
                    current_batch = [vec]
                    current_header_len = base_len + time_with_slice_len + vec_with_slice_len
                else:
                    current_batch.append(vec)
                    current_header_len += vec_with_slice_len
            
            # Add remaining batch
            if current_batch:
                batches.append(current_batch)
            
            if self.debug and len(batches) > 1:
                print(f"DEBUG: Split into {len(batches)} batches to avoid header truncation")
            
            # Execute print command for each batch and collect all output
            all_output_lines = []
            for batch_idx, batch in enumerate(batches):
                # Always include time in each batch
                vectors_in_batch = time_vectors + batch
                sliced_vectors = [f"{vec}[{start_idx},{end_idx}]" for vec in vectors_in_batch]
                print_cmd = f"print col {' '.join(sliced_vectors)}"
                
                if self.debug:
                    print(f"DEBUG: Batch {batch_idx+1}/{len(batches)}: {len(batch)} vectors, header_len~{len(print_cmd)}")
                    print(f"DEBUG: Command: {print_cmd[:100]}...")
                
                result = self.command(print_cmd)
                self._print_commands_count += 1
                
                if self.debug:
                    lines_in_batch = len(result.split("\n"))
                    print(f"DEBUG: Batch {batch_idx+1} returned {lines_in_batch} lines of output")
                
                all_output_lines.extend(result.split("\n"))
            
            if self.debug:
                print(f"DEBUG: Total batches={len(batches)}, total output lines={len(all_output_lines)}")
                print(f"DEBUG: Total print commands this session: {self._print_commands_count}")
            
            yield from all_output_lines

            # Update the last vector length
            self._last_vector_length = current_len

        except Exception as e:
            if self.debug:
                print(f"DEBUG: Error in _print_new_vectors_only: {e}, falling back to print all")
            # Fallback to print all on error
            yield from self.print_all()
            # Try to update length anyway
            try:
                len_result = self.command("let current_len = length(time)")
                len_print = self.command("print current_len")
                # Clean up
                self.command("unlet current_len")
                match = re.search(r'current_len\s*=\s*([\d.e+-]+)', len_print)
                if match:
                    self._last_vector_length = int(float(match.group(1)))
            except Exception as e:
                if self.debug:
                    print(f"[ngspice-subprocess] Error updating vector length: {e}")
                pass

    def _parse_and_enqueue_from_lines(
        self, lines: list, current_time: float, chunk_end: float, tstop: float | None
    ) -> None:
        """Parse print all output and enqueue data points for async simulation."""
        signal_data = {}
        signal_kinds = {}
        current_headers = None
        current_sliced_columns = None  # List of (col_index, clean_name) for sliced vectors
        time_column_index = None
        last_time_values = {}  # Map from row_index to time_value - used for cross-table row matching
        current_table_row_index = 0  # Track which row we're on in the current table
        global_row_index = 0  # Track absolute row index across all tables for cache key

        if self.debug:
            print(f"DEBUG: Parsing {len(lines)} lines, last_time_values cache starts empty")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip separator lines and headers
            if any(x in line for x in ("---", "print all", "Transient Analysis")):
                continue

            # Check for header line (contains "Index" and "time")
            if "Index" in line and "time" in line:
                # Reset row counter for new table
                current_table_row_index = 0
                
                # Parse headers
                import re
                raw_headers = line.split()

                if self.debug:
                    print(f"DEBUG: New table at global_row={global_row_index}. Headers: {raw_headers[:5]}..."  # Just first 5
                          + (f" (total {len(raw_headers)} headers)" if len(raw_headers) > 5 else ""))

                # Find sliced columns (those with [N,M] notation at the end)
                # Also detect incomplete slicing (headers ending with [N or [N,) and skip them
                sliced_columns = []
                has_incomplete_slicing = False
                for i, header in enumerate(raw_headers):
                    if re.search(r'\[\d+,\d+\]$', header):
                        # Complete slice notation
                        clean_name = re.sub(r'\[\d+,\d+\]$', '', header)
                        sliced_columns.append((i, clean_name))
                    elif re.search(r'\[\d+,?$', header):
                        # Incomplete slice notation (truncated header)
                        has_incomplete_slicing = True
                        if self.debug:
                            print(f"DEBUG: Detected incomplete slice notation in header: {repr(header)}")

                if sliced_columns and not has_incomplete_slicing:
                    # Using sliced vector output
                    current_sliced_columns = sliced_columns
                    current_headers = tuple([name for _, name in sliced_columns])
                    
                    # Find time column in sliced columns
                    time_column_index = None
                    for i, name in enumerate(current_headers):
                        if name.lower() == "time":
                            time_column_index = i
                            break
                    
                    if self.debug:
                        if time_column_index is not None:
                            print(f"DEBUG: Sliced vectors with time. Columns: {current_headers}, time_idx={time_column_index}")
                        else:
                            print(f"DEBUG: Sliced vectors without time. Columns: {current_headers}. Will use row matching.")
                else:
                    # Regular output (no slicing)
                    current_sliced_columns = None
                    # Remove any slice notation from headers (shouldn't be any, but just in case)
                    current_headers = tuple([re.sub(r'\[\d+,\d+\]$', '', h) for h in raw_headers])
                    
                    # Find time column - prefer last occurrence
                    time_column_index = None
                    for i in range(len(current_headers) - 1, -1, -1):
                        if current_headers[i].lower() == "time":
                            time_column_index = i
                            break
                    
                    if self.debug:
                        print(f"DEBUG: Regular output. Headers: {current_headers}, time_idx={time_column_index}")

                continue

            if not current_headers:
                continue

            # Parse data line
            if current_sliced_columns:
                # For sliced vectors, use tab separator and extract only sliced columns
                values_raw = line.split('\t')
                
                # Extract values from sliced columns only
                values = []
                has_data = False
                for col_idx, col_name in current_sliced_columns:
                    if col_idx < len(values_raw):
                        val = values_raw[col_idx].strip()
                        if val:
                            has_data = True
                        values.append(val)
                    else:
                        values.append("")
                
                # Skip rows where all sliced columns are empty
                if not has_data:
                    continue
                
                # Determine time value
                if time_column_index is not None:
                    # This table has a time column
                    try:
                        time_val = float(values[time_column_index])
                        # Cache this time using global row index for tables without time column
                        last_time_values[global_row_index] = time_val
                        if self.debug and global_row_index < 3:
                            print(f"DEBUG: Global row {global_row_index}: time={time_val}, caching")
                    except (ValueError, IndexError):
                        continue
                else:
                    # This table doesn't have time column, use cached time from same global row index
                    if global_row_index in last_time_values:
                        time_val = last_time_values[global_row_index]
                        if self.debug and global_row_index < 3:
                            print(f"DEBUG: Global row {global_row_index}: using cached time={time_val}")
                    else:
                        # No cached time for this row, skip
                        if self.debug and global_row_index < 3:
                            print(f"DEBUG: Global row {global_row_index}: NO cached time, skipping")
                        continue
                
                current_table_row_index += 1
                global_row_index += 1
            else:
                # For regular output, use whitespace splitting
                values = line.split()
                
                if len(values) < len(current_headers):
                    continue
                
                # Extract time value
                if time_column_index is None:
                    continue
                    
                try:
                    time_val = float(values[time_column_index])
                except (ValueError, IndexError):
                    continue

            if time_val not in signal_data:
                signal_data[time_val] = {}

            # Extract signal values
            for i, header in enumerate(current_headers):
                if i == 0 or header.lower() == "index":
                    continue  # Skip Index column
                if header.lower() == "time":
                    continue  # Skip time column (already extracted)
                if i < len(values) and values[i]:
                    try:
                        signal_val = float(values[i])
                        signal_data[time_val][header] = signal_val

                        temp_result = NgspiceResultBase()
                        signal_kinds[header] = temp_result.categorize_signal(header)
                    except (ValueError, IndexError):
                        continue

        if self.debug:
            print(f"DEBUG: Parsed {len(signal_data)} unique time points, last_time_values has {len(last_time_values)} entries")

        # Enqueue data points - no need to filter duplicates when using vector slicing
        for time_val, time_signals in sorted(signal_data.items()):
            with self._async_lock:
                if self._async_halt_requested:
                    if self.debug:
                        print(
                            f"DEBUG: Breaking due to halt request in chunk starting at {current_time}"
                        )
                    break

            data_point = {
                "timestamp": time.time(),
                "data": {"time": time_val},
                "signal_kinds": {"time": SignalKind.TIME},
                "index": self._data_points_sent,
                "progress": min(1.0, time_val / tstop)
                if tstop is not None and tstop > 0
                else 0.0,
            }

            signal_count = 0
            for signal_name, signal_val in time_signals.items():
                if signal_name == "time":
                    continue
                data_point["data"][signal_name] = signal_val
                data_point["signal_kinds"][signal_name] = signal_kinds.get(
                    signal_name, SignalKind.VOLTAGE
                )
                signal_count += 1

            # Skip data points with no actual signal data (only time)
            if signal_count == 0:
                if self.debug:
                    print(f"DEBUG: Skipping time_val={time_val} with no non-time signals")
                continue

            if self.debug and self._data_points_sent < 3:
                print(f"DEBUG: Data point {self._data_points_sent}: {data_point}")

            if self._async_queue:
                self._async_queue.put(data_point)
            self._data_points_sent += 1
            # Update the current time tracker
            self._async_current_time = time_val

    def _run_chunked_simulation(
        self,
        tstep: float,
        tstop: float | None,
        tstep_str: str,
        throttle_interval: float,
    ):
        """Run chunked transient simulation using stop after and step commands."""
        try:
            # Calculate chunk size (number of steps per chunk)
            if tstop is not None:
                # Aim for ~100 chunks across the simulation
                total_steps = int(tstop / tstep)
                chunk_steps = max(5, total_steps // 100)
            else:
                chunk_steps = 10

            # Start the transient analysis with "stop after" to pause after initial steps
            try:
                self.command(f"stop after {chunk_steps}")
                tran_cmd = f"tran {tstep_str} {tstop if tstop else tstep * 1000}"
                self.command(tran_cmd)
                self._is_running = True
                self._async_resume_event.set()  # Initially running
            except NgspiceError as e:
                error_data = {"error": f"Simulation failed to start: {str(e)}"}
                if self._async_queue:
                    self._async_queue.put(error_data)
                return

            # Main simulation loop
            simulation_complete = False
            while not simulation_complete:
                # Check if we should halt (wait for resume)
                if self._async_halt_requested:
                    with self._async_lock:
                        self._is_running = False

                    if self.debug:
                        print(f"DEBUG: Simulation halted, waiting for resume...")

                    # Wait for resume signal (with timeout to check for complete halt)
                    resumed = self._async_resume_event.wait(timeout=0.5)

                    # If still halted after timeout, check if we should exit
                    with self._async_lock:
                        if self._async_halt_requested and not resumed:
                            # Still halted, continue waiting
                            continue
                        elif self._async_halt_requested:
                            # Halt requested but no resume, exit
                            if self.debug:
                                print(f"DEBUG: Exiting due to halt without resume")
                            break
                        else:
                            # Resumed!
                            self._is_running = True
                            if self.debug:
                                print(f"DEBUG: Simulation resumed")

                # Get current simulation data using vector slicing (only new values)
                try:
                    print_all_res = "\n".join(self._print_new_vectors_only())
                except NgspiceError:
                    print_all_res = ""

                lines = print_all_res.split("\n") if print_all_res else []

                # Parse the current time from the output to determine chunk boundaries
                current_time = 0.0
                chunk_end = tstop or float('inf')

                # Extract time values from the output to determine current simulation time
                for line in lines:
                    line = line.strip()
                    if not line or "Index" in line:
                        continue
                    values = line.split()
                    if len(values) >= 2:
                        try:
                            time_val = float(values[1])
                            current_time = max(current_time, time_val)
                        except (ValueError, IndexError):
                            pass

                # Enqueue the data points
                self._parse_and_enqueue_from_lines(
                    lines, 0.0, chunk_end, tstop
                )

                # Continue simulation with step command (only if not halted)
                if not self._async_halt_requested:
                    try:
                        # Check if process is still alive before issuing step command
                        if self.p.poll() is not None:
                            if self.debug:
                                print(f"DEBUG: ngspice process terminated unexpectedly")
                            simulation_complete = True
                            break
                        
                        step_output = self.command(f"step {chunk_steps}")
                        # Step succeeded, simulation continues
                        if self.debug:
                            print(f"DEBUG: Step command succeeded")
                    except (NgspiceError, NgspiceFatalError) as e:
                        if self.debug:
                            print(f"DEBUG: Step command failed: {e}")
                        simulation_complete = True
                        break

                # Check if we've reached the target time (as a secondary check)
                if tstop is not None and current_time >= tstop * 0.9999:
                    if self.debug:
                        print(f"DEBUG: Reached target time {current_time} >= {tstop}")
                    # Get any remaining data
                    try:
                        final_print = "\n".join(self._print_new_vectors_only())
                        final_lines = final_print.split("\n") if final_print else []
                        if final_lines:
                            self._parse_and_enqueue_from_lines(
                                final_lines, 0.0, chunk_end, tstop
                            )
                    except Exception as e:
                        if self.debug:
                            print(f"DEBUG: Error getting final data: {e}")
                    simulation_complete = True
                    break

                time.sleep(min(throttle_interval, 0.05))

            with self._async_lock:
                self._is_running = False

            if not self._async_halt_requested:
                if self.debug:
                    print("DEBUG: Simulation completed normally")
                self._async_queue.put({"status": "completed"})
            else:
                if self.debug:
                    print("DEBUG: Simulation halted by request")
                self._async_queue.put({"status": "halted"})

        except Exception as e:
            with self._async_lock:
                self._is_running = False
            if self.debug:
                print(f"DEBUG: Exception in chunked simulation: {e}")
            error_data = {"error": f"Simulation error: {str(e)}"}
            if self._async_queue:
                self._async_queue.put(error_data)

    def _parse_ac_wrdata(self, file_path: str, vectors: list[str]) -> "NgspiceAcResult":
        """Parses the ASCII output of a wrdata command for AC analysis."""
        result = NgspiceAcResult()

        try:
            data = np.loadtxt(file_path)
        except (IOError, ValueError):
            return result

        if data.ndim == 1:
            # Handle case with only one row of data by reshaping it
            data = data.reshape(1, -1)

        if data.shape[0] == 0:
            return result

        result.freq = list(data[:, 0])

        # Subsequent columns are grouped in threes: freq, real, imag.
        for i, vec_name in enumerate(vectors):
            # The block for vector `i` starts at column i*3
            real_col_idx = i * 3 + 1
            imag_col_idx = i * 3 + 2

            if data.shape[1] > imag_col_idx:
                real_parts = data[:, real_col_idx]
                imag_parts = data[:, imag_col_idx]

                complex_data = [complex(r, i) for r, i in zip(real_parts, imag_parts)]
                kind = result.categorize_signal(vec_name)
                result.signals[vec_name] = SignalArray(kind=kind, values=complex_data)

        self._update_signal_kinds_from_vector_info(result)

        return result

    def ac(self, *args, wrdata_file: Optional[str] = None) -> "NgspiceAcResult":
        self.command(f"ac {' '.join(args)}")

        if wrdata_file is None:
            # Original logic using print all
            print_all_res = "".join(self.print_all())
            result = NgspiceAcResult()

            sections = re.split(r"AC Analysis\s+.*\n\s*-{60,}", print_all_res)

            for section in sections:
                if not section.strip():
                    continue

                lines = section.strip().split("\n")
                header_line = lines[0]
                data_lines = lines[1:]

                headers = header_line.split()
                if len(headers) < 2:
                    continue

                vector_name = headers[-1]

                if "frequency" in headers:
                    if not result.freq:
                        for line in data_lines:
                            match = re.match(r"\s*\d+\s+([\d.eE+-]+)", line)
                            if match:
                                result.freq.append(float(match.group(1)))

                signal_data = []
                for line in data_lines:
                    match = re.search(r"([\d.eE+-]+),\s*([\d.eE+-]+)", line)
                    if match:
                        real = float(match.group(1))
                        imag = float(match.group(2))
                        signal_data.append(complex(real, imag))

                if not signal_data:
                    continue

                kind = result.categorize_signal(vector_name)
                result.signals[vector_name] = SignalArray(kind=kind, values=signal_data)

            self._update_signal_kinds_from_vector_info(result)

            return result
        else:
            vectors_to_write = [
                v.name
                for v in self.vector_info()
                if v.name != "frequency" and v.length > 0
            ]
            if not vectors_to_write:
                return NgspiceAcResult()  # Return empty result if no vectors

            # Quote vector names to handle special characters
            vectors_quoted = [f'"{v}"' for v in vectors_to_write]
            self.command(f"wrdata {wrdata_file} {' '.join(vectors_quoted)}")
            return self._parse_ac_wrdata(wrdata_file, vectors_to_write)

    def vector_info(self) -> Iterator[NgspiceVector]:
        """Wrapper for ngspice's "display" command."""
        display_output = self.command("display")
        lines = display_output.split("\n")

        in_vectors_section = False
        for line in lines:
            if "Here are the vectors currently active:" in line:
                in_vectors_section = True
                continue

            if in_vectors_section:
                if (
                    len(line) == 0
                    or line.startswith("Title:")
                    or line.startswith("Name:")
                    or line.startswith("Date:")
                ):
                    continue
                res = re.match(
                    r"\s*([0-9a-zA-Z_.#@\[\]]*)\s*:\s*([a-zA-Z]+),\s*([a-zA-Z]+),\s*([0-9]+) long(.*)",
                    line,
                )
                if res:
                    name, vtype, dtype, length, rest = res.groups()
                    yield NgspiceVector(name, vtype, dtype, int(length), rest)

    def _is_numeric_row(self, row_data):
        """Check if a row contains mostly numeric data."""
        if not row_data:
            return False

        numeric_count = 0
        for item in row_data:
            try:
                float(item)
                numeric_count += 1
            except ValueError:
                # First column might be an index (integer)
                try:
                    int(item)
                    numeric_count += 1
                except ValueError:
                    pass

        # Consider it numeric if at least 80% of values are numbers
        return numeric_count >= len(row_data) * 0.8
