# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import queue

from multiprocessing import Process, Pipe
import pickle
from contextlib import contextmanager
import traceback

class FFIWorkerProcess:
    """Isolated FFI worker process that handles ngspice communication"""

    def __init__(self, conn):
        self.conn = conn
        self.backend = None
        self.debug = False
        self._captured_output = []
        self._captured_errors = []
        self._original_output_lines = None
        self._async_data_queue = queue.Queue()
        self._last_async_data = None

    def run(self):
        """Main worker loop"""
        # Initialize FFI backend in worker process
        from .ngspice_ffi import _FFIBackend

        # First message must be 'init'
        msg = self.conn.recv()
        if msg['type'] == 'init':
            self.debug = msg.get('debug', False)
            try:
                self.backend = _FFIBackend(debug=self.debug)
                print(f"Worker: Backend created, output lines: {len(self.backend._output_lines)}")
                for i, line in enumerate(self.backend._output_lines):
                    print(f"Worker: Initial output {i}: {line}")
                self.conn.send({'type': 'init_success'})
            except Exception as e:
                self.conn.send({'type': 'error', 'data': pickle.dumps(e), 'traceback': traceback.format_exc()})
                return # Terminate worker if init fails
        else:
            # Invalid startup sequence
            return

        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, BrokenPipeError):
                break  # Parent closed connection

            if msg['type'] == 'quit':
                if self.backend:
                    self.backend.cleanup()
                break

            # Dispatch command to the backend
            cmd = msg['type']
            args = msg.get('args', [])
            kwargs = msg.get('kwargs', {})

            # Handle polling requests
            if cmd == 'poll_async':
                data = self._get_async_data()
                self.conn.send({'type': 'result', 'data': pickle.dumps(data)})
                continue
            elif cmd == 'has_async_data':
                has_data = self._has_async_data()
                self.conn.send({'type': 'result', 'data': pickle.dumps(has_data)})
                continue

            try:
                method = getattr(self.backend, cmd)
                # For command execution, use a custom implementation to capture output
                if cmd == 'command':
                    # For command execution, capture output by temporarily replacing _output_lines
                    self._original_output_lines = self.backend._output_lines
                    self.backend._output_lines = self._captured_output

                    # Clear captured output before executing command
                    self._captured_output.clear()
                    self.backend._error_message = None
                    self.backend._has_fatal_error = False

                    print(f"Worker: Executing command: {args[0] if args else 'None'}")
                    print(f"Worker: Output lines before command: {len(self.backend._output_lines)}")

                    # Execute the command using the original method
                    result = method(*args, **kwargs)

                    print(f"Worker: Command executed, captured output: {len(self._captured_output)} items")
                    for i, line in enumerate(self._captured_output):
                        print(f"Worker: Captured {i}: {line}")

                    # Get the captured output and restore original output lines
                    output = "\n".join(self._captured_output)
                    self.backend._output_lines = self._original_output_lines

                    # Return the captured output instead of the method result
                    result = output
                    print(f"Worker: Final result length: {len(result)}")
                elif cmd in ['tran_async', 'op_async']:
                    # For async methods, we need to handle them specially
                    # since they return generators that can't be pickled
                    # We'll use a polling mechanism instead
                    result = method(*args, **kwargs)
                    # For async methods, we'll store the generator in the worker
                    # and provide polling methods to get results
                    if cmd == 'tran_async':
                        self._async_generator = result
                    elif cmd == 'op_async':
                        self._async_generator = result
                    # Return a token indicating async operation started
                    result = {'async_started': True, 'method': cmd}
                else:
                    # For other methods, use the normal implementation
                    result = method(*args, **kwargs)

                # Special handling for generators
                method_name = cmd
                if hasattr(result, '__iter__') and not isinstance(result, (list, tuple, dict, str)):
                    # For generators, collect all items and send as a list
                    if method_name.endswith('_async'):
                        # Async methods: stream items one by one
                        self.conn.send({'type': 'generator_start'})
                        for item in result:
                            self.conn.send({'type': 'generator_item', 'data': pickle.dumps(item)})
                        self.conn.send({'type': 'generator_end'})
                    else:
                        # Non-async generators: collect all items and send as list
                        items = list(result)
                        self.conn.send({'type': 'result', 'data': pickle.dumps(items)})
                else:
                    # For command methods, return the captured output instead of the result
                    if method_name == 'command':
                        # Return the command output
                        self.conn.send({'type': 'result', 'data': pickle.dumps(result)})
                        # Clear captured output for next command
                        self._captured_output.clear()
                    else:
                        self.conn.send({'type': 'result', 'data': pickle.dumps(result)})

            except Exception as e:
                self.conn.send({'type': 'error', 'data': pickle.dumps(e), 'traceback': traceback.format_exc()})



    def _get_async_data(self):
        """Get async data for polling"""
        if self._async_data_queue.empty():
            # Try to process more data if queue is empty
            self._process_async_data()

        if not self._async_data_queue.empty():
            return self._async_data_queue.get_nowait()
        return None

    def _has_async_data(self):
        """Check if there's async data available"""
        return not self._async_data_queue.empty() or self._process_async_data()




    # Note: The original FFI backend's _send_char_handler will now capture output
    # into our _captured_output list when we temporarily replace _output_lines


class IsolatedFFIBackend:
    """Isolated FFI backend using multiprocessing"""

    @staticmethod
    @contextmanager
    def launch(debug=False):
        parent_conn, child_conn = Pipe()

        # Start worker process
        worker = FFIWorkerProcess(child_conn)
        p = Process(target=worker.run)
        p.start()

        backend = None
        try:
            backend = IsolatedFFIBackend(parent_conn, p, debug)
            # Initialize the backend in the worker
            parent_conn.send({'type': 'init', 'debug': debug})
            # Wait for init confirmation
            response = parent_conn.recv()
            if response['type'] == 'error':
                 exc = pickle.loads(response['data'])
                 exc.args += (f"\n--- Traceback from worker process ---\n{response['traceback']}",)
                 raise exc

            yield backend
        finally:
            # Cleanup
            if backend:
                backend.close()
            if p.is_alive():
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()


    def __init__(self, conn, process, debug=False):
        self.conn = conn
        self.process = process
        self.debug = debug

    def close(self):
        try:
            if not self.conn.closed:
                self.conn.send({'type': 'quit'})
        except BrokenPipeError:
            pass  # Process might already be gone
        finally:
            if not self.conn.closed:
                self.conn.close()


    def _call_worker(self, msg_type, *args, **kwargs):
        if self.conn.closed:
            raise RuntimeError("Connection to FFI worker process is closed.")

        self.conn.send({'type': msg_type, 'args': args, 'kwargs': kwargs})
        response = self.conn.recv()

        if response['type'] == 'result':
            result_data = pickle.loads(response['data'])
            if self.debug:
                print(f"Main: Received result of type {type(result_data)}, length: {len(result_data) if hasattr(result_data, '__len__') else 'N/A'}")
                print(f"Main: Result content: {repr(result_data)}")
            return result_data
        elif response['type'] == 'error':
            exc = pickle.loads(response['data'])
            exc.args += (f"\n--- Traceback from worker process ---\n{response['traceback']}",)
            raise exc
        elif response['type'] == 'generator_start':
            # For async generators, collect all items and return as list
            items = []
            while True:
                item_response = self.conn.recv()
                if item_response['type'] == 'generator_end':
                    break
                elif item_response['type'] == 'generator_item':
                    items.append(pickle.loads(item_response['data']))
                elif item_response['type'] == 'error':
                    exc = pickle.loads(item_response['data'])
                    exc.args += (f"\n--- Traceback from worker process ---\n{item_response['traceback']}",)
                    raise exc
            return items
        else:
            raise RuntimeError(f"Unexpected response from worker: {response}")

    def _call_worker_async(self, msg_type, *args, **kwargs):
        """Special handling for async methods using polling"""
        if self.conn.closed:
            raise RuntimeError("Connection to FFI worker process is closed.")

        self.conn.send({'type': msg_type, 'args': args, 'kwargs': kwargs})
        response = self.conn.recv()

        if response['type'] == 'result':
            result = pickle.loads(response['data'])
            # For async methods, we return a special async handler
            if isinstance(result, dict) and result.get('async_started'):
                return AsyncResultHandler(self, msg_type)
            return result
        elif response['type'] == 'error':
            exc = pickle.loads(response['data'])
            exc.args += (f"\n--- Traceback from worker process ---\n{response['traceback']}",)
            raise exc
        else:
            raise RuntimeError(f"Unexpected response from worker: {response}")

    def _call_worker_sync(self, msg_type, *args, **kwargs):
        """Synchronous version of _call_worker that returns results directly"""
        result = self._call_worker(msg_type, *args, **kwargs)
        if self.debug:
            print(f"Main: _call_worker_sync received: {repr(result)} (type: {type(result)})")
        # If the result is a generator (from async methods), collect all items
        if hasattr(result, '__iter__') and not isinstance(result, (list, tuple, dict, str)):
            result_list = list(result)
            if self.debug:
                print(f"Main: _call_worker_sync converted generator to list: {len(result_list)} items")
            return result_list
        if self.debug:
            print(f"Main: _call_worker_sync returning: {repr(result)}")
        return result

    def _call_worker_str(self, msg_type, *args, **kwargs):
        """Synchronous version that ensures string return for commands"""
        result = self._call_worker_sync(msg_type, *args, **kwargs)
        if self.debug:
            print(f"Main: _call_worker_str received: {repr(result)} (type: {type(result)})")
        # For command results, join lists into strings
        if isinstance(result, list):
            return "\n".join(str(item) for item in result)
        final_result = str(result) if result is not None else ""
        if self.debug:
            print(f"Main: _call_worker_str returning: {repr(final_result)}")
        return final_result

    # --- Mirrored methods from _FFIBackend ---

    def command(self, command: str) -> str:
        if self.debug:
            print(f"Main: Sending command: {command}")
        result = self._call_worker_str('command', command)
        if self.debug:
            print(f"Main: Command result: {repr(result)}")
        return result

    def load_netlist(self, netlist: str, no_auto_gnd: bool = True):
        return self._call_worker_sync('load_netlist', netlist, no_auto_gnd=no_auto_gnd)

    def op(self):
        # op() returns a list of results from the worker
        return self._call_worker_sync('op')

    def tran(self, *args):
        # tran() returns a NgspiceTransientResult object, not a generator
        return self._call_worker_sync('tran', *args)

    def ac(self, *args, **kwargs):
        return self._call_worker_sync('ac', *args, **kwargs)

    def tran_async(self, *args, **kwargs):
        # Use async handler for polling-based async
        return self._call_worker_async('tran_async', *args, **kwargs)

    def op_async(self, *args, **kwargs):
        # Use async handler for polling-based async
        return self._call_worker_async('op_async', *args, **kwargs)

    def is_running(self) -> bool:
        return self._call_worker_sync('is_running')

    def stop_simulation(self):
        return self._call_worker_sync('stop_simulation')

    def reset(self):
        return self._call_worker_sync('reset')

    def cleanup(self):
        # The actual cleanup is handled by the worker process 'quit' message
        # The close() method on this object handles sending that message.
        pass


class AsyncResultHandler:
    """Handler for polling-based async results from multiprocessing backend"""

    def __init__(self, backend, method_name):
        self.backend = backend
        self.method_name = method_name
        self._completed = False
        self._error = None

    def poll(self, timeout=0.1):
        """Poll for new async data"""
        if self._completed or self._error:
            return None

        # Send poll request to worker
        try:
            self.backend.conn.send({'type': 'poll_async', 'method': self.method_name})
            response = self.backend.conn.recv()

            if response['type'] == 'result':
                data = pickle.loads(response['data'])
                if data is None:
                    # No data available
                    return None
                if data.get('status') == 'completed':
                    self._completed = True
                    return None
                if data.get('status') == 'error':
                    self._error = data.get('error')
                    raise RuntimeError(f"Async simulation error: {self._error}")
                return data
            elif response['type'] == 'error':
                exc = pickle.loads(response['data'])
                exc.args += (f"\n--- Traceback from worker process ---\n{response['traceback']}",)
                raise exc
        except (BrokenPipeError, EOFError):
            self._completed = True
            return None

        return None

    def has_data(self):
        """Check if there's data available without blocking"""
        try:
            # Use non-blocking check
            self.backend.conn.send({'type': 'has_async_data', 'method': self.method_name})
            response = self.backend.conn.recv()

            if response['type'] == 'result':
                return pickle.loads(response['data'])
        except (BrokenPipeError, EOFError):
            return False
        return False

    def is_completed(self):
        """Check if async operation is completed"""
        return self._completed

    def get_all(self):
        """Get all available results (blocking until completion)"""
        results = []
        while not self._completed and not self._error:
            data = self.poll()
            if data:
                results.append(data)
        return results

    def __iter__(self):
        """Make the handler iterable for convenience"""
        return self

    def __next__(self):
        """Get next result (blocking)"""
        if self._completed:
            raise StopIteration
        if self._error:
            raise RuntimeError(f"Async simulation error: {self._error}")

        while True:
            data = self.poll()
            if data:
                return data
            if self._completed:
                raise StopIteration
            # Small delay to avoid busy waiting
            import time
            time.sleep(0.01)
