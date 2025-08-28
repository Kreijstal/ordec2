# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import threading
import time
import uuid
from multiprocessing import Process, Pipe, Queue
import pickle
from contextlib import contextmanager
import traceback
from typing import Optional, Callable, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ngspice_mp')


class FFIWorkerProcess:
    """Isolated FFI worker process that handles ngspice communication"""

    def __init__(self, conn, event_queue):
        self.conn = conn
        self.event_queue = event_queue
        self.backend = None
        self.debug = False
        self._captured_output = []
        self._captured_errors = []
        self._original_output_lines = None
        self._async_data_queue = mp.Queue()
        self._last_async_data = None
        self._current_token = None
        self._async_generator = None

    def run(self):
        """Main worker loop"""
        logger.debug("Worker process started")
        # Initialize FFI backend in worker process
        from .ngspice_ffi import _FFIBackend

        # First message must be 'init'
        msg = self.conn.recv()
        if msg['type'] == 'init':
            self.debug = msg.get('debug', False)
            logger.debug(f"Worker received init message, debug={self.debug}")
            try:
                self.backend = _FFIBackend(debug=self.debug)
                logger.debug(f"Worker: Backend created, output lines: {len(self.backend._output_lines)}")
                for i, line in enumerate(self.backend._output_lines):
                    logger.debug(f"Worker: Initial output {i}: {line}")
                self.conn.send({'type': 'init_success'})
                logger.debug("Worker sent init_success")
            except Exception as e:
                logger.error(f"Worker init failed: {e}")
                self.conn.send({'type': 'error', 'data': pickle.dumps(e), 'traceback': traceback.format_exc()})
                return # Terminate worker if init fails
        else:
            # Invalid startup sequence
            logger.error(f"Invalid startup message: {msg}")
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
            token = msg.get('token')
            logger.debug(f"Worker received command: {cmd}, token: {token}")

            # Store current token for callback identification
            self._current_token = token

            # Handle polling requests
            if cmd == 'poll_async':
                logger.debug("Worker processing poll_async request")
                data = self._get_async_data()
                self.conn.send({'type': 'result', 'data': pickle.dumps(data)})
                logger.debug("Worker sent poll_async response")
                continue
            elif cmd == 'has_async_data':
                logger.debug("Worker processing has_async_data request")
                has_data = self._has_async_data()
                self.conn.send({'type': 'result', 'data': pickle.dumps(has_data)})
                logger.debug("Worker sent has_async_data response")
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

                    logger.debug(f"Worker: Executing command: {args[0] if args else 'None'}")
                    logger.debug(f"Worker: Output lines before command: {len(self.backend._output_lines)}")

                    # Execute the command using the original method
                    result = method(*args, **kwargs)

                    logger.debug(f"Worker: Command executed, captured output: {len(self._captured_output)} items")
                    for i, line in enumerate(self._captured_output):
                        logger.debug(f"Worker: Captured {i}: {line}")

                    # Get the captured output and restore original output lines
                    output = "\n".join(self._captured_output)
                    self.backend._output_lines = self._original_output_lines

                    # Return the captured output instead of the method result
                    result = output
                    logger.debug(f"Worker: Final result length: {len(result)}")
                elif cmd in ['tran_async', 'op_async']:
                    # For async methods with callback support
                    # We need to wrap the callback to send events to the main process
                    user_callback = kwargs.pop('callback', None)
                    throttle_interval = kwargs.pop('throttle_interval', 0.1)

                    if user_callback:
                        # Replace with our callback that sends events
                        def wrapped_callback(data_point):
                            # Send callback event to main process
                            logger.debug(f"Worker callback for token {token}, data: {data_point}")
                            try:
                                self.event_queue.put({
                                    'type': 'callback',
                                    'token': token,
                                    'data': pickle.dumps(data_point)
                                })
                                logger.debug("Worker callback event queued successfully")
                            except Exception as e:
                                logger.error(f"Worker callback queue error: {e}")

                        kwargs['callback'] = wrapped_callback
                        kwargs['throttle_interval'] = throttle_interval

                    result = method(*args, **kwargs)

                    # For async methods, store the generator and return token
                    self._async_generator = result
                    result = {'async_started': True, 'method': cmd, 'token': token}
                    logger.debug(f"Async method {cmd} started with token {token}")
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

    def _process_async_data(self):
        """Process async data from generator"""
        if self._async_generator:
            try:
                item = next(self._async_generator)
                self._async_data_queue.put(item)
                return True
            except StopIteration:
                self._async_generator = None
        return False


class IsolatedFFIBackend:
    """Isolated FFI backend using multiprocessing with callback support"""

    @staticmethod
    @contextmanager
    def launch(debug=False):
        parent_conn, child_conn = Pipe()
        event_queue = Queue()

        # Start worker process
        worker = FFIWorkerProcess(child_conn, event_queue)
        p = Process(target=worker.run)
        p.start()

        backend = None
        try:
            backend = IsolatedFFIBackend(parent_conn, p, event_queue, debug)
            # Initialize the backend in the worker
            parent_conn.send({'type': 'init', 'debug': debug})
            # Wait for init confirmation
            response = parent_conn.recv()
            if response['type'] == 'error':
                 exc = pickle.loads(response['data'])
                 exc.args += (f"\n--- Traceback from worker process ---\n{response['traceback']}",)
                 raise exc

            # Start event listener thread
            backend._start_event_listener()

            yield backend
        finally:
            # Cleanup
            if backend:
                backend.close()
            if p.is_alive():
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()

    def __init__(self, conn, process, event_queue, debug=False):
        self.conn = conn
        self.process = process
        self.event_queue = event_queue
        self.debug = debug
        self._callbacks: Dict[str, Callable] = {}
        self._event_listener_thread = None
        self._stop_event = threading.Event()

    def close(self):
        """Close the backend and stop event listener"""
        self._stop_event.set()
        if self._event_listener_thread and self._event_listener_thread.is_alive():
            self._event_listener_thread.join(timeout=1.0)

        try:
            if not self.conn.closed:
                self.conn.send({'type': 'quit'})
        except BrokenPipeError:
            pass  # Process might already be gone
        finally:
            if not self.conn.closed:
                self.conn.close()

    def _start_event_listener(self):
        """Start the event listener thread"""
        self._event_listener_thread = threading.Thread(
            target=self._event_listener_loop,
            daemon=True
        )
        self._event_listener_thread.start()

    def _event_listener_loop(self):
        """Main loop for processing events from worker process"""
        logger.debug("Event listener thread started")
        while not self._stop_event.is_set():
            try:
                # Check for events with timeout to allow graceful shutdown
                try:
                    event = self.event_queue.get(timeout=0.1)
                    logger.debug(f"Event listener received event: {event['type']}")
                except mp.queues.Empty:
                    continue

                if event['type'] == 'callback':
                    token = event.get('token')
                    callback = self._callbacks.get(token)
                    if callback:
                        try:
                            data = pickle.loads(event['data'])
                            logger.debug(f"Event listener calling callback for token {token}")
                            callback(data)
                            logger.debug(f"Callback for token {token} completed successfully")
                        except Exception as e:
                            logger.error(f"Error in callback for token {token}: {e}")
                            if self.debug:
                                import traceback
                                traceback.print_exc()
                    else:
                        logger.warning(f"No callback found for token {token}")

            except Exception as e:
                logger.error(f"Error in event listener: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def _call_worker(self, msg_type, *args, **kwargs):
        if self.conn.closed:
            raise RuntimeError("Connection to FFI worker process is closed.")

        self.conn.send({'type': msg_type, 'args': args, 'kwargs': kwargs})
        response = self.conn.recv()

        if response['type'] == 'result':
            result_data = pickle.loads(response['data'])
            if self.debug:
                logger.debug(f"Main: Received result of type {type(result_data)}, length: {len(result_data) if hasattr(result_data, '__len__') else 'N/A'}")
                logger.debug(f"Main: Result content: {repr(result_data)}")
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
                logger.debug(f"Main received generator item: {item_response['type']}")
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
        """Special handling for async methods with callback support"""
        if self.conn.closed:
            raise RuntimeError("Connection to FFI worker process is closed.")

        # Extract callback from kwargs if present
        callback = kwargs.pop('callback', None)
        throttle_interval = kwargs.pop('throttle_interval', 0.1)

        # Generate unique token for this async operation
        token = str(uuid.uuid4())
        logger.debug(f"Main: Generated token {token} for async operation")

        # Store callback if provided
        if callback:
            self._callbacks[token] = callback
            logger.debug(f"Main: Stored callback for token {token}")

        # Send command to worker with token
        logger.debug(f"Main: Sending async command {msg_type} with token {token}")
        self.conn.send({
            'type': msg_type,
            'args': args,
            'kwargs': kwargs,
            'token': token
        })
        logger.debug("Main: Waiting for worker response...")
        response = self.conn.recv()
        logger.debug(f"Main: Received response: {response['type']}")

        if response['type'] == 'result':
            result = pickle.loads(response['data'])
            # For async methods, we return a special async handler
            if isinstance(result, dict) and result.get('async_started'):
                logger.debug(f"Main: Async operation started, returning handler for token {token}")
                return AsyncResultHandler(self, msg_type, token)
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

    def tran_async(self, *args, callback: Optional[Callable] = None, throttle_interval: float = 0.1):
        # Use async handler with callback support
        return self._call_worker_async('tran_async', *args, callback=callback, throttle_interval=throttle_interval)

    def op_async(self, *args, callback: Optional[Callable] = None):
        # Use async handler with callback support
        return self._call_worker_async('op_async', *args, callback=callback)

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

    def __init__(self, backend, method_name, token):
        self.backend = backend
        self.method_name = method_name
        self.token = token
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
                    # Remove callback registration
                    if self.token in self.backend._callbacks:
                        del self.backend._callbacks[self.token]
                    return None
                if data.get('status') == 'error':
                    self._error = data.get('error')
                    # Remove callback registration
                    if self.token in self.backend._callbacks:
                        del self.backend._callbacks[self.token]
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
            time.sleep(0.01)
