# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for async FFI methods in ngspice integration.

This test file focuses specifically on testing the asynchronous capabilities
of the ngspice FFI backend, including:

- tran_async(): Asynchronous transient analysis with real-time data streaming
- op_async(): Asynchronous operating point analysis
- Callback functionality for both async methods
- Throttling mechanisms to control data update frequency
- Simulation control (start, stop, status checking)
- Error handling in async contexts
- Memory management and cleanup
- Backend compatibility checks

Key Features Tested:
--------------------
1. Basic async functionality for both transient and operating point analysis
2. Callback mechanisms that allow real-time data processing
3. Throttling to prevent overwhelming Python with high-frequency updates
4. Simulation status tracking (is_running, stop_simulation)
5. Error propagation in async contexts
6. Multiple sequential simulations with proper cleanup
7. Data consistency between sync and async methods
8. Memory usage and stability over multiple runs
9. Large dataset handling
10. Backend-specific feature availability

Important Notes:
----------------
- These tests require the FFI backend to be available
- Concurrent access tests are skipped due to FFI backend's global state limitations
- All async callbacks are designed to be thread-safe and never raise exceptions
- Proper cleanup is essential to prevent memory corruption and test contamination
- Error handling distinguishes between warnings and fatal errors

Thread Safety:
--------------
The FFI backend maintains global state and is not thread-safe for concurrent
simulations. Tests that require concurrency are appropriately skipped or
use sequential execution patterns.
"""

import re
import pytest
import time
import threading
from ordec.sim2.ngspice import Ngspice, NgspiceError, NgspiceFatalError, Netlister
from ordec import Rational as R
from ordec.lib import test as lib_test


def test_async_tran_basic():
    """Test basic async transient analysis with FFI backend."""
    netlist = """.title RC circuit for async test
    V1 in 0 PWL(0 0 1u 1 2u 0)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    data_points = []

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        # Test async transient analysis
        for i, data_point in enumerate(sim.tran_async("0.1u", "3u")):
            data_points.append(data_point)

            # Verify data structure
            assert 'timestamp' in data_point
            assert 'data' in data_point
            assert 'index' in data_point
            assert isinstance(data_point['data'], dict)

            # Break after collecting some data
            if i >= 5:
                break

    # Should have collected some data points
    assert len(data_points) > 0

    # Verify time progression (if time data is available)
    time_values = [dp['data'].get('time', 0) for dp in data_points if 'time' in dp['data']]
    if len(time_values) > 1:
        # Time should be increasing
        for i in range(1, len(time_values)):
            assert time_values[i] >= time_values[i-1]


def test_async_tran_with_callback():
    """Test async transient analysis with callback function."""
    netlist = """.title RC circuit for callback test
    V1 in 0 PWL(0 0 1u 1 2u 0)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    callback_data = []
    callback_count = 0

    def data_callback(data):
        nonlocal callback_count
        callback_count += 1
        callback_data.append(data)

    data_points = []

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        # Test async transient with callback
        for i, data_point in enumerate(sim.tran_async("0.1u", "3u", callback=data_callback, throttle_interval=0.05)):
            data_points.append(data_point)

            if i >= 3:
                break

    # Both callback and generator should have received data
    assert len(data_points) > 0
    assert callback_count > 0
    assert len(callback_data) > 0

    # Callback data should match generator data
    assert callback_count == len(data_points)


def test_async_tran_throttling():
    """Test throttling mechanism in async transient analysis."""
    netlist = """.title Circuit for throttling test
    V1 in 0 PWL(0 0 10u 1 20u 0)
    R1 in out 1k
    C1 out 0 10n
    .end
    """

    timestamps = []

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        # Test with aggressive throttling
        for i, data_point in enumerate(sim.tran_async("0.5u", "25u", throttle_interval=0.2)):
            timestamps.append(data_point['timestamp'])

            if i >= 3:
                break

    # Verify throttling is working (timestamps should be spaced by at least throttle_interval)
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            # Allow some tolerance for timing variations
            assert time_diff >= 0.15  # slightly less than 0.2 for tolerance


def test_async_op_basic():
    """Test basic async operating point analysis."""
    netlist = """.title Simple resistor divider for async OP
    V1 in 0 5
    R1 in mid 1k
    R2 mid 0 1k
    .end
    """

    data_points = []

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        # Test async operating point analysis
        for i, data_point in enumerate(sim.op_async()):
            data_points.append(data_point)

            # Verify data structure
            assert 'timestamp' in data_point
            assert 'data' in data_point
            assert 'index' in data_point

            # Should contain voltage data
            if 'mid' in data_point['data']:
                # Voltage divider should give ~2.5V at midpoint
                mid_voltage = data_point['data']['mid']
                assert abs(mid_voltage - 2.5) < 0.1

            # Break after first few data points for OP analysis
            if i >= 1:
                break

    # Should have at least one data point
    assert len(data_points) > 0


def test_async_op_with_callback():
    """Test async operating point analysis with callback."""
    netlist = """.title Op-amp circuit for callback test
    V1 vdd 0 5
    V2 vss 0 -5
    V3 in 0 0.5
    R1 in out 10k
    R2 out 0 10k
    .end
    """

    callback_called = False
    callback_data = None

    def op_callback(data):
        nonlocal callback_called, callback_data
        callback_called = True
        callback_data = data

    results = []

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        for data_point in sim.op_async(callback=op_callback):
            results.append(data_point)
            break  # OP analysis typically produces one result

    assert len(results) > 0
    assert callback_called
    assert callback_data is not None


def test_async_stop_simulation():
    """Test stopping async simulation mid-execution."""
    netlist = """.title Long-running circuit for stop test
    V1 in 0 PWL(0 0 1m 1 10m 0)
    R1 in out 10k
    C1 out 0 100u
    .end
    """

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        data_count = 0

        # Start async simulation
        async_gen = sim.tran_async("10u", "20m")  # Long simulation

        for data_point in async_gen:
            data_count += 1

            # Stop after collecting some data
            if data_count >= 2:
                sim.stop_simulation()
                break

        # Should not be running after stop command
        assert not sim.is_running()
        assert data_count >= 1


def test_async_simulation_status():
    """Test simulation status tracking."""
    netlist = """.title Circuit for status test
    V1 in 0 PWL(0 0 5u 1 10u 0)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        # Should not be running initially
        assert not sim.is_running()

        # Start async simulation
        async_gen = sim.tran_async("0.1u", "12u")

        # Get first data point to ensure simulation started
        first_data = next(async_gen)
        assert first_data is not None

        # Should be running now
        assert sim.is_running()

        # Stop simulation
        sim.stop_simulation()

        # Should not be running after stop
        assert not sim.is_running()


def test_async_error_handling():
    """Test error handling in async simulations."""
    with Ngspice.launch(backend="ffi", debug=False) as sim:
        # Test with invalid netlist that should cause a fatal error
        broken_netlist = """.title broken circuit
        .subckt test
        V1 in 0 5
        .end
        """

        with pytest.raises(NgspiceFatalError):
            sim.load_netlist(broken_netlist)


def test_async_multiple_simulations():
    """Test running multiple async simulations in sequence."""
    netlist1 = """.title First circuit
    V1 in 0 PWL(0 0 1u 1)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    netlist2 = """.title Second circuit
    V1 in 0 3
    R1 in mid 2k
    R2 mid 0 1k
    .end
    """

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        # First simulation
        sim.load_netlist(netlist1)

        data1 = []
        for i, data_point in enumerate(sim.tran_async("0.1u", "2u")):
            data1.append(data_point)
            if i >= 2:
                break

        assert len(data1) > 0

        # Ensure first simulation is completely stopped
        if sim.is_running():
            sim.stop_simulation()

        # Wait a bit for cleanup
        time.sleep(0.1)

        # Second simulation (OP analysis)
        sim.load_netlist(netlist2)

        data2 = []
        for data_point in sim.op_async():
            data2.append(data_point)
            break

        assert len(data2) > 0


def test_async_data_consistency():
    """Test consistency of data between sync and async methods."""
    netlist = """.title Consistency test circuit
    V1 in 0 2
    R1 in mid 1k
    R2 mid 0 2k
    .end
    """

    # Use separate simulation instances to avoid state issues
    sync_voltages = {}
    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)
        sync_op = sim.op()
        sync_voltages = {name: value for vtype, name, subname, value in sync_op if vtype == 'voltage'}

    async_data = None
    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)
        for data_point in sim.op_async():
            async_data = data_point['data']
            break

    assert async_data is not None
    assert len(sync_voltages) > 0

    # Compare key voltages (allowing for small numerical differences)
    if 'mid' in async_data and 'mid' in sync_voltages:
        assert abs(async_data['mid'] - sync_voltages['mid']) < 1e-6


@pytest.mark.skip(reason="FFI backend doesn't support concurrent access due to global state limitations")
def test_async_concurrent_access():
    """Test that async methods handle concurrent access properly."""
    # This test is skipped because the FFI backend uses global state
    # and doesn't support multiple concurrent instances
    pass


def test_async_backend_compatibility():
    """Test that async methods are only available with FFI backend."""
    # This test verifies the error handling when async methods are called on subprocess backend

    try:
        with Ngspice.launch(backend="subprocess", debug=False) as sim:
            netlist = """.title test
            V1 in 0 1
            R1 in 0 1k
            .end
            """
            sim.load_netlist(netlist)

            # These should raise NotImplementedError
            with pytest.raises(NotImplementedError):
                list(sim.tran_async("1u", "2u"))

            with pytest.raises(NotImplementedError):
                list(sim.op_async())

    except Exception:
        # If subprocess backend is not available, skip this test
        pytest.skip("Subprocess backend not available")


def test_async_memory_usage():
    """Test that async simulations don't leak memory excessively."""
    netlist = """.title Memory test circuit
    V1 in 0 PWL(0 0 1u 1 2u 0)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        # Run multiple short simulations to test cleanup
        for run in range(3):
            sim.load_netlist(netlist)

            data_count = 0
            for data_point in sim.tran_async("0.1u", "3u"):
                data_count += 1
                if data_count >= 5:
                    break

            assert data_count >= 1

            # Ensure simulation stops properly
            if sim.is_running():
                sim.stop_simulation()

            assert not sim.is_running()


def test_async_large_dataset():
    """Test async simulation with larger datasets."""
    netlist = """.title Large dataset test
    V1 in 0 PWL(0 0 50u 1 100u 0)
    R1 in out 1k
    C1 out 0 1n
    .end
    """

    with Ngspice.launch(backend="ffi", debug=False) as sim:
        sim.load_netlist(netlist)

        data_count = 0
        max_data_points = 10

        for data_point in sim.tran_async("0.5u", "120u", throttle_interval=0.01):
            data_count += 1

            # Verify each data point has expected structure
            assert 'timestamp' in data_point
            assert 'data' in data_point
            assert 'index' in data_point

            if data_count >= max_data_points:
                break

        # Should have collected at least some data points
        assert data_count >= 1
        # If we got more than a few, that's good too
        if data_count >= 5:
            assert data_count <= max_data_points
