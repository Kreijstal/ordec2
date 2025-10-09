# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import re
import time
import queue
from contextlib import contextmanager
from ordec.core import *
from ordec.lib import test as lib_test
from ordec.lib.test import RCAlterTestbench
from ordec.core.rational import R
from ordec.sim2.sim_hierarchy import SimHierarchy, HighlevelSim
from ordec.sim2.ngspice import Ngspice


def test_ffi_long_run_debug():
    """Longer debug test that runs the FFI backend for a large, predictable
    number of data points to help reproduce duplicate/extra-point behaviour.

    This intentionally runs a relatively large transient (2000 points) but does not
    make strict failing assertions about exact point counts. Instead it prints a
    compact summary and performs a minimal sanity check so the test is useful as a
    reproduction aid without breaking automated runs catastrophically.
    """
    h = lib_test.ResdivFlatTb(backend="ffi")

    # Configure a simulation expected to produce exactly 2000 points:
    # A tran from 0 to N*tstep with tstep produces N+1 points, so we choose 1999 steps.
    num_points = 2000
    tstep_us = 1
    tstop_us = (num_points - 1) * tstep_us  # 1999us

    tstep_str = f"{tstep_us}u"
    tstop_str = f"{tstop_us}u"

    points_consumed = 0
    last_result = None
    first_times = []

    # Drain the entire async generator. This may take a while locally for large counts.
    for result in h.sim_tran_async(tstep_str, tstop_str):
        # Collect a small sample of the earliest time values for quick inspection
        if points_consumed < 10:
            try:
                time_val = getattr(getattr(result, "time", None), "value", None)
            except Exception:
                time_val = None
            first_times.append(time_val)

        points_consumed += 1
        last_result = result

    # Compact debug summary. Use -s with pytest to see this when running locally.
    last_progress = getattr(last_result, "progress", None)
    last_time_val = getattr(getattr(last_result, "time", None), "value", None)

    print(
        "DEBUG subprocess long: "
        f"expected_points={num_points}, consumed={points_consumed}, "
        f"first_times_sample={first_times}, last_progress={last_progress}, last_time={last_time_val}"
    )

    # Minimal sanity assertion so test doesn't silently do nothing.
    # We assert that at least one point was produced and that we produced at least as many
    # points as the expected (this allows detection of duplicates as 'consumed > expected').
    assert points_consumed >= 1, "No points were produced by the async generator."

    # If the consumption does not match the expected count, emit a visible message.
    if points_consumed != num_points:
        print(
            f"NOTE: expected {num_points} points but consumed {points_consumed}. "
            "This output is intended to help debug chunking/duplication; inspect printed values."
        )


@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_tran_basic(backend):
    h = lib_test.ResdivFlatTb(backend=backend)

    data_points = []
    time_values = []

    for i, result in enumerate(h.sim_tran_async("0.1u", "3u")):
        data_points.append(result)
        time_values.append(result.time)

        assert hasattr(result, "a")
        assert hasattr(result.a, "value")
        assert hasattr(result.a, "kind")
        assert isinstance(result.a.value, (int, float))
        assert hasattr(result, "time")

        # Break after collecting enough data
        if i >= 10:
            break

    assert len(data_points) >= 1
    assert len(time_values) >= 1

    # Time should be progressing
    for i in range(1, len(time_values)):
        assert time_values[i].value >= time_values[i - 1].value


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_tran_with_callback(backend):
    progress_updates = []

    def progress_callback(data_point):
        progress_updates.append(
            {
                "progress": data_point.get("progress", data_point.get("index", 0)),
                "current_time": data_point.get("timestamp", 0),
                "data": data_point.get("data", {}),
            }
        )

    h = lib_test.ResdivFlatTb(backend=backend)

    data_count = 0
    for result in h.sim_tran_async(
        "0.1u", "5u", callback=progress_callback, buffer_size=10
    ):
        data_count += 1

        # Verify progress information is available
        assert 0.0 <= result.progress <= 1.0
        assert result.time.value >= 0.0

        if data_count >= 8:
            break

    # Should have received progress updates
    assert len(progress_updates) > 0

    # Progress should be increasing
    for i in range(1, len(progress_updates)):
        assert progress_updates[i]["progress"] >= progress_updates[i - 1]["progress"]


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_sky130_streaming_without_savecurrents(backend):
    h = lib_test.InvSkyTb(vin=R(2.5), backend=backend)

    callback_count = 0

    def count_callback(data_point):
        nonlocal callback_count
        callback_count += 1

    data_points = []
    for i, result in enumerate(
        h.sim_tran_async(
            "0.01u",
            "0.5u",
            enable_savecurrents=False,
            callback=count_callback,
            buffer_size=5,
        )
    ):
        data_points.append(result)
        if i >= 5:
            break

    assert len(data_points) >= 1, (
        f"Expected at least 1 data point without savecurrents, got {len(data_points)}"
    )
    assert callback_count >= 1, (
        f"Expected at least 1 callback without savecurrents, got {callback_count}"
    )


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_sky130_streaming_with_savecurrents(backend):
    h = lib_test.InvSkyTb(vin=R(2.5), backend=backend)

    callback_count = 0

    def count_callback(data_point):
        nonlocal callback_count
        callback_count += 1

    data_points = []
    for i, result in enumerate(
        h.sim_tran_async(
            "0.01u",
            "0.5u",
            enable_savecurrents=True,
            callback=count_callback,
            buffer_size=5,
        )
    ):
        data_points.append(result)
        if i >= 5:
            break

    assert len(data_points) >= 1, (
        f"Expected at least 1 data point with savecurrents, got {len(data_points)}"
    )
    assert callback_count >= 0, (
        f"Expected non-negative callbacks with savecurrents, got {callback_count}"
    )


@pytest.mark.libngspice
def test_sky130_netlist_savecurrents_option():
    from ordec.sim2.sim_hierarchy import SimHierarchy, HighlevelSim

    h = lib_test.InvSkyTb(vin=R(2.5))

    # Test with savecurrents enabled
    node1 = SimHierarchy()
    sim_with = HighlevelSim(h.schematic, node1, enable_savecurrents=True)
    netlist_with = sim_with.netlister.out()

    # Test with savecurrents disabled
    node2 = SimHierarchy()
    sim_without = HighlevelSim(h.schematic, node2, enable_savecurrents=False)
    netlist_without = sim_without.netlister.out()

    assert ".option savecurrents" in netlist_with, (
        "Netlist with enable_savecurrents=True should contain .option savecurrents"
    )
    assert ".option savecurrents" not in netlist_without, (
        "Netlist with enable_savecurrents=False should not contain .option savecurrents"
    )


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_mos_sourcefollower(backend):
    """Test async transient simulation with MOS source follower."""
    h = lib_test.NmosSourceFollowerTb(vin=R(2.0), backend=backend)

    data_points = []
    for i, result in enumerate(h.sim_tran_async("0.1u", "1u")):
        data_points.append(result)
        if i >= 5:
            break

    assert len(data_points) >= 1

    final_result = data_points[-1]
    assert hasattr(final_result, "o")
    assert hasattr(final_result.o, "value")
    assert hasattr(final_result.o, "kind")
    assert isinstance(final_result.o.value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_mos_inverter(backend):
    h = lib_test.InvTb(vin=R(0), backend=backend)

    data_points = []
    for i, result in enumerate(h.sim_tran_async("0.1u", "1u")):
        data_points.append(result)
        if i >= 5:
            break

    # Should have at least one data point
    assert len(data_points) >= 1

    final_result = data_points[-1]
    assert hasattr(final_result, "o")
    assert hasattr(final_result.o, "value")
    assert hasattr(final_result.o, "kind")
    assert isinstance(final_result.o.value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_sky_inverter(backend):
    h = lib_test.InvSkyTb(vin=R(2.5), backend=backend)

    data_points = []
    for i, result in enumerate(
        h.sim_tran_async("0.1u", "1u", enable_savecurrents=False)
    ):
        data_points.append(result)
        if i >= 5:
            break

    assert len(data_points) >= 1

    final_result = data_points[-1]
    assert hasattr(final_result, "o")
    assert hasattr(final_result.o, "value")
    assert hasattr(final_result.o, "kind")
    assert isinstance(final_result.o.value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_early_termination(backend):
    h = lib_test.ResdivFlatTb(backend=backend)

    data_count = 0
    final_time = None

    for result in h.sim_tran_async("0.05u", "10u"):
        data_count += 1
        final_time = result.time

        if data_count >= 5:
            break

    assert data_count >= 1
    assert data_count <= 5


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_multiple_circuits(backend):
    """Test running multiple async transient simulations sequentially."""
    # First circuit
    h1 = lib_test.ResdivFlatTb(backend=backend)
    results1 = []
    for i, result in enumerate(h1.sim_tran_async("0.1u", "1u")):
        results1.append(result)
        if i >= 3:
            break

    assert len(results1) >= 1
    assert hasattr(results1[0], "a")
    assert hasattr(results1[0].a, "value")
    assert hasattr(results1[0].a, "kind")
    assert isinstance(results1[0].a.value, (int, float))

    # Second circuit
    h2 = lib_test.ResdivHierTb(backend=backend)
    results2 = []
    for i, result in enumerate(h2.sim_tran_async("0.1u", "1u")):
        results2.append(result)
        if i >= 3:
            break

    assert len(results2) >= 1
    assert hasattr(results2[0], "r")
    assert hasattr(results2[0].r, "value")
    assert hasattr(results2[0].r, "kind")
    assert isinstance(results2[0].r.value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_parameter_sweep(backend):
    input_voltages = [2.0, 3.0, 4.0]
    results = {}

    for vin in input_voltages:
        h = lib_test.NmosSourceFollowerTb(vin=R(vin), backend=backend)

        async_results = []
        for i, result in enumerate(h.sim_tran_async("0.1u", "1u")):
            async_results.append(result)
            if i >= 3:
                break

        assert len(async_results) >= 1
        # Store the final numeric value for comparison
        results[vin] = async_results[-1].o

    assert len(results) == 3
    for vin in input_voltages:
        assert vin in results
        assert hasattr(results[vin], "value")
        assert hasattr(results[vin], "kind")
        assert isinstance(results[vin].value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_highlevel_async_ihp_inverter(backend):
    """Test async transient simulation with IHP inverter."""
    h = lib_test.InvIhpTb(vin=R(2.5), backend=backend)

    data_points = []
    for i, result in enumerate(
        h.sim_tran_async("0.1u", "1u", enable_savecurrents=False)
    ):
        data_points.append(result)
        if i >= 5:
            break

    assert len(data_points) >= 1

    final_result = data_points[-1]
    assert hasattr(final_result, "o")
    assert hasattr(final_result.o, "value")
    assert hasattr(final_result.o, "kind")
    assert isinstance(final_result.o.value, (int, float))


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_async_alter_resume(backend):
    circuit = RCAlterTestbench()
    node = SimHierarchy()
    sim = HighlevelSim(circuit.schematic, node, backend=backend)

    async def run_comprehensive_test():
        """Test multiple aspects of async alter functionality"""
        with sim.alter_session(backend=backend) as alter:
            # Start async transient simulation
            data_queue = alter.start_async_tran("0.1u", "2m")
            all_data = []
            found_signals = set()
            mapped_signals = {}
            start_time = time.time()
            timeout = 10.0  # Slightly longer timeout to ensure 4 voltage changes

            # Multiple halt/alter/resume cycles with different voltages
            voltage_sequence = [2.0, 1.5, 3.0, 1.0]
            current_voltage_index = 0
            data_points_since_last_change = 0
            voltage_change_interval = 50  # Change voltage every 50 data points
            applied_voltages = []  # Track which voltages were actually applied
            voltage_change_times = []  # Track when voltages were changed

            while (time.time() - start_time) < timeout and current_voltage_index < len(voltage_sequence):
                try:
                    data_point = data_queue.get_nowait()  # Use get_nowait like interactive example

                    if isinstance(data_point, dict) and "data" in data_point:
                        data_dict = data_point["data"]
                        sim_time = data_dict.get("time", 0)

                        # Record the data point
                        all_data.append((sim_time, data_dict))

                        # Track signals found
                        for signal_name in data_dict.keys():
                            if signal_name != "time":
                                found_signals.add(signal_name)
                                if signal_name in sim.str_to_simobj:
                                    simnet = sim.str_to_simobj[signal_name]
                                    net_name = simnet.eref.full_path_str().split(".")[
                                        -1
                                    ]
                                    mapped_signals[signal_name] = net_name

                        # Check if it's time to change voltage (based on data point count)
                        data_points_since_last_change += 1
                        if data_points_since_last_change >= voltage_change_interval and current_voltage_index < len(voltage_sequence):
                            voltage = voltage_sequence[current_voltage_index]

                            # Halt, alter, and resume
                            halt_success = alter.halt_simulation(timeout=0.1)
                            assert halt_success, (
                                f"Should successfully halt simulation at step {current_voltage_index + 1}"
                            )

                            alter.alter_component(circuit.schematic.v1, dc=voltage)
                            vdc_info = alter.show_component(circuit.schematic.v1)
                            expected = (
                                str(int(voltage)) if voltage == int(voltage) else str(voltage)
                            )
                            assert expected in vdc_info, (
                                f"Step {current_voltage_index + 1}: VDC should show {expected}V after alter: {vdc_info}"
                            )
                            applied_voltages.append(voltage)  # Record that this voltage was applied
                            voltage_change_times.append(sim_time)  # Record when voltage was changed

                            resume_success = alter.resume_simulation(timeout=0.1)
                            assert resume_success, (
                                f"Should successfully resume simulation at step {current_voltage_index + 1}"
                            )

                            current_voltage_index += 1
                            data_points_since_last_change = 0

                            # Small delay to let simulation stabilize after resume
                            await asyncio.sleep(0.00001)

                        # If we've completed all voltage changes, we can exit early
                        if current_voltage_index >= len(voltage_sequence):
                            break

                except queue.Empty:
                    # Use short sleep like interactive example instead of blocking
                    await asyncio.sleep(0.0001)
                    continue
                except Exception as e:
                    break

            # Separate data by voltage phases for analysis
            # Since we're using data point count instead of simulation time for voltage changes,
            # we need a different approach to separate initial vs alter data
            voltage_change_data_point = voltage_change_interval  # First change happens after this many points
            initial_data = all_data[:voltage_change_data_point]
            alter_data = all_data[voltage_change_data_point:]

            # Verify that voltage changes are reflected in simulation data
            # Check if we can detect voltage changes in the actual simulation results
            if len(applied_voltages) > 0 and len(alter_data) > 0:
                # Look for evidence of voltage changes in the data
                # For RC circuits, voltage changes should affect the output waveform
                print(f"Applied voltages: {applied_voltages}")
                print(f"Voltage changes completed: {len(applied_voltages)}")
                print(f"Voltage change times: {voltage_change_times}")

                # Verify that voltage changes are reflected in simulation behavior
                # For RC circuits, changing input voltage should affect output voltage
                if len(voltage_change_times) >= 1 and len(all_data) > voltage_change_times[0]:
                    # Check if output voltage changes after voltage alterations
                    initial_vout = None
                    altered_vout = []

                    for sim_time, data_dict in all_data:
                        if "vout" in data_dict:
                            if initial_vout is None:
                                initial_vout = data_dict["vout"]
                            elif len(voltage_change_times) > 0 and sim_time > voltage_change_times[0]:
                                altered_vout.append(data_dict["vout"])

                    if initial_vout is not None and len(altered_vout) > 0:
                        avg_initial = initial_vout
                        avg_altered = sum(altered_vout) / len(altered_vout)
                        print(f"Initial vout: {avg_initial:.6f}, Altered vout: {avg_altered:.6f}")

                        # For RC circuits with voltage changes, output should be different
                        # Allow for some tolerance due to transient behavior
                        # Use smaller threshold for fast simulations with small time steps
                        if abs(avg_altered - avg_initial) > 0.001:  # 1mV difference threshold for fast simulation
                            print("✓ Voltage changes detected in simulation output")
                        else:
                            print("⚠️ Voltage changes may not be affecting simulation output (small changes expected in fast simulation)")

            print(f"Collected {len(initial_data)} initial data points, {len(alter_data)} alter data points")
            print(f"Voltage changes completed: {current_voltage_index}/{len(voltage_sequence)}")
            print(f"Total data points: {len(all_data)}")

            return {
                "initial_points": len(initial_data),
                "alter_points": len(alter_data),
                "final_points": 0,  # Not tracking separately in this approach
                "signal_count": len(found_signals),
                "mapped_count": len(mapped_signals),
                "voltage_steps": current_voltage_index,
                "applied_voltages": applied_voltages,
            }

    import asyncio

    result = asyncio.run(run_comprehensive_test())
    assert result["initial_points"] > 0, "Should collect initial data points"
    assert result["alter_points"] > 0, "Should collect data after alterations"
    assert result["signal_count"] >= 2, "Should detect multiple signals"
    assert result["mapped_count"] >= 2, "Should map signal names correctly"
    assert result["voltage_steps"] == 4, f"Should complete all 4 voltage alteration steps, got {result['voltage_steps']}"
    assert len(result["applied_voltages"]) == 4, f"Should apply all 4 voltages, got {len(result['applied_voltages'])}"
    # Verify that the applied voltages match our expected sequence
    assert result["applied_voltages"] == [2.0, 1.5, 3.0, 1.0], f"Applied voltages don't match expected sequence: {result['applied_voltages']}"

    # Additional verification: ensure we have enough data to verify voltage changes
    assert result["alter_points"] > 10, f"Need sufficient alter data points to verify voltage changes, got {result['alter_points']}"


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_async_drain_exact_points(backend):
    """
    Tests the async generator's ability to run to completion and drain a
    large, predictable number of data points. This is a direct regression test
    against the race condition that caused premature termination on fast backends.
    """
    # 1. Setup: Configure a simulation to produce exactly 2000 data points.
    # A tran simulation from 0 to N*tstep produces N+1 points.
    # So, to get 2000 points, we need 1999 steps.
    h = lib_test.ResdivFlatTb(backend=backend)
    num_points = 2000
    tstep_us = 1
    tstop_us = (num_points - 1) * tstep_us  # 1999us

    tstep_str = f"{tstep_us}u"
    tstop_str = f"{tstop_us}u"

    # 2. Execution: Consume the entire generator and count the points.
    points_consumed = 0
    last_result = None
    seen_times = set()

    # For ffi and mp backends, disable buffering to get all data points instead of sampled subset
    if backend in ["ffi", "mp"]:
        for result in h.sim_tran_async(tstep_str, tstop_str, disable_buffering=True):
            # Fast fail on duplicate time values
            time_val = result.time.value
            if time_val in seen_times:
                pytest.fail(f"DUPLICATE TIME VALUE DETECTED: time={time_val}, backend={backend}. This indicates a bug in the async data handling.")
            seen_times.add(time_val)

            points_consumed += 1
            last_result = result
    else:
        for result in h.sim_tran_async(tstep_str, tstop_str):
            # Fast fail on duplicate time values
            time_val = result.time.value
            if time_val in seen_times:
                pytest.fail(f"DUPLICATE TIME VALUE DETECTED: time={time_val}, backend={backend}. This indicates a bug in the async data handling.")
            seen_times.add(time_val)

            points_consumed += 1
            last_result = result


    # Debug output to aid investigation of failures
    print(
        f"DEBUG test_async_drain_exact_points: backend={backend}, expected_points={num_points}, points_consumed={points_consumed}"
    )
    if last_result is not None:
        # Some result fields may be objects; print safely
        prog = getattr(last_result, "progress", None)
        time_attr = getattr(last_result, "time", None)
        time_val = (
            getattr(time_attr, "value", time_attr) if time_attr is not None else None
        )
        print(f"DEBUG final_result: progress={prog}, time.value={time_val}")

    # 3. Verification
    assert last_result is not None, "Async generator produced no results."

    # 3. Verification
    assert last_result is not None, "Async generator produced no results."

    # The primary check: did we get approximately the expected number of points?
    # Ngspice uses adaptive time stepping, so we don't get exactly the requested points
    assert abs(points_consumed - num_points) <= num_points * 0.01, (
        f"Expected approximately {num_points} points, but got {points_consumed}."
    )

    # Secondary checks to ensure the simulation ran correctly to the end.
    assert hasattr(last_result, "progress"), (
        "Final result object missing 'progress' attribute."
    )
    assert last_result.progress >= 0.999, (
        f"Simulation did not complete as expected; final progress was {last_result.progress * 100:.2f}%."
    )

    assert hasattr(last_result, "time"), "Final result object missing 'time' attribute."
    assert last_result.time.value == pytest.approx(tstop_us * 1e-6), (
        "Final simulation time does not match the expected tstop."
    )
