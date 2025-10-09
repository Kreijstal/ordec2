# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from ordec.lib import test as lib_test


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_consecutive_async_simulations_with_early_termination(backend):
    """Test that consecutive async simulations work correctly even when the first one is terminated early.
    
    This is a regression test for the defensive workaround issue where a relay thread
    from a previous simulation would still be alive when starting a new simulation.
    """
    h = lib_test.ResdivFlatTb(backend=backend)
    
    # First simulation - terminate early
    data_count_1 = 0
    for result in h.sim_tran_async("0.05u", "10u"):
        data_count_1 += 1
        if data_count_1 >= 5:
            break
    
    assert data_count_1 >= 1
    assert data_count_1 <= 5
    
    # Second simulation - should start cleanly without any relay thread conflicts
    # This should NOT hang or raise an error
    data_count_2 = 0
    for result in h.sim_tran_async("0.05u", "10u"):
        data_count_2 += 1
        if data_count_2 >= 5:
            break
    
    assert data_count_2 >= 1
    assert data_count_2 <= 5


@pytest.mark.libngspice
@pytest.mark.parametrize("backend", ["ffi", "mp"])
def test_multiple_consecutive_async_simulations(backend):
    """Test multiple consecutive async simulations in a loop."""
    h = lib_test.ResdivFlatTb(backend=backend)
    
    for iteration in range(3):
        data_count = 0
        for result in h.sim_tran_async("0.05u", "5u"):
            data_count += 1
            if data_count >= 3:
                break
        
        assert data_count >= 1, f"Iteration {iteration}: No data received"
        assert data_count <= 3, f"Iteration {iteration}: Too much data received"
