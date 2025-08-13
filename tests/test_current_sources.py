# SPDX-FileCopyrightText: 2024 ORDeC contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ordec.lib.base import (
    Idc,
    PieceWiseLinearCurrentSource,
    PulseCurrentSource,
    SinusoidalCurrentSource,
)
from ordec.core import Rational as R


def test_idc_current_source():
    """
    Tests the Idc current source.
    """
    i = Idc(dc=R("1u"))
    assert i.dc == R("1u")


def test_pwl_current_source():
    """
    Tests the PieceWiseLinearCurrentSource.
    """
    i = PieceWiseLinearCurrentSource(I=[(0, 0), ("1n", "1u")])
    assert i.I == [(0, 0), ("1n", "1u")]


def test_pulse_current_source():
    """
    Tests the PulseCurrentSource.
    """
    i = PulseCurrentSource(
        initial_value=R(0),
        pulsed_value=R("1u"),
        pulse_width=R("1m"),
        period=R("2m"),
    )
    assert i.initial_value == R(0)
    assert i.pulsed_value == R("1u")
    assert i.pulse_width == R("1m")
    assert i.period == R("2m")


def test_sinusoidal_current_source():
    """
    Tests the SinusoidalCurrentSource.
    """
    i = SinusoidalCurrentSource(amplitude=R("1u"), frequency=R(1000))
    assert i.amplitude == R("1u")
    assert i.frequency == R(1000)
