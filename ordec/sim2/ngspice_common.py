# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

import re
from collections import namedtuple
from enum import Enum
from dataclasses import dataclass
from typing import Dict

NgspiceValue = namedtuple('NgspiceValue', ['type', 'name', 'subname', 'value'])

class SignalKind(Enum):
    TIME = 1
    VOLTAGE = 3
    CURRENT = 4
    OTHER = 99

    @classmethod
    def from_vtype(cls, vtype: int):
        """Map ngspice VectorInfo.v_type integer codes to SignalKind.

        Common observed mapping (may vary across ngspice versions):
        - 1 : TIME
        - 3 : VOLTAGE
        - 4 : CURRENT

        Falls back to OTHER for unknown values.
        """
        if vtype == 1:
            return cls.TIME
        elif vtype == 3:
            return cls.VOLTAGE
        elif vtype == 4:
            return cls.CURRENT
        else:
            return cls.OTHER

@dataclass
class SignalArray:
    kind: SignalKind
    values: list

class NgspiceError(Exception):
    pass

class NgspiceFatalError(NgspiceError):
    pass

class NgspiceConfigError(NgspiceError):
    """Raised when backend configuration fails."""
    pass

class NgspiceTable:
    def __init__(self, name):
        self.name = name
        self.headers = []
        self.data = []

class NgspiceTransientResult:
    def __init__(self):
        self.time: list = []
        # Map signal name -> SignalArray
        self.signals: Dict[str, SignalArray] = {}
        self.tables: list = []
        # legacy buckets (kept for compatibility with higher-level code)
        self.voltages: Dict[str, list] = {}
        self.currents: Dict[str, Dict[str, list]] = {}
        self.branches: Dict[str, list] = {}

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
            # Filter out any rows that might contain header strings
            valid_time_data = []
            for row in table.data:
                if len(row) > time_idx:
                    try:
                        time_val = float(row[time_idx])
                        valid_time_data.append(time_val)
                    except (ValueError, TypeError):
                        # Skip rows that can't be converted to float (likely headers)
                        continue
            self.time = valid_time_data

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
                    except (ValueError, TypeError, IndexError):
                        # Skip rows that can't be converted to float or are malformed
                        continue

            # Only add signal if we got valid data
            if signal_data:
                # Wrap the array in SignalArray with a best-effort kind classification
                kind = self._guess_signal_kind(signal_name)
                self.signals[signal_name] = SignalArray(kind=kind, values=signal_data)
                # Categorize signals for easier access (populate voltages/currents/branches)
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

    def _guess_signal_kind(self, signal_name) -> SignalKind:
        """Best-effort heuristic to classify a signal name into a SignalKind."""
        if not signal_name:
            return SignalKind.OTHER
        name = signal_name.lower()
        if name in ('time', 'index'):
            return SignalKind.TIME
        if signal_name.startswith('@') and '[' in signal_name:
            return SignalKind.CURRENT
        if signal_name.endswith('#branch'):
            return SignalKind.CURRENT
        # Fallback: treat as node voltage
        return SignalKind.VOLTAGE

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

class NgspiceAcResult:
    def __init__(self):
        self.freq = []
        self.voltages = {}
        self.currents = {}
        self.branches = {}

    def _categorize_signal(self, signal_name, signal_data):
        """Categorize signals into voltages, currents, and branches."""
        if signal_name.startswith('@') and '[' in signal_name:
            device_part = signal_name.split('[')[0][1:]
            current_type = signal_name.split('[')[1].rstrip(']')
            if device_part not in self.currents:
                self.currents[device_part] = {}
            self.currents[device_part][current_type] = signal_data
        elif signal_name.endswith('#branch'):
            branch_name = signal_name.replace('#branch', '')
            self.branches[branch_name] = signal_data
        else:
            self.voltages[signal_name] = signal_data

def check_errors(ngspice_out):
    """Helper function to raise NgspiceError in Python from "Error: ..."
    messages in Ngspice's output."""
    first_error_msg = None
    has_fatal_indicator = False

    for line in ngspice_out.split('\n'):
        if "no such vector" in line:
            # This error can occur when a simulation (like 'op') is run that doesn't
            # produce any plot output. It's not a fatal error, so we ignore it.
            continue
        # Handle both "Error: ..." and "stderr Error: ..." formats
        m = re.match(r"(?:stderr )?Error:\s*(.*)", line)
        if m and first_error_msg is None:
            first_error_msg = "Error: " + m.group(1)

        # Check if this line indicates a fatal error
        if "cannot recover" in line or "awaits to be reset" in line:
            has_fatal_indicator = True

    # Raise appropriate exception if we found an error
    if first_error_msg:
        if has_fatal_indicator:
            raise NgspiceFatalError(first_error_msg)
        else:
            raise NgspiceError(first_error_msg)
