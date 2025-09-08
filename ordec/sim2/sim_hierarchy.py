# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from ..core import *
from ..core.schema import SimType
from .ngspice import Ngspice, Netlister

def build_hier_symbol(simhier, symbol):
    simhier.schematic = symbol
    for pin in symbol.all(Pin):
        # TODO: implement hierarchical construction within schematic
        setattr(simhier, pin.full_path_str(), SimNet(eref=pin))

def build_hier_schematic(simhier, schematic):
    simhier.schematic = schematic
    for net in schematic.all(Net):
        # TODO: implement hierarchical construction within schematic
        setattr(simhier, net.full_path_str(), SimNet(eref=net))

    for inst in schematic.all(SchemInstance):
        # TODO: implement hierarchical construction
        setattr(simhier, inst.full_path_str(), SimInstance(eref=inst))
        subnode = getattr(simhier, inst.full_path_str())
        try:
            subschematic = inst.symbol.cell.schematic
        except AttributeError:
            build_hier_symbol(subnode, inst.symbol)
        else:
            build_hier_schematic(subnode, subschematic)

class HighlevelSim:
    def __init__(self, top: Schematic, simhier: SimHierarchy, enable_savecurrents: bool = True, backend: str = 'subprocess'):
        self.top = top
        self.simhier = simhier
        self.enable_savecurrents = enable_savecurrents
        self.backend = backend
        self.cm = None
        self.sim = None

    def __enter__(self):
        self.netlister = Netlister(enable_savecurrents=self.enable_savecurrents)
        self.netlister.netlist_hier(self.top)

        #self.simhier.schematic = self.top
        build_hier_schematic(self.simhier, self.top)
        self.str_to_simobj = {}
        for sn in self.simhier.all(SimNet):
            name = self.netlister.name_hier_simobj(sn)
            self.str_to_simobj[name] = sn

        for sn in self.simhier.all(SimInstance):
            name = self.netlister.name_hier_simobj(sn)
            self.str_to_simobj[name] = sn

        # Collect simulation setup hooks from netlister
        self.sim_setup_hooks = []
        if hasattr(self.netlister, '_sim_setup_hooks'):
            self.sim_setup_hooks = list(self.netlister._sim_setup_hooks)

        self.cm = Ngspice.launch(debug=False, backend=self.backend)
        self.sim = self.cm.__enter__()

        for hook in self.sim_setup_hooks:
            hook(self.sim)

        self.sim.load_netlist(self.netlister.out())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cm:
            self.cm.__exit__(exc_type, exc_val, exc_tb)

    def op(self):
        self.simhier.sim_type = SimType.DC
        for vtype, name, subname, value in self.sim.op():
            if vtype == 'voltage':
                try:
                    simnet = self.str_to_simobj[name]
                    simnet.dc_voltage = value
                except KeyError:
                    # Silently ignore internal subcircuit voltages that can't be mapped to hierarchy
                    # These are typically internal nodes within subcircuits (e.g. device body nodes)
                    continue
            elif vtype == 'current':
                if subname not in ('id', 'branch', 'i'):
                    continue
                try:
                    siminstance = self.str_to_simobj[name]
                    siminstance.dc_current = value
                except KeyError:
                    # Silently ignore internal subcircuit device currents that can't be mapped to hierarchy
                    # These are typically internal devices within subcircuits (e.g. MOSFET models)
                    continue

    def tran(self, tstep, tstop):
        self.simhier.sim_type = SimType.TRAN
        data = self.sim.tran(tstep, tstop)
        self.simhier.time = tuple(data.time)
        for name, value in data.voltages.items():
            try:
                simnet = self.str_to_simobj[name]
                simnet.trans_voltage = tuple(value)
            except KeyError:
                continue
        for name, value in data.currents.items():
            try:
                siminstance = self.str_to_simobj[name]
                siminstance.trans_current = tuple(value)
            except KeyError:
                continue

    def ac(self, *args, wrdata_file: Optional[str] = None):
        self.simhier.sim_type = SimType.AC
        data = self.sim.ac(*args, wrdata_file=wrdata_file)
        self.simhier.freq = tuple(data.freq)
        for name, value in data.voltages.items():
            try:
                simnet = self.str_to_simobj[name]
                simnet.ac_voltage = tuple([(c.real, c.imag) for c in value])
            except KeyError:
                continue

        for name, currents in data.currents.items():
            try:
                siminstance = self.str_to_simobj[name]
                if 'id' in currents:
                    main_current = currents['id']
                elif 'branch' in currents:
                    main_current = currents['branch']
                elif currents:
                    main_current = next(iter(currents.values()))
                else:
                    continue # No currents available for this instance

                siminstance.ac_current = tuple([(c.real, c.imag) for c in main_current])
            except (KeyError, StopIteration):
                continue

    def alter_device(self, device_ref, **parameters):
        """Alter device parameters in the simulation hierarchy.
        
        Args:
            device_ref: ORDeC SimInstance/SimNet object or SPICE device name string
            **parameters: Device parameters to change (e.g., w=10u, l=180n)
        """
        # Determine device name from reference
        if isinstance(device_ref, (SimInstance, SimNet)):
            # Use netlister to get hierarchical SPICE name from ORDeC object
            device_name = self.netlister.name_hier_simobj(device_ref)
        else:
            # Assume it's a SPICE device name string (support both formats)
            # For hierarchical strings like "top.X1.M1", we could lookup in str_to_simobj
            # but for simplicity, assume it matches the netlist naming
            if '.' in device_ref:
                # Try to find Sim object by hierarchical path and get name
                try:
                    sim_obj = self.str_to_simobj[device_ref]
                    device_name = self.netlister.name_hier_simobj(sim_obj)
                except KeyError:
                    # Fallback to the string as-is if not found in mapping
                    device_name = device_ref
            else:
                device_name = device_ref

        # Alter the device parameters
        self.sim.alter_device(device_name, **parameters)

    def tran_async(self, *args, **kwargs):
        yield from self.sim.tran_async(*args, **kwargs)

    def is_running(self) -> bool:
        return self.sim.is_running()

    def stop_simulation(self):
        self.sim.stop_simulation()

    def bg_tran(self, *args):
        self.sim.command(f"bg_tran {' '.join(args)}")

    def bg_halt(self):
        self.sim.command("bg_halt")

    def bg_run(self):
        self.sim.command("bg_run")

    def get_async_data(self, timeout=0.1):
        return self.sim.get_async_data(timeout=timeout)
