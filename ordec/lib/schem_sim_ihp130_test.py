# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

from ordec import (
    Cell,
    Vec2R,
    Rect4R,
    Rational as R,
    SchemInstance,
    Net,
    Schematic,
    generate,
)
from ordec.lib import (
    Vdc,
    Res,
    Gnd,
)
from ordec import helpers
from ordec.lib.ihp130 import Nmos, Pmos, Inv


class TBSimpleInv(Cell):
    """Simple inverter testbench for IHP130"""
    @generate
    def schematic(self):
        s = Schematic(cell=self)

        s.vdd = Net()
        s.gnd = Net()
        s.input_node = Net()
        s.output_node = Net()

        # Simple DC voltage source
        vdc_inst = Vdc(dc=R("1.8")).symbol
        s.vdc = SchemInstance(
            vdc_inst.portmap(p=s.vdd, m=s.gnd),
            pos=Vec2R(2, 8),
        )

        # IHP130 Inverter
        inv = Inv().symbol
        s.inv = SchemInstance(
            inv.portmap(
                a=s.input_node,
                vdd=s.vdd,
                vss=s.gnd,
                y=s.output_node,
            ),
            pos=Vec2R(8, 5),
        )

        # Ground reference
        gnd_inst = Gnd().symbol
        s.gnd_inst = SchemInstance(
            gnd_inst.portmap(p=s.gnd),
            pos=Vec2R(12, 12),
        )
        
        s.outline = Rect4R(lx=0, ly=2, ux=20, uy=14)

        helpers.schem_check(s, add_conn_points=True, add_terminal_taps=True)
        return s