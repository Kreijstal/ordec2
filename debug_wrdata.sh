#!/bin/bash
python -c '
from ordec.lib import test as lib_test
from ordec.sim2.ngspice import Netlister
from ordec.core import R
tb = lib_test.RcFilterTb(r=R(1e3), c=R(1e-9))
netlister = Netlister()
netlister.netlist_hier(tb.schematic)
netlist_str = netlister.out()
control_block = (
    "\n.control\n"
    "ac dec 10 1 1G\n"
    "wrdata ac_data.txt v(out) v(inp)\n"
    ".endc\n"
)
netlist_with_ac = netlist_str.replace(".end", control_block + ".end")
print(netlist_with_ac)
' > rc_filter_wrdata.sp

ngspice -b rc_filter_wrdata.sp
cat ac_data.txt
