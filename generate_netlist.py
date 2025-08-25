from ordec.lib import test as lib_test
from ordec.sim2.ngspice import Netlister
from ordec.core import R

tb = lib_test.RcFilterTb(r=R(1e3), c=R(1e-9))
netlister = Netlister()
netlister.netlist_hier(tb.schematic)
netlist_str = netlister.out()

print(netlist_str)
