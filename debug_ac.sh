#!/bin/bash
python generate_netlist.py > rc_filter.sp

ngspice -p > ngspice_out.txt <<EOF
source rc_filter.sp
ac dec 10 1 1G
display
exit
EOF
cat ngspice_out.txt
