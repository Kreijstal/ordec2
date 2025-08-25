import subprocess
import tempfile
from pathlib import Path

netlist = """
* RC Filter
V1 inp 0 dc 0 ac 1
R1 inp out 1k
C1 out 0 1n
.end
"""

with tempfile.TemporaryDirectory() as cwd_str:
    cwd = Path(cwd_str)
    netlist_fn = cwd / 'netlist.sp'
    netlist_fn.write_text(netlist)
    data_fn = cwd / 'ac_data.txt'

    commands = (
        f"source {netlist_fn}\n"
        "ac dec 10 1 1G\n"
        "display\n"
        f"wrdata {data_fn} frequency v(out)\n"
        "exit\n"
    )

    try:
        p = subprocess.run(
            ['ngspice', '-p'],
            input=commands,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=cwd_str
        )
        print("--- NGSPICE STDOUT ---")
        print(p.stdout)
        print("--- END NGSPICE STDOUT ---")
        if p.stderr:
            print("--- NGSPICE STDERR ---")
            print(p.stderr)
            print("--- END NGSPICE STDERR ---")

        if data_fn.exists():
            print(f"\n--- CONTENTS OF {data_fn} ---")
            print(data_fn.read_text())
            print(f"--- END CONTENTS OF {data_fn} ---")
        else:
            print(f"\n--- {data_fn} was not created. ---")

    except subprocess.TimeoutExpired as e:
        print("ngspice process timed out.")
        if e.stdout:
            print("--- NGSPICE STDOUT (timeout) ---")
            print(e.stdout)
            print("--- END NGSPICE STDOUT (timeout) ---")
        if e.stderr:
            print("--- NGSPICE STDERR (timeout) ---")
            print(e.stderr)
            print("--- END NGSPICE STDERR (timeout) ---")
