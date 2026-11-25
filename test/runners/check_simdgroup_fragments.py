"""Compare simdgroup_matrix fragments measured on the GPU with the plugin's layouts.

Usage:
  check_simdgroup_fragments.py --tensor-layout PATH --plugin PATH TYPE=RUN_GRID_OUTPUT...

Each RUN_GRID_OUTPUT is what run-grid printed for one kernel of
test/TritonPlugin/simdgroup_mma_hardware.metal: buffer 0 holds each lane's two
elements of a loaded matrix of 0..63, buffer 1 the positions the kernel
assumed, buffer 2 the 8x8 product of operands placed at those positions.

The expected positions come from the plugin itself, through Triton's
triton-tensor-layout, for the result layout and both operand layouts. Prints
one line of counts per type and exits with an error on any mismatch.
"""

import argparse
import os
import re
import subprocess
import sys

MMA = "#apple_tt.simdgroup_mma<{warpsPerCTA = [1, 1]}>"
LAYOUTS = {
    "result": (MMA, "tensor<8x8xf32>"),
    "A": (f"#ttg.dot_op<{{opIdx = 0, parent = {MMA}}}>", "tensor<8x8xf16>"),
    "B": (f"#ttg.dot_op<{{opIdx = 1, parent = {MMA}}}>", "tensor<8x8xf16>"),
}


def plugin_layout(tool, plugin, layout, tensor):
    """(lane, register) -> (row, col), as the plugin defines the layout."""
    env = dict(os.environ, TRITON_PLUGIN_PATHS=plugin)
    out = subprocess.run([tool, "-l", layout, "-t", tensor], env=env,
                         capture_output=True, text=True, check=True).stdout
    rows = [line.strip().strip("[]") for line in out.splitlines() if line.strip().startswith("[")]
    where = {}
    for r, line in enumerate(rows):
        for c, cell in enumerate(line.split(",")):
            for holder in cell.split("|"):
                m = re.fullmatch(r"T(\d+):(\d+)", holder.strip())
                if not m:
                    sys.exit(f"error: cannot read '{holder}' in the output of {tool}")
                where[(int(m.group(1)), int(m.group(2)))] = (r, c)
    if len(where) != 64:
        sys.exit(f"error: expected 64 (lane, register) pairs for {layout}, got {len(where)}")
    return where


def run_grid_buffers(path):
    bufs = {}
    for line in open(path):
        m = re.fullmatch(r"arg(\d+)\[(\d+)\] = (\S+)\s*", line)
        if m:
            bufs.setdefault(int(m.group(1)), {})[int(m.group(2))] = float(m.group(3))
    return [[bufs[i][j] for j in range(64)] for i in range(3)]


# The same values as aValue and bValue in the kernel.
def a_value(m, k):
    return float((m + 2 * k) % 5 - 2)


def b_value(k, n):
    return float((3 * k + n) % 7 - 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tensor-layout", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("outputs", nargs="+", help="TYPE=run-grid output file")
    args = parser.parse_args()
    expected = {name: plugin_layout(args.tensor_layout, args.plugin, *lt) for name, lt in LAYOUTS.items()}
    cpu = [sum(a_value(m, k) * b_value(k, n) for k in range(8)) for m in range(8) for n in range(8)]

    failed = False
    for spec in args.outputs:
        dtype, path = spec.split("=", 1)
        loaded, positions, product = run_grid_buffers(path)
        hw = {(lane, r): divmod(int(loaded[lane * 2 + r]), 8) for lane in range(32) for r in range(2)}
        assumed = {(lane, r): divmod(int(positions[lane * 2 + r]), 8) for lane in range(32) for r in range(2)}
        checks = {
            "loaded": {k: hw[k] == expected["result"][k] for k in hw},
            "positions": {k: assumed[k] == expected["result"][k] for k in assumed},
            "operand A": {k: hw[k] == expected["A"][k] for k in hw},
            "operand B": {k: hw[k] == expected["B"][k] for k in hw},
            "product": {i: product[i] == cpu[i] for i in range(64)},
        }
        line = ", ".join(f"{name} {sum(v.values())}/64" for name, v in checks.items())
        print(f"{dtype}: {line}")
        for name, v in checks.items():
            bad = [k for k, ok in v.items() if not ok]
            if bad:
                failed = True
                k = bad[0]
                if name == "product":
                    detail = f"element ({k // 8}, {k % 8}) is {product[k]}, expected {cpu[k]}"
                else:
                    got = hw[k] if name != "positions" else assumed[k]
                    ref = expected["result" if name in ("loaded", "positions") else name.split()[-1]][k]
                    detail = f"lane {k[0]}, register {k[1]}: GPU/kernel position {got}, plugin {ref}"
                print(f"error: {dtype} {name}: {len(bad)} mismatches, first: {detail}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
