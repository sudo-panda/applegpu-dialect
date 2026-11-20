# End to end: Triton kernel -> TTIR -> applegpu-opt -> MSL -> metallib -> GPU.
#
# XFAIL: *
# The convert-triton-to-gpu pass does not exist yet. Remove the XFAIL line
# when this test passes.
#
# Python -> TTIR, in generic form without locations:
# RUN: %triton_python %S/../../runners/triton_to_ttir.py %s add_kernel \
# RUN:   --signature 'x_ptr=*fp32,y_ptr=*fp32,out_ptr=*fp32,n=i32' \
# RUN:   --constexpr BLOCK=128 \
# RUN:   | %triton_opt --mlir-print-op-generic > %t.ttir
#
# TTIR -> MSL. One Triton program is one threadgroup of BLOCK threads, one
# element per thread.
# RUN: applegpu-opt --allow-unregistered-dialect --convert-triton-to-gpu %t.ttir \
# RUN:   | applegpu-translate --mlir-to-msl > %t.metal
#
# MSL -> metallib:
# RUN: xcrun -sdk macosx metal -c %t.metal -o %t.air
# RUN: xcrun -sdk macosx metallib %t.air -o %t.metallib
#
# Run 8 programs of 128 elements over n = 1000, so the last program is only
# partly masked in. The output starts as -1 to show the masked tail is not
# written. Buffers follow the kernel's argument order.
# RUN: %run_grid %t.metallib add_kernel 8 128 \
# RUN:   f32:1024:iota 'f32:1024:iota*2' f32:1024:fill=-1 i32:1000 > %t.out
#
# Compare every element with the reference printed by this file:
# RUN: %triton_python %s > %t.expected
# RUN: diff %t.expected %t.out

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def reference(size=1024, n=1000):
    """Buffers after the run, printed in the runner's format."""
    x = [float(i) for i in range(size)]
    y = [2.0 * i for i in range(size)]
    out = [x[i] + y[i] if i < n else -1.0 for i in range(size)]
    for index, buffer in enumerate((x, y, out)):
        for j, value in enumerate(buffer):
            print(f"arg{index}[{j}] = {value}")


if __name__ == "__main__":
    reference()
