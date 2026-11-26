# Triton loads pass plugins. This needs Triton built with TRITON_EXT_ENABLED=1
# (utils/setup.sh does this); otherwise libtriton exports no symbols and Triton
# does not build its example plugin. This example pass renames every function
# to "foo".
#
# RUN: %triton_python %s %t.mlir | FileCheck %s
# CHECK: plugin passes: add_plugin
# CHECK: tt.func @foo()
# CHECK-NOT: @bar
import pathlib
import sys

import triton
from triton._C.libtriton import ir, passes  # pyright: ignore[reportMissingImports]

# CMake names it .dylib on macOS and .so on Linux.
PLUGIN = pathlib.Path(triton.__path__[0]) / "plugins" / (
    "libTritonPluginsTestLib" + (".dylib" if sys.platform == "darwin" else ".so"))
if not PLUGIN.is_file():
    sys.exit(f"error: {PLUGIN} not found; build Triton with TRITON_EXT_ENABLED=1 "
             "(rerun utils/setup.sh)")

passes.plugin.extend_with(str(PLUGIN))
print("plugin passes:", " ".join(n for n in dir(passes.plugin) if n.startswith("add_")))

src = pathlib.Path(sys.argv[1])
src.write_text("""
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @bar() {
    tt.return
  }
}
""")
context = ir.context()
ir.load_dialects(context)
module = ir.parse_mlir_module(str(src), context)
pm = ir.pass_manager(context)
passes.plugin.add_plugin(pm)
pm.run(module, "plugin")
print(module)
