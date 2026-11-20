// Run one kernel from a .metallib over a 1-D grid and print its buffers.
//
// Usage:
//   run-grid <file.metallib> <kernel> <threadgroups> <threads-per-group> <arg>...
//
// Each <arg> binds the next [[buffer(i)]] slot, in order:
//   f32:<count>:iota        float buffer holding 0, 1, 2, ...
//   f32:<count>:iota*<k>    float buffer holding 0, k, 2k, ...
//   f32:<count>:fill=<v>    float buffer with every element set to v
//   i32:<value>             32-bit integer scalar, passed by value (setBytes)
//
// After the run, every float buffer is printed to stdout, one element per
// line, as "arg<i>[<j>] = <value>". Values use the shortest form that reads
// back as the same float, with ".0" added to whole numbers (3.0, -1.0), which
// is also how Python prints them. The GPU name goes to stderr, so stdout can
// be compared exactly with a reference.
//
// Built by test/CMakeLists.txt on macOS and used in tests as %run_grid.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "error: %s\n", message.c_str());
  std::exit(1);
}

bool parseInt(const std::string &text, long long &value) {
  auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  return ec == std::errc() && end == text.data() + text.size();
}

bool parseFloat(const std::string &text, float &value) {
  auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  return ec == std::errc() && end == text.data() + text.size();
}

std::vector<float> initialValues(long long count, const std::string &init) {
  std::vector<float> values(count);
  float k;
  if (init == "iota") {
    for (long long i = 0; i < count; ++i)
      values[i] = float(i);
  } else if (init.rfind("iota*", 0) == 0 && parseFloat(init.substr(5), k)) {
    for (long long i = 0; i < count; ++i)
      values[i] = float(i) * k;
  } else if (init.rfind("fill=", 0) == 0 && parseFloat(init.substr(5), k)) {
    values.assign(count, k);
  } else {
    fail("unknown buffer initializer '" + init + "'");
  }
  return values;
}

std::string formatFloat(float value) {
  char buffer[64];
  auto [end, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  std::string text(buffer, end);
  if (text.find_first_of(".eEna") == std::string::npos)
    text += ".0";
  return text;
}

struct Binding {
  bool isScalar;
  id<MTLBuffer> buffer; // float buffer
  long long count;      // float buffer
  int32_t scalar;       // i32 scalar
};

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 5)
      fail("usage: run-grid <file.metallib> <kernel> <threadgroups> "
           "<threads-per-group> <arg>...");
    long long groups, threads;
    if (!parseInt(argv[3], groups) || groups <= 0 || !parseInt(argv[4], threads) ||
        threads <= 0)
      fail("threadgroups and threads-per-group must be positive integers");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      fail("no Metal GPU found");
    std::fprintf(stderr, "GPU: %s\n", device.name.UTF8String);

    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:@(argv[1])];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library)
      fail(std::string("cannot load library: ") + error.localizedDescription.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:@(argv[2])];
    if (!function)
      fail(std::string("kernel '") + argv[2] + "' not found");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline)
      fail(std::string("cannot create pipeline: ") + error.localizedDescription.UTF8String);
    if (threads > (long long)pipeline.maxTotalThreadsPerThreadgroup)
      fail(std::to_string(threads) + " threads per group exceeds the limit of " +
           std::to_string(pipeline.maxTotalThreadsPerThreadgroup));

    std::vector<Binding> bindings;
    for (int a = 5; a < argc; ++a) {
      std::string spec = argv[a];
      long long value;
      if (spec.rfind("f32:", 0) == 0) {
        size_t colon = spec.find(':', 4);
        if (colon == std::string::npos || !parseInt(spec.substr(4, colon - 4), value) ||
            value <= 0)
          fail("bad float buffer '" + spec + "'");
        std::vector<float> values = initialValues(value, spec.substr(colon + 1));
        id<MTLBuffer> buffer = [device newBufferWithBytes:values.data()
                                                   length:values.size() * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        bindings.push_back({false, buffer, value, 0});
      } else if (spec.rfind("i32:", 0) == 0) {
        if (!parseInt(spec.substr(4), value) || value < INT32_MIN || value > INT32_MAX)
          fail("bad integer '" + spec + "'");
        bindings.push_back({true, nil, 0, int32_t(value)});
      } else {
        fail("bad argument '" + spec + "'");
      }
    }

    id<MTLCommandBuffer> commands = [[device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (size_t i = 0; i < bindings.size(); ++i) {
      if (bindings[i].isScalar)
        [encoder setBytes:&bindings[i].scalar length:sizeof(int32_t) atIndex:i];
      else
        [encoder setBuffer:bindings[i].buffer offset:0 atIndex:i];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.error)
      fail(commands.error.localizedDescription.UTF8String);

    for (size_t i = 0; i < bindings.size(); ++i) {
      if (bindings[i].isScalar)
        continue;
      const float *values = static_cast<const float *>(bindings[i].buffer.contents);
      for (long long j = 0; j < bindings[i].count; ++j)
        std::printf("arg%zu[%lld] = %s\n", i, j, formatFloat(values[j]).c_str());
    }
  }
  return 0;
}
