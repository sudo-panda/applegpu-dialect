import Foundation
import Metal

let args = CommandLine.arguments
guard args.count == 3 else {
    print("usage: run_kernel <file.metallib> <kernel-name>")
    exit(2)
}
let path = args[1]
let kernelName = args[2]

guard let device = MTLCreateSystemDefaultDevice() else {
    print("ERROR: no Metal GPU found")
    exit(1)
}
print("GPU: \(device.name)")

let library = try device.makeLibrary(URL: URL(fileURLWithPath: path))
guard let function = library.makeFunction(name: kernelName) else {
    print("ERROR: kernel '\(kernelName)' not found")
    exit(1)
}
let pipeline = try device.makeComputePipelineState(function: function)

let count = 32
let input: [Float] = (1...count).map { Float($0) }
let bytes = count * MemoryLayout<Float>.stride
let inBuf = device.makeBuffer(bytes: input, length: bytes, options: .storageModeShared)!
let outBuf = device.makeBuffer(length: bytes, options: .storageModeShared)!

let commands = device.makeCommandQueue()!.makeCommandBuffer()!
let encoder = commands.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(inBuf, offset: 0, index: 0)
encoder.setBuffer(outBuf, offset: 0, index: 1)
encoder.dispatchThreadgroups(
    MTLSize(width: 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
encoder.endEncoding()
commands.commit()
commands.waitUntilCompleted()
if let error = commands.error {
    print("ERROR: \(error)")
    exit(1)
}

let out = outBuf.contents().bindMemory(to: Float.self, capacity: count)
for i in 0..<count {
    print("out[\(i)] = \(out[i])")
}