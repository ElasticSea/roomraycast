//
//  RayTracingAccelerationBuilder.swift
//  roomraycast
//

import Metal
import MetalKit
import ModelIO

enum RayTracingAccelerationBuilderError: LocalizedError {
    case unsupported
    case commandQueueUnavailable
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case allocationFailed
    case invalidMesh
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .unsupported: "This Metal device does not support ray tracing."
        case .commandQueueUnavailable: "Could not create the ray-tracing command queue."
        case .commandBufferUnavailable: "Could not create a ray-tracing command buffer."
        case .commandEncoderUnavailable: "Could not create an acceleration-structure encoder."
        case .allocationFailed: "Could not allocate an acceleration structure."
        case .invalidMesh: "A ray-tracing mesh has no usable triangle data."
        case .commandFailed(let message): "Acceleration-structure build failed: \(message)"
        }
    }
}

struct RayTracingIndexedSubmesh: @unchecked Sendable {
    let indexBuffer: MTLBuffer
    let indexBufferOffset: Int
    let indexType: MTLIndexType
    let indexCount: Int
}

struct RayTracingTriangleGeometrySource: @unchecked Sendable {
    let id: RayTracingGeometryID
    let vertexBuffer: MTLBuffer
    let vertexBufferOffset: Int
    let vertexStride: Int
    let submeshes: [RayTracingIndexedSubmesh]

    init(id: RayTracingGeometryID, mesh: MTKMesh) throws {
        guard let positionLayout = mesh.vertexDescriptor.layouts[BufferIndex.meshPositions.rawValue]
                as? MDLVertexBufferLayout,
              positionLayout.stride > 0,
              mesh.vertexBuffers.indices.contains(BufferIndex.meshPositions.rawValue),
              !mesh.submeshes.isEmpty else {
            throw RayTracingAccelerationBuilderError.invalidMesh
        }

        let positionBuffer = mesh.vertexBuffers[BufferIndex.meshPositions.rawValue]
        self.id = id
        self.vertexBuffer = positionBuffer.buffer
        self.vertexBufferOffset = positionBuffer.offset
        self.vertexStride = positionLayout.stride
        self.submeshes = mesh.submeshes.map { submesh in
            RayTracingIndexedSubmesh(indexBuffer: submesh.indexBuffer.buffer,
                                     indexBufferOffset: submesh.indexBuffer.offset,
                                     indexType: submesh.indexType,
                                     indexCount: submesh.indexCount)
        }
    }
}

final class RayTracingAccelerationBuilder {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init(device: MTLDevice) throws {
        guard device.supportsRaytracing else {
            throw RayTracingAccelerationBuilderError.unsupported
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw RayTracingAccelerationBuilderError.commandQueueUnavailable
        }
        self.device = device
        self.commandQueue = commandQueue
        self.commandQueue.label = "Ray Tracing Acceleration Builds"
    }

    func buildBottomLevelStructures(
        for sources: [RayTracingTriangleGeometrySource]
    ) throws -> [RayTracingGeometryID: MTLAccelerationStructure] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw RayTracingAccelerationBuilderError.commandBufferUnavailable
        }
        guard let encoder = commandBuffer.makeAccelerationStructureCommandEncoder() else {
            throw RayTracingAccelerationBuilderError.commandEncoderUnavailable
        }

        commandBuffer.label = "Build Ray Tracing BLAS"
        encoder.label = "Build Static Triangle BLAS"
        var structures: [RayTracingGeometryID: MTLAccelerationStructure] = [:]
        var scratchBuffers: [MTLBuffer] = []

        for source in sources {
            let geometryDescriptors = source.submeshes.map { submesh in
                let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
                geometry.vertexBuffer = source.vertexBuffer
                geometry.vertexBufferOffset = source.vertexBufferOffset
                geometry.vertexStride = source.vertexStride
                geometry.vertexFormat = .float3
                geometry.indexBuffer = submesh.indexBuffer
                geometry.indexBufferOffset = submesh.indexBufferOffset
                geometry.indexType = submesh.indexType
                geometry.triangleCount = submesh.indexCount / 3
                geometry.opaque = true
                return geometry
            }

            let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
            descriptor.geometryDescriptors = geometryDescriptors
            descriptor.usage = [.preferFastTrace]
            let sizes = device.accelerationStructureSizes(descriptor: descriptor)
            guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
                  let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize,
                                                        options: .storageModePrivate) else {
                throw RayTracingAccelerationBuilderError.allocationFailed
            }

            structure.label = "BLAS \(source.id)"
            encoder.build(accelerationStructure: structure,
                          descriptor: descriptor,
                          scratchBuffer: scratchBuffer,
                          scratchBufferOffset: 0)
            structures[source.id] = structure
            scratchBuffers.append(scratchBuffer)
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            throw RayTracingAccelerationBuilderError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Unknown Metal error")
        }
        _ = scratchBuffers
        return structures
    }
}
