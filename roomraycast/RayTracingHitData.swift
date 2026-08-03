//
//  RayTracingHitData.swift
//  roomraycast
//

import Foundation
import Metal
import ModelIO
import simd

struct RayTracingPackedVertex {
    var position: SIMD4<Float>
    var normal: SIMD4<Float>
    var texCoordAndPadding: SIMD4<Float>
}

struct RayTracingGeometryMetadata {
    var vertexOffset: UInt32
    var indexOffset: UInt32
    var textureIndex: UInt32
    var triangleCount: UInt32
}

struct RayTracingInstanceMetadata {
    var geometryOffset: UInt32
    var geometryCount: UInt32
    var materialKind: UInt32
    var padding: UInt32 = 0
}

struct RayTracingObjectMetadata {
    var objectToWorld: matrix_float4x4
    var worldToObject: matrix_float4x4
    var sphereCenterAndRadius: SIMD4<Float>
}

struct RayTracingCPUHitData {
    var vertices: [RayTracingPackedVertex]
    var indices: [UInt32]
    var geometries: [RayTracingGeometryMetadata]
    var instances: [RayTracingInstanceMetadata]

    static func make(roomMeshes: [MDLMesh],
                     reflectiveObjectCapacity: Int) throws -> RayTracingCPUHitData {
        var vertices: [RayTracingPackedVertex] = []
        var indices: [UInt32] = []
        var geometries: [RayTracingGeometryMetadata] = []
        var instances: [RayTracingInstanceMetadata] = []
        var textureIndex: UInt32 = 0

        for mesh in roomMeshes {
            guard let positionData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition) else {
                throw RayTracingAccelerationBuilderError.invalidMesh
            }
            let normalData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal)
            let textureData = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate)
            let vertexOffset = UInt32(vertices.count)

            for vertexIndex in 0..<mesh.vertexCount {
                let position = readFloat3(positionData, index: vertexIndex)
                let normal = normalData.map { readFloat3($0, index: vertexIndex) } ?? SIMD3<Float>(0, 1, 0)
                let texCoord = textureData.map { readFloat2($0, index: vertexIndex) } ?? .zero
                vertices.append(RayTracingPackedVertex(
                    position: SIMD4<Float>(position.x, position.y, position.z, 1),
                    normal: SIMD4<Float>(normal.x, normal.y, normal.z, 0),
                    texCoordAndPadding: SIMD4<Float>(texCoord.x, texCoord.y, 0, 0)))
            }

            let geometryOffset = UInt32(geometries.count)
            let submeshes = (mesh.submeshes as? [MDLSubmesh]) ?? []
            for submesh in submeshes {
                let indexOffset = UInt32(indices.count)
                let indexMap = submesh.indexBuffer.map()
                for index in 0..<submesh.indexCount {
                    indices.append(vertexOffset + readIndex(indexMap.bytes,
                                                            index: index,
                                                            type: submesh.indexType))
                }
                geometries.append(RayTracingGeometryMetadata(
                    vertexOffset: vertexOffset,
                    indexOffset: indexOffset,
                    textureIndex: textureIndex,
                    triangleCount: UInt32(submesh.indexCount / 3)))
                textureIndex += 1
            }

            instances.append(RayTracingInstanceMetadata(
                geometryOffset: geometryOffset,
                geometryCount: UInt32(submeshes.count),
                materialKind: UInt32(RayTracingMaterialKind.room.rawValue)))
        }

        instances.append(contentsOf: repeatElement(
            RayTracingInstanceMetadata(
                geometryOffset: 0,
                geometryCount: 0,
                materialKind: UInt32(RayTracingMaterialKind.pureReflection.rawValue)),
            count: reflectiveObjectCapacity))

        return RayTracingCPUHitData(vertices: vertices,
                                    indices: indices,
                                    geometries: geometries,
                                    instances: instances)
    }

    private static func readFloat3(_ data: MDLVertexAttributeData, index: Int) -> SIMD3<Float> {
        let source = data.dataStart.advanced(by: index * data.stride)
        var values = (Float.zero, Float.zero, Float.zero)
        memcpy(&values, source, MemoryLayout<Float>.stride * 3)
        return SIMD3<Float>(values.0, values.1, values.2)
    }

    private static func readFloat2(_ data: MDLVertexAttributeData, index: Int) -> SIMD2<Float> {
        let source = data.dataStart.advanced(by: index * data.stride)
        var values = (Float.zero, Float.zero)
        memcpy(&values, source, MemoryLayout<Float>.stride * 2)
        return SIMD2<Float>(values.0, values.1)
    }

    private static func readIndex(_ bytes: UnsafeMutableRawPointer,
                                  index: Int,
                                  type: MDLIndexBitDepth) -> UInt32 {
        switch type {
        case .uInt8:
            UInt32(bytes.load(fromByteOffset: index, as: UInt8.self))
        case .uInt16:
            UInt32(bytes.load(fromByteOffset: index * 2, as: UInt16.self))
        case .uInt32:
            bytes.load(fromByteOffset: index * 4, as: UInt32.self)
        @unknown default:
            0
        }
    }
}

final class RayTracingHitDataBuffers: @unchecked Sendable {
    let vertices: MTLBuffer
    let indices: MTLBuffer
    let geometries: MTLBuffer
    let instances: MTLBuffer
    let objects: MTLBuffer

    init(device: MTLDevice, data: RayTracingCPUHitData) throws {
        guard let vertices = Self.makeBuffer(device: device, values: data.vertices),
              let indices = Self.makeBuffer(device: device, values: data.indices),
              let geometries = Self.makeBuffer(device: device, values: data.geometries),
              let instances = Self.makeBuffer(device: device, values: data.instances),
              let objects = device.makeBuffer(
                length: MemoryLayout<RayTracingObjectMetadata>.stride * data.instances.count,
                options: .storageModeShared) else {
            throw RayTracingAccelerationBuilderError.allocationFailed
        }
        self.vertices = vertices
        self.indices = indices
        self.geometries = geometries
        self.instances = instances
        self.objects = objects
        vertices.label = "Ray Hit Vertices"
        indices.label = "Ray Hit Indices"
        geometries.label = "Ray Hit Geometry Metadata"
        instances.label = "Ray Hit Instance Metadata"
        objects.label = "Ray Hit Object Transforms"
    }

    func setObjectTransform(_ transform: matrix_float4x4,
                            sphereRadius: Float,
                            at index: Int) {
        let pointer = objects.contents()
            .bindMemory(to: RayTracingObjectMetadata.self,
                        capacity: objects.length / MemoryLayout<RayTracingObjectMetadata>.stride)
        pointer[index] = RayTracingObjectMetadata(
            objectToWorld: transform,
            worldToObject: transform.inverse,
            sphereCenterAndRadius: SIMD4<Float>(transform.columns.3.x,
                                                 transform.columns.3.y,
                                                 transform.columns.3.z,
                                                 sphereRadius))
    }

    private static func makeBuffer<T>(device: MTLDevice, values: [T]) -> MTLBuffer? {
        guard !values.isEmpty else { return nil }
        return values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress,
                                     length: bytes.count,
                                     options: .storageModeShared)
        }
    }
}
