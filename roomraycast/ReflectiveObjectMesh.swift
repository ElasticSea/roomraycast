//
//  ReflectiveObjectMesh.swift
//  roomraycast
//

import MetalKit
import ModelIO

struct ReflectiveObjectMesh {
    private static let maximumBoundingBoxDimension: Float = 0.25

    let modelMesh: MDLMesh
    let mesh: MTKMesh
    let localTransform: matrix_float4x4

    static func makeAll(device: MTLDevice) throws -> [ReflectiveObjectKind: ReflectiveObjectMesh] {
        let sphere = try ReflectiveSphereMesh.make(device: device)
        var result: [ReflectiveObjectKind: ReflectiveObjectMesh] = [
            .sphere: ReflectiveObjectMesh(modelMesh: sphere.modelMesh,
                                          mesh: sphere.metalMesh,
                                          localTransform: matrix_identity_float4x4)
        ]

        for kind in ReflectiveObjectKind.allCases where kind != .sphere {
            result[kind] = try load(kind, device: device)
        }
        return result
    }

    private static func load(_ kind: ReflectiveObjectKind,
                             device: MTLDevice) throws -> ReflectiveObjectMesh {
        guard let resourceName = kind.resourceName else {
            throw RendererError.missingReflectiveObjectAsset(kind.rawValue)
        }
        let url = Bundle.main.url(forResource: resourceName, withExtension: "usdz")
            ?? Bundle.main.url(forResource: resourceName,
                               withExtension: "usdz",
                               subdirectory: "meshes")
        guard let url else {
            throw RendererError.missingReflectiveObjectAsset(kind.rawValue)
        }

        let allocator = MTKMeshBufferAllocator(device: device)
        let vertexDescriptor = MTKModelIOVertexDescriptorFromMetal(
            ReflectiveSphereMesh.metalVertexDescriptor())
        guard let attributes = vertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        attributes[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal

        let asset = MDLAsset(url: url,
                             vertexDescriptor: vertexDescriptor,
                             bufferAllocator: allocator)
        guard let modelMesh = asset.childObjects(of: MDLMesh.self).first as? MDLMesh else {
            throw RendererError.badVertexDescriptor
        }
        let mesh = try MTKMesh(mesh: modelMesh, device: device)

        let bounds = asset.boundingBox
        let minimum = bounds.minBounds
        let maximum = bounds.maxBounds
        let size = maximum - minimum
        let largestDimension = max(size.x, max(size.y, size.z))
        guard largestDimension.isFinite, largestDimension > 0.0001 else {
            throw RendererError.badVertexDescriptor
        }

        let center = (minimum + maximum) * 0.5
        let scale = maximumBoundingBoxDimension / largestDimension
        let normalization = matrix4x4_scale(scale)
            * matrix4x4_translation(-center.x, -center.y, -center.z)
        let assetTransform = MDLTransform.globalTransform(with: modelMesh, atTime: 0)

        return ReflectiveObjectMesh(modelMesh: modelMesh,
                                    mesh: mesh,
                                    localTransform: normalization * assetTransform)
    }
}
