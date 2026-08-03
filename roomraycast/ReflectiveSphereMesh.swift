//
//  ReflectiveSphereMesh.swift
//  roomraycast
//

import MetalKit
import ModelIO

enum ReflectiveSphereMesh {
    static func metalVertexDescriptor() -> MTLVertexDescriptor {
        let descriptor = MTLVertexDescriptor()
        descriptor.attributes[VertexAttribute.position.rawValue].format = .float3
        descriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        descriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        descriptor.layouts[BufferIndex.meshPositions.rawValue].stride = MemoryLayout<SIMD3<Float>>.stride

        descriptor.attributes[VertexAttribute.texcoord.rawValue].format = .float2
        descriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        descriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        descriptor.attributes[VertexAttribute.normal.rawValue].format = .float3
        descriptor.attributes[VertexAttribute.normal.rawValue].offset = 8
        descriptor.attributes[VertexAttribute.normal.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue
        descriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 20
        return descriptor
    }

    static func make(device: MTLDevice) throws -> MTKMesh {
        let allocator = MTKMeshBufferAllocator(device: device)
        let radii = SIMD3<Float>(repeating: ReflectiveObject.radius)
        let modelMesh = MDLMesh.newEllipsoid(withRadii: radii,
                                             radialSegments: 64,
                                             verticalSegments: 32,
                                             geometryType: .triangles,
                                             inwardNormals: false,
                                             hemisphere: false,
                                             allocator: allocator)
        modelMesh.name = "Pure Reflection Sphere"

        let modelVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(metalVertexDescriptor())
        guard let attributes = modelVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
        attributes[VertexAttribute.normal.rawValue].name = MDLVertexAttributeNormal
        modelMesh.vertexDescriptor = modelVertexDescriptor

        return try MTKMesh(mesh: modelMesh, device: device)
    }
}
