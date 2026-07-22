//
//  ReflectiveSphereMesh.swift
//  roomraycast
//

import MetalKit
import ModelIO

enum ReflectiveSphereMesh {
    static func make(device: MTLDevice) throws -> MTKMesh {
        let allocator = MTKMeshBufferAllocator(device: device)
        let radii = SIMD3<Float>(repeating: ReflectiveSphere.radius)
        let modelMesh = MDLMesh.newEllipsoid(withRadii: radii,
                                             radialSegments: 64,
                                             verticalSegments: 32,
                                             geometryType: .triangles,
                                             inwardNormals: false,
                                             hemisphere: false,
                                             allocator: allocator)
        modelMesh.name = "Pure Reflection Sphere"
        return try MTKMesh(mesh: modelMesh, device: device)
    }
}
