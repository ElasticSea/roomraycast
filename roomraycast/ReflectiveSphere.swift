//
//  ReflectiveSphere.swift
//  roomraycast
//

import simd
import Foundation

nonisolated struct PureReflectionMaterial: Sendable, Equatable {
    var reflectivity: Float = 1
    var roughness: Float = 0
    var metallic: Float = 1
    var diffuseContribution: Float = 0
    var baseColor = SIMD3<Float>(repeating: 1)
    var missColor = SIMD3<Float>(repeating: 1)
}

nonisolated final class ReflectiveObjectMaterialState: @unchecked Sendable {
    private let lock = NSLock()
    private var material = PureReflectionMaterial()

    func snapshot() -> PureReflectionMaterial {
        lock.lock()
        defer { lock.unlock() }
        return material
    }

    func set(roughness: Float,
             metallic: Float,
             baseColor: SIMD3<Float>,
             missColor: SIMD3<Float>) {
        lock.lock()
        material.roughness = min(max(roughness, 0), 1)
        material.metallic = min(max(metallic, 0), 1)
        material.diffuseContribution = 1 - material.metallic
        material.baseColor = simd_clamp(baseColor, .zero, SIMD3<Float>(repeating: 1))
        material.missColor = simd_clamp(missColor, .zero, SIMD3<Float>(repeating: 1))
        lock.unlock()
    }
}

nonisolated struct ReflectiveObjectID: Hashable, Sendable {
    let rawValue: Int
}

nonisolated struct ReflectiveObject: Sendable {
    static let diameter: Float = 0.25
    static let radius: Float = diameter / 2
    static let spawnDistance: Float = 0.5
    static let maximumCount = 32

    let id: ReflectiveObjectID
    let kind: ReflectiveObjectKind
    var originFromObjectTransform: matrix_float4x4
    var material = PureReflectionMaterial()
}
