//
//  ReflectiveSphere.swift
//  roomraycast
//

import simd

nonisolated struct PureReflectionMaterial: Sendable, Equatable {
    let reflectivity: Float = 1
    let roughness: Float = 0
    let metallic: Float = 1
    let diffuseContribution: Float = 0
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
