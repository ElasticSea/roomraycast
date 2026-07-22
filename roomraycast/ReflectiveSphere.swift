//
//  ReflectiveSphere.swift
//  roomraycast
//

import simd

struct PureReflectionMaterial: Sendable, Equatable {
    let reflectivity: Float = 1
    let roughness: Float = 0
    let metallic: Float = 1
    let diffuseContribution: Float = 0
}

struct ReflectiveSphere: Sendable {
    static let diameter: Float = 0.25
    static let radius: Float = diameter / 2

    var originFromSphereTransform: matrix_float4x4?
    var material = PureReflectionMaterial()
}
