//
//  ReflectiveSphere.swift
//  roomraycast
//

import simd

struct ReflectiveSphere: Sendable {
    static let diameter: Float = 0.25
    static let radius: Float = diameter / 2

    var originFromSphereTransform: matrix_float4x4?
}
