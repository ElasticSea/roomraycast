//
//  ReflectiveSphereGrabController.swift
//  roomraycast
//

import simd

struct ReflectiveSphereGrabController {
    let collisionTolerance: Float

    private(set) var isGrabbed = false
    private var driverFromSphereTransform = matrix_identity_float4x4

    init(collisionTolerance: Float = 0.02) {
        self.collisionTolerance = collisionTolerance
    }

    mutating func update(pinch: RightHandPinchFrame,
                         sphere: inout ReflectiveSphere) -> Bool {
        switch pinch.phase {
        case .began:
            return beginGrabIfPossible(midpoint: pinch.midpoint,
                                       originFromDriverTransform: pinch.originFromDriverTransform,
                                       sphere: sphere)
        case .pinching:
            return moveGrabbedSphere(
                originFromDriverTransform: pinch.originFromDriverTransform,
                sphere: &sphere)
        case .ended, .unavailable:
            endGrab()
            return false
        case .open:
            return false
        }
    }

    private mutating func beginGrabIfPossible(midpoint: SIMD3<Float>?,
                                              originFromDriverTransform: matrix_float4x4?,
                                              sphere: ReflectiveSphere) -> Bool {
        guard let midpoint,
              let originFromDriverTransform,
              let transform = sphere.originFromSphereTransform else {
            return false
        }

        let sphereCenter = SIMD3<Float>(transform.columns.3.x,
                                        transform.columns.3.y,
                                        transform.columns.3.z)
        guard simd_distance(midpoint, sphereCenter)
                <= ReflectiveSphere.radius + collisionTolerance else {
            return false
        }

        driverFromSphereTransform = originFromDriverTransform.inverse * transform
        isGrabbed = true
        return false
    }

    private mutating func moveGrabbedSphere(originFromDriverTransform: matrix_float4x4?,
                                            sphere: inout ReflectiveSphere) -> Bool {
        guard isGrabbed,
              let originFromDriverTransform,
              let transform = sphere.originFromSphereTransform else {
            return false
        }

        let newTransform = originFromDriverTransform * driverFromSphereTransform
        guard transformChanged(transform, newTransform) else {
            return false
        }

        sphere.originFromSphereTransform = newTransform
        return true
    }

    private mutating func endGrab() {
        isGrabbed = false
        driverFromSphereTransform = matrix_identity_float4x4
    }

    private func transformChanged(_ old: matrix_float4x4,
                                  _ new: matrix_float4x4) -> Bool {
        simd_distance(old.columns.0, new.columns.0) > 0.0001
            || simd_distance(old.columns.1, new.columns.1) > 0.0001
            || simd_distance(old.columns.2, new.columns.2) > 0.0001
            || simd_distance(old.columns.3, new.columns.3) > 0.0001
    }
}
