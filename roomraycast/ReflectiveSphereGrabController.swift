//
//  ReflectiveSphereGrabController.swift
//  roomraycast
//

import simd

struct ReflectiveSphereGrabController {
    let collisionTolerance: Float
    let smoothingRetention: Float

    private(set) var isGrabbed = false
    private var driverFromSphereTransform = matrix_identity_float4x4

    init(collisionTolerance: Float = 0.02,
         smoothingRetention: Float = 0.99) {
        precondition(smoothingRetention >= 0 && smoothingRetention < 1)
        self.collisionTolerance = collisionTolerance
        self.smoothingRetention = smoothingRetention
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

        let targetTransform = originFromDriverTransform * driverFromSphereTransform
        let newTransform = smooth(transform, toward: targetTransform)
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

    private func smooth(_ transform: matrix_float4x4,
                        toward target: matrix_float4x4) -> matrix_float4x4 {
        let blend = 1 - smoothingRetention
        let translation = SIMD3<Float>(transform.columns.3.x,
                                       transform.columns.3.y,
                                       transform.columns.3.z)
        let targetTranslation = SIMD3<Float>(target.columns.3.x,
                                             target.columns.3.y,
                                             target.columns.3.z)
        let smoothedTranslation = translation + (targetTranslation - translation) * blend

        let rotation = rotationQuaternion(of: transform)
        let targetRotation = rotationQuaternion(of: target)
        var smoothedTransform = matrix_float4x4(
            simd_slerp(rotation, targetRotation, blend))
        smoothedTransform.columns.3 = SIMD4<Float>(smoothedTranslation, 1)
        return smoothedTransform
    }

    private func rotationQuaternion(of transform: matrix_float4x4) -> simd_quatf {
        let rotationMatrix = matrix_float3x3(columns: (
            SIMD3<Float>(transform.columns.0.x,
                         transform.columns.0.y,
                         transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x,
                         transform.columns.1.y,
                         transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x,
                         transform.columns.2.y,
                         transform.columns.2.z)))
        return simd_quatf(rotationMatrix)
    }

    private func transformChanged(_ old: matrix_float4x4,
                                  _ new: matrix_float4x4) -> Bool {
        simd_distance(old.columns.0, new.columns.0) > 0.0001
            || simd_distance(old.columns.1, new.columns.1) > 0.0001
            || simd_distance(old.columns.2, new.columns.2) > 0.0001
            || simd_distance(old.columns.3, new.columns.3) > 0.0001
    }
}
