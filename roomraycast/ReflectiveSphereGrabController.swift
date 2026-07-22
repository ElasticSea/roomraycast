//
//  ReflectiveSphereGrabController.swift
//  roomraycast
//

import simd

struct ReflectiveSphereGrabController {
    let collisionTolerance: Float

    private(set) var isGrabbed = false
    private var sphereFromPinchOffset = SIMD3<Float>.zero

    init(collisionTolerance: Float = 0.02) {
        self.collisionTolerance = collisionTolerance
    }

    mutating func update(pinch: RightHandPinchFrame,
                         sphere: inout ReflectiveSphere) -> Bool {
        switch pinch.phase {
        case .began:
            return beginGrabIfPossible(midpoint: pinch.midpoint, sphere: sphere)
        case .pinching:
            return moveGrabbedSphere(midpoint: pinch.midpoint, sphere: &sphere)
        case .ended, .unavailable:
            endGrab()
            return false
        case .open:
            return false
        }
    }

    private mutating func beginGrabIfPossible(midpoint: SIMD3<Float>?,
                                              sphere: ReflectiveSphere) -> Bool {
        guard let midpoint,
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

        sphereFromPinchOffset = sphereCenter - midpoint
        isGrabbed = true
        return false
    }

    private mutating func moveGrabbedSphere(midpoint: SIMD3<Float>?,
                                            sphere: inout ReflectiveSphere) -> Bool {
        guard isGrabbed,
              let midpoint,
              var transform = sphere.originFromSphereTransform else {
            return false
        }

        let newCenter = midpoint + sphereFromPinchOffset
        let oldCenter = SIMD3<Float>(transform.columns.3.x,
                                     transform.columns.3.y,
                                     transform.columns.3.z)
        guard simd_distance(oldCenter, newCenter) > 0.0001 else {
            return false
        }

        transform.columns.3 = SIMD4<Float>(newCenter.x, newCenter.y, newCenter.z, 1)
        sphere.originFromSphereTransform = transform
        return true
    }

    private mutating func endGrab() {
        isGrabbed = false
        sphereFromPinchOffset = .zero
    }
}
