//
//  ReflectiveSphereGrabController.swift
//  roomraycast
//

import simd

nonisolated struct ReflectiveObjectGrabController {
    private struct GrabState {
        let objectID: ReflectiveObjectID
        var driverFromObjectTransform: matrix_float4x4
    }

    let collisionTolerance: Float
    let smoothingRetention: Float

    private var grabsByHand: [HandSide: GrabState] = [:]

    init(collisionTolerance: Float = 0.02,
         smoothingRetention: Float = 0.8) {
        precondition(smoothingRetention >= 0 && smoothingRetention < 1)
        self.collisionTolerance = collisionTolerance
        self.smoothingRetention = smoothingRetention
    }

    mutating func update(hand: HandSide,
                         pinch: HandPinchFrame,
                         objects: inout [ReflectiveObject]) -> ReflectiveObjectID? {
        switch pinch.phase {
        case .began:
            return beginGrabIfPossible(midpoint: pinch.midpoint,
                                       originFromDriverTransform: pinch.originFromDriverTransform,
                                       hand: hand,
                                       objects: objects)
        case .pinching:
            return moveGrabbedObject(
                originFromDriverTransform: pinch.originFromDriverTransform,
                hand: hand,
                objects: &objects)
        case .ended, .unavailable:
            endGrab(for: hand)
            return nil
        case .open:
            return nil
        }
    }

    mutating func cancelAllGrabs() {
        grabsByHand.removeAll(keepingCapacity: true)
    }

    private mutating func beginGrabIfPossible(midpoint: SIMD3<Float>?,
                                              originFromDriverTransform: matrix_float4x4?,
                                              hand: HandSide,
                                              objects: [ReflectiveObject]) -> ReflectiveObjectID? {
        guard let midpoint,
              let originFromDriverTransform else {
            return nil
        }

        let closest = objects.compactMap { object -> (ReflectiveObject, Float)? in
            let transform = object.originFromObjectTransform
            let center = SIMD3<Float>(transform.columns.3.x,
                                      transform.columns.3.y,
                                      transform.columns.3.z)
            let distance = simd_distance(midpoint, center)
            guard distance <= ReflectiveObject.radius + collisionTolerance else { return nil }
            return (object, distance)
        }.min { $0.1 < $1.1 }
        guard let object = closest?.0 else {
            return nil
        }

        grabsByHand = grabsByHand.filter { existingHand, grab in
            existingHand == hand || grab.objectID != object.id
        }
        grabsByHand[hand] = GrabState(
            objectID: object.id,
            driverFromObjectTransform: originFromDriverTransform.inverse
                * object.originFromObjectTransform)
        return nil
    }

    private mutating func moveGrabbedObject(originFromDriverTransform: matrix_float4x4?,
                                            hand: HandSide,
                                            objects: inout [ReflectiveObject]) -> ReflectiveObjectID? {
        guard let grab = grabsByHand[hand],
              let originFromDriverTransform,
              let objectIndex = objects.firstIndex(where: { $0.id == grab.objectID }) else {
            return nil
        }

        let transform = objects[objectIndex].originFromObjectTransform
        let targetTransform = originFromDriverTransform * grab.driverFromObjectTransform
        let newTransform = smooth(transform, toward: targetTransform)
        guard transformChanged(transform, newTransform) else {
            return nil
        }

        objects[objectIndex].originFromObjectTransform = newTransform
        return grab.objectID
    }

    private mutating func endGrab(for hand: HandSide) {
        grabsByHand.removeValue(forKey: hand)
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
