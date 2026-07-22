//
//  ReflectiveSpherePlacement.swift
//  roomraycast
//

import ARKit
import simd

struct ReflectiveSpherePlacement {
    private(set) var spawnHeadTransform: matrix_float4x4?

    @discardableResult
    mutating func captureFirstTrackedHeadPose(from deviceAnchor: DeviceAnchor?) -> Bool {
        guard spawnHeadTransform == nil,
              let deviceAnchor,
              deviceAnchor.isTracked else {
            return false
        }

        spawnHeadTransform = deviceAnchor.originFromAnchorTransform
        return true
    }

    @discardableResult
    func placeSphereIfPossible(_ sphere: inout ReflectiveSphere) -> Bool {
        guard sphere.originFromSphereTransform == nil,
              let spawnHeadTransform else {
            return false
        }

        var headFromSphereTransform = matrix_identity_float4x4
        headFromSphereTransform.columns.3.z = -ReflectiveSphere.spawnDistance
        sphere.originFromSphereTransform = spawnHeadTransform * headFromSphereTransform
        return true
    }
}
