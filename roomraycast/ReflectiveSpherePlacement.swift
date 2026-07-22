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
}
