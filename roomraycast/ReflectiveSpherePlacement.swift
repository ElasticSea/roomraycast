//
//  ReflectiveSpherePlacement.swift
//  roomraycast
//

import ARKit
import simd

nonisolated struct ReflectiveObjectPlacement {
    private(set) var latestHeadTransform: matrix_float4x4?

    mutating func updateTrackedHeadPose(from deviceAnchor: DeviceAnchor?) {
        guard let deviceAnchor,
              deviceAnchor.isTracked else {
            return
        }

        latestHeadTransform = deviceAnchor.originFromAnchorTransform
    }

    func makeSpawnTransform() -> matrix_float4x4? {
        guard let latestHeadTransform else { return nil }

        var headFromObjectTransform = matrix_identity_float4x4
        headFromObjectTransform.columns.3.z = -ReflectiveObject.spawnDistance
        return latestHeadTransform * headFromObjectTransform
    }
}
