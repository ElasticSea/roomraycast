//
//  RightHandPinchTracker.swift
//  roomraycast
//

import ARKit
import simd

enum RightHandPinchPhase: Sendable {
    case unavailable
    case open
    case began
    case pinching
    case ended
}

struct RightHandPinchFrame: Sendable {
    let phase: RightHandPinchPhase
    let midpoint: SIMD3<Float>?
    let fingerDistance: Float?
    let originFromDriverTransform: matrix_float4x4?

    static let unavailable = RightHandPinchFrame(phase: .unavailable,
                                                  midpoint: nil,
                                                  fingerDistance: nil,
                                                  originFromDriverTransform: nil)
}

struct RightHandPinchTracker {
    let enterDistance: Float
    let exitDistance: Float

    private var isPinching = false

    init(enterDistance: Float = 0.025, exitDistance: Float = 0.04) {
        precondition(enterDistance < exitDistance)
        self.enterDistance = enterDistance
        self.exitDistance = exitDistance
    }

    mutating func update(with handAnchor: HandAnchor?) -> RightHandPinchFrame {
        guard let handAnchor,
              handAnchor.chirality == .right,
              handAnchor.isTracked,
              let skeleton = handAnchor.handSkeleton else {
            return trackingUnavailableFrame()
        }

        let thumbTip = skeleton.joint(.thumbTip)
        let indexTip = skeleton.joint(.indexFingerTip)
        let wrist = skeleton.joint(.wrist)
        guard thumbTip.isTracked, indexTip.isTracked, wrist.isTracked else {
            return trackingUnavailableFrame()
        }

        let thumbPosition = worldPosition(of: thumbTip, handAnchor: handAnchor)
        let indexPosition = worldPosition(of: indexTip, handAnchor: handAnchor)
        let midpoint = (thumbPosition + indexPosition) * 0.5
        let distance = simd_distance(thumbPosition, indexPosition)
        let originFromDriverTransform = handAnchor.originFromAnchorTransform
            * wrist.anchorFromJointTransform

        if isPinching {
            if distance >= exitDistance {
                isPinching = false
                return RightHandPinchFrame(phase: .ended,
                                           midpoint: midpoint,
                                           fingerDistance: distance,
                                           originFromDriverTransform: originFromDriverTransform)
            }
            return RightHandPinchFrame(phase: .pinching,
                                       midpoint: midpoint,
                                       fingerDistance: distance,
                                       originFromDriverTransform: originFromDriverTransform)
        }

        if distance <= enterDistance {
            isPinching = true
            return RightHandPinchFrame(phase: .began,
                                       midpoint: midpoint,
                                       fingerDistance: distance,
                                       originFromDriverTransform: originFromDriverTransform)
        }

        return RightHandPinchFrame(phase: .open,
                                   midpoint: midpoint,
                                   fingerDistance: distance,
                                   originFromDriverTransform: originFromDriverTransform)
    }

    private mutating func trackingUnavailableFrame() -> RightHandPinchFrame {
        let phase: RightHandPinchPhase = isPinching ? .ended : .unavailable
        isPinching = false
        return RightHandPinchFrame(phase: phase,
                                   midpoint: nil,
                                   fingerDistance: nil,
                                   originFromDriverTransform: nil)
    }

    private func worldPosition(of joint: HandSkeleton.Joint,
                               handAnchor: HandAnchor) -> SIMD3<Float> {
        let originFromJoint = handAnchor.originFromAnchorTransform
            * joint.anchorFromJointTransform
        return SIMD3<Float>(originFromJoint.columns.3.x,
                            originFromJoint.columns.3.y,
                            originFromJoint.columns.3.z)
    }
}
