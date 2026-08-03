//
//  RightHandPinchTracker.swift
//  roomraycast
//

import ARKit
import simd

nonisolated enum HandSide: Hashable, Sendable {
    case left
    case right
}

nonisolated enum HandPinchPhase: Sendable {
    case unavailable
    case open
    case began
    case pinching
    case ended
}

nonisolated struct HandPinchFrame: Sendable {
    let phase: HandPinchPhase
    let midpoint: SIMD3<Float>?
    let fingerDistance: Float?
    let originFromDriverTransform: matrix_float4x4?

    static let unavailable = HandPinchFrame(phase: .unavailable,
                                            midpoint: nil,
                                            fingerDistance: nil,
                                            originFromDriverTransform: nil)
}

nonisolated struct HandPinchTracker {
    let side: HandSide
    let enterDistance: Float
    let exitDistance: Float

    private var isPinching = false

    init(side: HandSide,
         enterDistance: Float = 0.0125,
         exitDistance: Float = 0.02) {
        precondition(enterDistance < exitDistance)
        self.side = side
        self.enterDistance = enterDistance
        self.exitDistance = exitDistance
    }

    mutating func update(with handAnchor: HandAnchor?) -> HandPinchFrame {
        guard let handAnchor,
              matchesExpectedSide(handAnchor.chirality),
              handAnchor.isTracked,
              let skeleton = handAnchor.handSkeleton else {
            return trackingUnavailableFrame()
        }

        let thumbTip = skeleton.joint(.thumbTip)
        let indexTip = skeleton.joint(.indexFingerTip)
        let wrist = skeleton.joint(.wrist)
        guard wrist.isTracked else {
            return trackingUnavailableFrame()
        }

        let originFromDriverTransform = handAnchor.originFromAnchorTransform
            * wrist.anchorFromJointTransform
        guard thumbTip.isTracked, indexTip.isTracked else {
            let phase: HandPinchPhase = isPinching ? .pinching : .unavailable
            return HandPinchFrame(
                phase: phase,
                midpoint: nil,
                fingerDistance: nil,
                originFromDriverTransform: originFromDriverTransform)
        }

        let thumbPosition = worldPosition(of: thumbTip, handAnchor: handAnchor)
        let indexPosition = worldPosition(of: indexTip, handAnchor: handAnchor)
        let midpoint = (thumbPosition + indexPosition) * 0.5
        let distance = simd_distance(thumbPosition, indexPosition)

        if isPinching {
            if distance >= exitDistance {
                isPinching = false
                return HandPinchFrame(phase: .ended,
                                           midpoint: midpoint,
                                           fingerDistance: distance,
                                           originFromDriverTransform: originFromDriverTransform)
            }
            return HandPinchFrame(phase: .pinching,
                                       midpoint: midpoint,
                                       fingerDistance: distance,
                                       originFromDriverTransform: originFromDriverTransform)
        }

        if distance <= enterDistance {
            isPinching = true
            return HandPinchFrame(phase: .began,
                                       midpoint: midpoint,
                                       fingerDistance: distance,
                                       originFromDriverTransform: originFromDriverTransform)
        }

        return HandPinchFrame(phase: .open,
                                   midpoint: midpoint,
                                   fingerDistance: distance,
                                   originFromDriverTransform: originFromDriverTransform)
    }

    private mutating func trackingUnavailableFrame() -> HandPinchFrame {
        let phase: HandPinchPhase = isPinching ? .ended : .unavailable
        isPinching = false
        return HandPinchFrame(phase: phase,
                              midpoint: nil,
                              fingerDistance: nil,
                              originFromDriverTransform: nil)
    }

    private func matchesExpectedSide(_ chirality: HandAnchor.Chirality) -> Bool {
        switch (side, chirality) {
        case (.left, .left), (.right, .right): true
        default: false
        }
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
