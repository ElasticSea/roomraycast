//
//  RoomManipulation.swift
//  roomraycast
//

import SwiftUI

struct ModelTransformSnapshot: Sendable {
    var translation = SIMD3<Float>.zero
    var yaw: Float = 0
    var scale: Float = 1
}

final class ModelTransformState: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ModelTransformSnapshot()

    func snapshot() -> ModelTransformSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func move(x: Float = 0, y: Float = 0, z: Float = 0) {
        lock.lock()
        value.translation += SIMD3<Float>(x, y, z)
        lock.unlock()
    }

    func rotate(by degrees: Float) {
        lock.lock()
        value.yaw += degrees * .pi / 180
        lock.unlock()
    }

    func resize(by factor: Float) {
        lock.lock()
        value.scale = min(max(value.scale * factor, 0.1), 10)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        value = ModelTransformSnapshot()
        lock.unlock()
    }
}

struct RoomManipulationControls: View {
    let modelTransform: ModelTransformState

    var body: some View {
        VStack(spacing: 8) {
            Text("Position Room")
                .font(.headline)

            axisRow("X", color: .red,
                    negativeIcon: "arrow.left", positiveIcon: "arrow.right",
                    negativeAction: { modelTransform.move(x: -0.1) },
                    positiveAction: { modelTransform.move(x: 0.1) })

            axisRow("Y", color: .green,
                    negativeIcon: "arrow.down", positiveIcon: "arrow.up",
                    negativeAction: { modelTransform.move(y: -0.1) },
                    positiveAction: { modelTransform.move(y: 0.1) })

            axisRow("Z", color: .blue,
                    negativeIcon: "arrow.up", positiveIcon: "arrow.down",
                    negativeAction: { modelTransform.move(z: -0.1) },
                    positiveAction: { modelTransform.move(z: 0.1) })

            HStack(spacing: 8) {
                Text("Rotate")
                    .frame(width: 54, alignment: .leading)
                controlButton("rotate.left") { modelTransform.rotate(by: -5) }
                controlButton("rotate.right") { modelTransform.rotate(by: 5) }
            }

            HStack(spacing: 8) {
                Text("Scale")
                    .frame(width: 54, alignment: .leading)
                controlButton("minus") { modelTransform.resize(by: 1 / 1.1) }
                controlButton("plus") { modelTransform.resize(by: 1.1) }
            }

            Button("Reset") {
                modelTransform.reset()
            }
            .buttonStyle(.bordered)
        }
    }

    private func axisRow(_ label: String,
                         color: Color,
                         negativeIcon: String,
                         positiveIcon: String,
                         negativeAction: @escaping () -> Void,
                         positiveAction: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .fontWeight(.bold)
                .foregroundStyle(color)
                .frame(width: 54, alignment: .leading)
            controlButton(negativeIcon, action: negativeAction)
            controlButton(positiveIcon, action: positiveAction)
        }
    }

    private func controlButton(_ systemName: String,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .frame(width: 24, height: 20)
        }
        .buttonStyle(.bordered)
    }
}
