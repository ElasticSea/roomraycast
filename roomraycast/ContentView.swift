//
//  ContentView.swift
//  roomraycast
//
//  Created by Elastic Sea on 7/22/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var appModel
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace

    @State private var isImporting = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 12) {
            Button("Open") {
                isImporting = true
            }
            .fontWeight(.semibold)

            if appModel.importedModelURL != nil {
                Divider()
                RoomManipulationControls(modelTransform: appModel.modelTransform)
            }
        }
        .padding()
        .disabled(appModel.immersiveSpaceState == .inTransition)
        .fileImporter(isPresented: $isImporting,
                      allowedContentTypes: [.usdz]) { result in
            Task { @MainActor in
                await openModel(result)
            }
        }
        .alert("Couldn’t Open Model",
               isPresented: Binding(get: { errorMessage != nil },
                                    set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage ?? "Unknown error")
        }
    }

    @MainActor
    private func openModel(_ result: Result<URL, any Error>) async {
        do {
            let selectedURL = try result.get()

            if appModel.immersiveSpaceState == .open {
                appModel.immersiveSpaceState = .inTransition
                await dismissImmersiveSpace()
                appModel.immersiveSpaceState = .closed
            }

            try appModel.importModel(from: selectedURL)
            appModel.immersiveSpaceState = .inTransition

            switch await openImmersiveSpace(id: appModel.immersiveSpaceID) {
                case .opened:
                    break
                case .userCancelled:
                    appModel.immersiveSpaceState = .closed
                case .error:
                    appModel.immersiveSpaceState = .closed
                    errorMessage = "visionOS could not open the immersive space."
                @unknown default:
                    appModel.immersiveSpaceState = .closed
                    errorMessage = "visionOS returned an unknown result."
            }
        } catch {
            appModel.immersiveSpaceState = .closed
            errorMessage = error.localizedDescription
        }
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
