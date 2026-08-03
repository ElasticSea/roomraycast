//
//  ContentView.swift
//  roomraycast
//
//  Created by Elastic Sea on 7/22/26.
//

import Darwin
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var appModel
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.scenePhase) private var scenePhase

    @State private var isImporting = false
    @State private var errorMessage: String?
    @State private var errorTitle = "Couldn’t Open Model"

    var body: some View {
        VStack(spacing: 12) {
            Button("Open") {
                isImporting = true
            }
            .fontWeight(.semibold)

            if appModel.hasSavedAnchoredModel {
                ForEach(appModel.savedAnchoredModels, id: \.anchorID) { savedModel in
                    Button("Open \(savedModel.displayName)") {
                        Task { @MainActor in
                            await openSavedAnchoredModel(savedModel)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            if appModel.importedModelURL != nil {
                Divider()
                RoomManipulationControls(modelTransform: appModel.modelTransform)

                Toggle("Show Room", isOn: Binding(
                    get: { appModel.isRoomVisible },
                    set: { appModel.isRoomVisible = $0 }
                ))

                HStack {
                    ForEach(ReflectiveObjectKind.allCases) { kind in
                        Button("Add \(kind.title)") {
                            appModel.spawnReflectiveObject(kind)
                        }
                    }
                }

                Button("Remove All Objects", role: .destructive) {
                    appModel.removeAllReflectiveObjects()
                }
                .controlSize(.small)

                VStack(alignment: .leading) {
                    Text("Reflection Bounces: \(appModel.reflectionBounceCount)")
                    Slider(value: Binding(
                        get: { Double(appModel.reflectionBounceCount) },
                        set: { appModel.reflectionBounceCount = Int($0) }
                    ), in: 0...10, step: 1)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Object Material")
                        Spacer()
                        ColorPicker("Color", selection: Binding(
                            get: { appModel.reflectiveObjectColor },
                            set: { appModel.reflectiveObjectColor = $0 }
                        ), supportsOpacity: false)
                        .labelsHidden()
                    }

                    HStack {
                        Text("Miss Color")
                        Spacer()
                        ColorPicker("Miss Color", selection: Binding(
                            get: { appModel.reflectionMissColor },
                            set: { appModel.reflectionMissColor = $0 }
                        ), supportsOpacity: false)
                        .labelsHidden()
                    }

                    Text("Roughness: \(appModel.reflectiveObjectRoughness, specifier: "%.2f")")
                    Slider(value: Binding(
                        get: { appModel.reflectiveObjectRoughness },
                        set: { appModel.reflectiveObjectRoughness = $0 }
                    ), in: 0...1)

                    Text("Metallic: \(appModel.reflectiveObjectMetallic, specifier: "%.2f")")
                    Slider(value: Binding(
                        get: { appModel.reflectiveObjectMetallic },
                        set: { appModel.reflectiveObjectMetallic = $0 }
                    ), in: 0...1)
                }

                Button("Anchor") {
                    Task { @MainActor in
                        do {
                            try await appModel.anchorCurrentModel()
                        } catch {
                            errorTitle = "Couldn’t Anchor Model"
                            errorMessage = error.localizedDescription
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!appModel.canAnchorModel)

                if let anchorStatus = appModel.anchorStatus {
                    Text(anchorStatus)
                        .font(.caption)
                }
            }
        }
        .padding()
        .disabled(appModel.immersiveSpaceState == .inTransition)
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                appModel.refreshSavedAnchoredModels()
            case .background:
                exit(EXIT_SUCCESS)
            case .inactive:
                break
            @unknown default:
                break
            }
        }
        .onDisappear {
            exit(EXIT_SUCCESS)
        }
        .fileImporter(isPresented: $isImporting,
                      allowedContentTypes: [.usdz]) { result in
            Task { @MainActor in
                await openModel(result)
            }
        }
        .alert(errorTitle,
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
            errorTitle = "Couldn’t Open Model"
            let selectedURL = try result.get()

            if appModel.immersiveSpaceState == .open {
                appModel.immersiveSpaceState = .inTransition
                await dismissImmersiveSpace()
                appModel.immersiveSpaceState = .closed
            }

            try appModel.importModel(from: selectedURL)
            await openCurrentModel()
        } catch {
            appModel.immersiveSpaceState = .closed
            errorMessage = error.localizedDescription
        }
    }

    @MainActor
    private func openSavedAnchoredModel(_ savedModel: AnchoredModelRecord) async {
        do {
            errorTitle = "Couldn’t Open Anchored Room"

            if appModel.immersiveSpaceState == .open {
                appModel.immersiveSpaceState = .inTransition
                await dismissImmersiveSpace()
                appModel.immersiveSpaceState = .closed
            }

            try appModel.loadSavedAnchoredModel(savedModel)
            await openCurrentModel()
        } catch {
            appModel.immersiveSpaceState = .closed
            errorMessage = error.localizedDescription
        }
    }

    @MainActor
    private func openCurrentModel() async {
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
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
