//
//  AppModel.swift
//  roomraycast
//
//  Created by Elastic Sea on 7/22/26.
//

import SwiftUI

/// Maintains app-wide state
@MainActor
@Observable
class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"
    let modelTransform = ModelTransformState()
    let roomVisibility = RoomVisibilityState()
    let reflectiveObjectSpawns = ReflectiveObjectSpawnState()

    private var securityScopedModelURL: URL?

    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }

    var immersiveSpaceState = ImmersiveSpaceState.closed
    var isRoomVisible = true {
        didSet { roomVisibility.setVisible(isRoomVisible) }
    }
    var importedModelURL: URL?
    private(set) var anchoredModelURL: URL?
    private(set) var savedAnchoredModels: [AnchoredModelRecord] = []
    private(set) var activeAnchoredModel: AnchoredModelRecord?
    private(set) var isAnchoringAvailable = false
    private(set) var isAnchoring = false
    private(set) var anchorStatus: String?

    private var anchorAction: (@Sendable () async throws -> URL)?

    var canAnchorModel: Bool {
        importedModelURL != nil
            && isAnchoringAvailable
            && !isAnchoring
    }

    var hasSavedAnchoredModel: Bool {
        !savedAnchoredModels.isEmpty
    }

    init() {
        refreshSavedAnchoredModels()

        guard let applicationSupportURL = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                                        in: .userDomainMask,
                                                                        appropriateFor: nil,
                                                                        create: false) else {
            return
        }
        try? FileManager.default.removeItem(at: applicationSupportURL
            .appending(path: "ImportedModels", directoryHint: .isDirectory))
    }

    func spawnReflectiveObject(_ kind: ReflectiveObjectKind) {
        reflectiveObjectSpawns.request(kind)
    }

    func importModel(from sourceURL: URL) throws {
        if sourceURL == securityScopedModelURL {
            importedModelURL = sourceURL
            return
        }

        guard sourceURL.startAccessingSecurityScopedResource() else {
            throw CocoaError(.fileReadNoPermission)
        }

        securityScopedModelURL?.stopAccessingSecurityScopedResource()
        securityScopedModelURL = sourceURL
        importedModelURL = sourceURL
        anchoredModelURL = nil
        activeAnchoredModel = nil
        modelTransform.reset()
    }

    func loadSavedAnchoredModel(_ savedModel: AnchoredModelRecord) throws {
        guard savedAnchoredModels.contains(where: { $0.anchorID == savedModel.anchorID }) else {
            throw ModelAnchorError.noSavedModel
        }

        securityScopedModelURL?.stopAccessingSecurityScopedResource()
        securityScopedModelURL = nil
        importedModelURL = savedModel.modelURL
        anchoredModelURL = savedModel.modelURL
        activeAnchoredModel = savedModel
        modelTransform.restore(savedModel.transform)
    }

    func refreshSavedAnchoredModels() {
        savedAnchoredModels = AnchoredModelStore.all()

        print("[AnchoredModels] Directory: \(AnchoredModelStore.directoryURL?.path ?? "unavailable")")
        if AnchoredModelStore.storedFiles.isEmpty {
            print("[AnchoredModels] No files found")
        } else {
            for fileURL in AnchoredModelStore.storedFiles {
                print("[AnchoredModels] File: \(fileURL.lastPathComponent)")
            }
        }

        print("[AnchoredModels] Valid saved models: \(savedAnchoredModels.count)")
        for savedModel in savedAnchoredModels {
            print("[AnchoredModels] Model: \(savedModel.displayName), anchor: \(savedModel.anchorID)")
        }
    }

    func setAnchorAction(_ action: (@Sendable () async throws -> URL)?) {
        anchorAction = action
        isAnchoringAvailable = action != nil
    }

    func anchorCurrentModel() async throws {
        guard canAnchorModel, let anchorAction else {
            throw ModelAnchorError.unavailable
        }

        isAnchoring = true
        anchorStatus = "Saving…"
        print("[AnchoredModels] Anchor button pressed for: \(importedModelURL?.path ?? "unknown model")")
        defer { isAnchoring = false }

        do {
            let savedModelURL = try await anchorAction()
            securityScopedModelURL?.stopAccessingSecurityScopedResource()
            securityScopedModelURL = nil
            importedModelURL = savedModelURL
            anchoredModelURL = savedModelURL
            refreshSavedAnchoredModels()
            activeAnchoredModel = savedAnchoredModels.first(where: { $0.modelURL == savedModelURL })
            anchorStatus = "Saved as \(savedModelURL.lastPathComponent)"
            print("[AnchoredModels] Anchor completed: \(savedModelURL.path)")
        } catch {
            refreshSavedAnchoredModels()
            anchorStatus = "Anchor failed"
            print("[AnchoredModels] Anchor failed: \(error.localizedDescription)")
            throw error
        }
    }
}
