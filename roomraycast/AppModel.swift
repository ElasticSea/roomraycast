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

    private var securityScopedModelURL: URL?

    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }

    var immersiveSpaceState = ImmersiveSpaceState.closed
    var importedModelURL: URL?

    init() {
        guard let applicationSupportURL = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                                        in: .userDomainMask,
                                                                        appropriateFor: nil,
                                                                        create: false) else {
            return
        }
        try? FileManager.default.removeItem(at: applicationSupportURL
            .appending(path: "ImportedModels", directoryHint: .isDirectory))
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
    }
}
