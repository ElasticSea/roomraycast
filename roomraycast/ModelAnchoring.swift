//
//  ModelAnchoring.swift
//  roomraycast
//

import Foundation

enum ModelAnchorError: LocalizedError {
    case unavailable
    case noSavedModel

    var errorDescription: String? {
        switch self {
        case .unavailable:
            "Anchoring is not available until the room is loaded and world tracking is running."
        case .noSavedModel:
            "No saved anchored room is available."
        }
    }
}

struct AnchoredModelRecord: Sendable {
    let modelURL: URL
    let anchorID: UUID
    let transform: ModelTransformSnapshot
    let createdAt: Date

    var displayName: String {
        modelURL.deletingPathExtension().lastPathComponent
    }
}

enum AnchoredModelStore {
    private struct Placement: Codable {
        let anchorID: UUID
        let modelFileName: String?
        let x: Float
        let y: Float
        let z: Float
        let yaw: Float
        let scale: Float
        let createdAt: Date
    }

    static var directoryURL: URL? {
        guard let applicationSupportURL = try? FileManager.default.url(for: .applicationSupportDirectory,
                                                                       in: .userDomainMask,
                                                                       appropriateFor: nil,
                                                                       create: false) else {
            return nil
        }
        return applicationSupportURL
            .appending(path: "AnchoredModels", directoryHint: .isDirectory)
    }

    static var storedFiles: [URL] {
        guard let directoryURL,
              let files = try? FileManager.default.contentsOfDirectory(at: directoryURL,
                                                                       includingPropertiesForKeys: nil,
                                                                       options: [.skipsHiddenFiles]) else {
            return []
        }
        return files.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func save(modelAt sourceURL: URL,
                     anchorID: UUID,
                     transform: ModelTransformSnapshot,
                     replacing existingRecord: AnchoredModelRecord?) throws -> AnchoredModelRecord {
        let fileManager = FileManager.default
        let applicationSupportURL = try fileManager.url(for: .applicationSupportDirectory,
                                                        in: .userDomainMask,
                                                        appropriateFor: nil,
                                                        create: true)
        let anchoredModelsURL = applicationSupportURL
            .appending(path: "AnchoredModels", directoryHint: .isDirectory)
        try fileManager.createDirectory(at: anchoredModelsURL,
                                        withIntermediateDirectories: true)

        let destinationURL = existingRecord?.modelURL
            ?? uniqueModelURL(for: sourceURL,
                              in: anchoredModelsURL,
                              fileManager: fileManager)
        let placementURL = destinationURL
            .deletingPathExtension()
            .appendingPathExtension("anchor.json")
        let createdAt = Date()

        if existingRecord == nil {
            print("[AnchoredModels] Copy starting: \(sourceURL.path) -> \(destinationURL.path)")
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            print("[AnchoredModels] Copy finished: \(destinationURL.path)")
        } else {
            print("[AnchoredModels] Overwriting anchor metadata for: \(destinationURL.path)")
        }

        do {
            let placement = Placement(anchorID: anchorID,
                                      modelFileName: destinationURL.lastPathComponent,
                                      x: transform.translation.x,
                                      y: transform.translation.y,
                                      z: transform.translation.z,
                                      yaw: transform.yaw,
                                      scale: transform.scale,
                                      createdAt: createdAt)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(placement).write(to: placementURL, options: .atomic)
            print("[AnchoredModels] Metadata saved: \(placementURL.path)")
        } catch {
            if existingRecord == nil {
                try? fileManager.removeItem(at: destinationURL)
            }
            throw error
        }

        return AnchoredModelRecord(modelURL: destinationURL,
                                   anchorID: anchorID,
                                   transform: transform,
                                   createdAt: createdAt)
    }

    static func all() -> [AnchoredModelRecord] {
        let fileManager = FileManager.default
        guard let anchoredModelsURL = directoryURL else {
            return []
        }

        let decoder = JSONDecoder()
        return storedFiles
            .filter { $0.lastPathComponent.hasSuffix(".anchor.json") }
            .compactMap { placementURL -> AnchoredModelRecord? in
                guard let data = try? Data(contentsOf: placementURL),
                      let placement = try? decoder.decode(Placement.self, from: data) else {
                    return nil
                }

                let fallbackName = placementURL
                    .deletingPathExtension()
                    .deletingPathExtension()
                    .lastPathComponent + ".usdz"
                let modelURL = anchoredModelsURL
                    .appending(path: placement.modelFileName ?? fallbackName)
                guard fileManager.fileExists(atPath: modelURL.path) else {
                    return nil
                }

                return AnchoredModelRecord(modelURL: modelURL,
                                           anchorID: placement.anchorID,
                                           transform: ModelTransformSnapshot(
                                            translation: SIMD3<Float>(placement.x,
                                                                      placement.y,
                                                                      placement.z),
                                            yaw: placement.yaw,
                                            scale: placement.scale),
                                           createdAt: placement.createdAt)
            }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private static func uniqueModelURL(for sourceURL: URL,
                                       in directoryURL: URL,
                                       fileManager: FileManager) -> URL {
        let originalName = sourceURL.deletingPathExtension().lastPathComponent
        let baseName = originalName.isEmpty ? "Room" : originalName
        let fileExtension = sourceURL.pathExtension.isEmpty ? "usdz" : sourceURL.pathExtension
        var copyNumber = 1

        while true {
            let name = copyNumber == 1 ? baseName : "\(baseName) \(copyNumber)"
            let candidateURL = directoryURL
                .appending(path: name)
                .appendingPathExtension(fileExtension)
            let placementURL = candidateURL
                .deletingPathExtension()
                .appendingPathExtension("anchor.json")

            if !fileManager.fileExists(atPath: candidateURL.path)
                && !fileManager.fileExists(atPath: placementURL.path) {
                return candidateURL
            }

            copyNumber += 1
        }
    }
}
