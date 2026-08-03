//
//  ReflectiveObjectKind.swift
//  roomraycast
//

import Foundation

nonisolated enum ReflectiveObjectKind: String, CaseIterable, Identifiable, Sendable {
    case sphere
    case monkey
    case teapot

    var id: Self { self }

    var title: String {
        rawValue.capitalized
    }

    var resourceName: String? {
        switch self {
        case .sphere: nil
        case .monkey: "monkey"
        case .teapot: "teapot"
        }
    }
}

nonisolated final class ReflectiveObjectSpawnState: @unchecked Sendable {
    private let lock = NSLock()
    private var pendingKinds: [ReflectiveObjectKind] = []

    func request(_ kind: ReflectiveObjectKind) {
        lock.lock()
        pendingKinds.append(kind)
        lock.unlock()
    }

    func consumeAll() -> [ReflectiveObjectKind] {
        lock.lock()
        defer { lock.unlock() }
        let kinds = pendingKinds
        pendingKinds.removeAll(keepingCapacity: true)
        return kinds
    }
}
