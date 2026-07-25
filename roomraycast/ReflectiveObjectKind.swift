//
//  ReflectiveObjectKind.swift
//  roomraycast
//

import Foundation

enum ReflectiveObjectKind: String, CaseIterable, Identifiable, Sendable {
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

final class ReflectiveObjectSelectionState: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ReflectiveObjectKind.sphere

    func snapshot() -> ReflectiveObjectKind {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func select(_ kind: ReflectiveObjectKind) {
        lock.lock()
        value = kind
        lock.unlock()
    }
}
