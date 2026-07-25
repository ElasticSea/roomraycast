//
//  RoomVisibility.swift
//  roomraycast
//

import Foundation

final class RoomVisibilityState: @unchecked Sendable {
    private let lock = NSLock()
    private var value = true

    func snapshot() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func setVisible(_ isVisible: Bool) {
        lock.lock()
        value = isVisible
        lock.unlock()
    }
}
