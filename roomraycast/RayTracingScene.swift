//
//  RayTracingScene.swift
//  roomraycast
//

import Metal

enum RayTracingGeometryID: Hashable, Sendable {
    case roomMesh(Int)
    case reflectiveSphere
}

enum RayTracingInstanceID: Hashable, Sendable {
    case room
    case reflectiveSphere
}

enum RayTracingSceneBuildState: Sendable, Equatable {
    case idle
    case dirty
    case building
    case ready
    case failed(String)
}

enum RayTracingSceneEvent: Hashable, Sendable {
    case geometryChanged(RayTracingGeometryID)
    case transformChanged(RayTracingInstanceID)
    case instanceAdded(RayTracingInstanceID)
    case instanceRemoved(RayTracingInstanceID)
    case materialChanged(RayTracingInstanceID)
}

struct RayTracingRebuildPlan: Sendable, Equatable {
    var bottomLevelGeometry: Set<RayTracingGeometryID> = []
    var transformedInstances: Set<RayTracingInstanceID> = []
    var addedInstances: Set<RayTracingInstanceID> = []
    var removedInstances: Set<RayTracingInstanceID> = []
    var changedMaterials: Set<RayTracingInstanceID> = []
    var rebuildTopLevel = false
    var refitTopLevel = false

    var hasWork: Bool {
        !bottomLevelGeometry.isEmpty
            || !transformedInstances.isEmpty
            || !addedInstances.isEmpty
            || !removedInstances.isEmpty
            || !changedMaterials.isEmpty
    }
}

final class RayTracingScene: @unchecked Sendable {
    let device: MTLDevice

    private(set) var bottomLevelStructures: [RayTracingGeometryID: MTLAccelerationStructure] = [:]
    private(set) var topLevelStructure: MTLAccelerationStructure?
    private(set) var instanceBuffer: MTLBuffer?
    private(set) var materialBuffer: MTLBuffer?
    private(set) var geometryBuffers: [RayTracingGeometryID: [MTLBuffer]] = [:]
    private(set) var textures: [MTLTexture] = []
    private(set) var buildState = RayTracingSceneBuildState.idle
    private var pendingEvents: Set<RayTracingSceneEvent> = []

    init(device: MTLDevice) {
        self.device = device
    }

    func setBottomLevelStructure(_ structure: MTLAccelerationStructure,
                                 for geometryID: RayTracingGeometryID) {
        bottomLevelStructures[geometryID] = structure
    }

    func setTopLevelStructure(_ structure: MTLAccelerationStructure,
                              instanceBuffer: MTLBuffer) {
        topLevelStructure = structure
        self.instanceBuffer = instanceBuffer
    }

    func registerGeometryBuffers(_ buffers: [MTLBuffer],
                                 for geometryID: RayTracingGeometryID) {
        geometryBuffers[geometryID] = buffers
    }

    func setMaterialResources(buffer: MTLBuffer, textures: [MTLTexture]) {
        materialBuffer = buffer
        self.textures = textures
    }

    func setBuildState(_ state: RayTracingSceneBuildState) {
        buildState = state
    }

    func markDirty(_ event: RayTracingSceneEvent) {
        pendingEvents.insert(event)
        buildState = .dirty
    }

    func consumeRebuildPlan() -> RayTracingRebuildPlan? {
        guard !pendingEvents.isEmpty else { return nil }

        var plan = RayTracingRebuildPlan()
        for event in pendingEvents {
            switch event {
            case .geometryChanged(let geometryID):
                plan.bottomLevelGeometry.insert(geometryID)
                plan.rebuildTopLevel = true
            case .transformChanged(let instanceID):
                plan.transformedInstances.insert(instanceID)
                plan.refitTopLevel = true
            case .instanceAdded(let instanceID):
                plan.addedInstances.insert(instanceID)
                plan.rebuildTopLevel = true
            case .instanceRemoved(let instanceID):
                plan.removedInstances.insert(instanceID)
                plan.rebuildTopLevel = true
            case .materialChanged(let instanceID):
                plan.changedMaterials.insert(instanceID)
            }
        }

        if plan.rebuildTopLevel {
            plan.refitTopLevel = false
        }

        pendingEvents.removeAll(keepingCapacity: true)
        buildState = .building
        return plan
    }

    func finishBuild() {
        buildState = pendingEvents.isEmpty ? .ready : .dirty
    }

    func failBuild(_ error: any Error) {
        buildState = .failed(error.localizedDescription)
    }
}
