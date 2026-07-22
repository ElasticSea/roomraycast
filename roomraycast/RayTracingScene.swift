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

final class RayTracingScene: @unchecked Sendable {
    let device: MTLDevice

    private(set) var bottomLevelStructures: [RayTracingGeometryID: MTLAccelerationStructure] = [:]
    private(set) var topLevelStructure: MTLAccelerationStructure?
    private(set) var instanceBuffer: MTLBuffer?
    private(set) var materialBuffer: MTLBuffer?
    private(set) var geometryBuffers: [RayTracingGeometryID: [MTLBuffer]] = [:]
    private(set) var textures: [MTLTexture] = []
    private(set) var buildState = RayTracingSceneBuildState.idle

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
}
