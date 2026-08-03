//
//  Renderer.swift
//  roomraycast
//
//  Created by Elastic Sea on 7/22/26.
//

import CompositorServices
import Metal
import MetalKit
import ModelIO
import simd

// The 256 byte aligned size of our uniform structure
nonisolated let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100
nonisolated let alignedViewProjectionArraySize = (MemoryLayout<ViewProjectionArray>.size + 0xFF) & -0x100

nonisolated let maxBuffersInFlight = 3
nonisolated let maxRayTracingTextures = 64

enum RendererError: Error {
    case badVertexDescriptor
    case missingReflectiveObjectAsset(String)
}

extension MTLDevice {
    nonisolated var supportsMSAA: Bool {
        supports32BitMSAA && supportsTextureSampleCount(4)
    }

    nonisolated var rasterSampleCount: Int {
        supportsMSAA ? 4 : 1
    }
}

extension LayerRenderer.Clock.Instant {
    nonisolated var timeInterval: TimeInterval {
        let components = LayerRenderer.Clock.Instant.epoch.duration(to: self).components
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

final class RendererTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
          job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    nonisolated func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        return UnownedTaskExecutor(ordinary: self)
    }

    static var shared: RendererTaskExecutor = RendererTaskExecutor()
}

actor Renderer {

    struct RenderMesh {
        let mesh: MTKMesh
        let assetTransform: matrix_float4x4
        let baseColorTextures: [MTLTexture?]
    }

    struct LoadedModel {
        let meshes: [RenderMesh]
        let normalizationTransform: matrix_float4x4
        let rayTracingHitData: RayTracingCPUHitData
    }

    let device: MTLDevice
    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let vertexArgumentTable: MTL4ArgumentTable
    let fragmentArgumentTable: MTL4ArgumentTable
    #if !targetEnvironment(simulator)
    let residencySets: [MTLResidencySet]
    let commandQueueResidencySet: MTLResidencySet
    #endif

    let dynamicUniformBuffer: MTLBuffer
    let reflectiveMaterialBuffer: MTLBuffer
    let rayTracingSettingsBuffer: MTLBuffer
    let uniformsPerFrameSize: Int
    let pipelineState: MTLRenderPipelineState
    let reflectiveSpherePipelineState: MTLRenderPipelineState
    let rayTracedReflectiveSpherePipelineState: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let colorMap: MTLTexture

    let endFrameEvent: MTLSharedEvent
    var committedFrameIndex: UInt64 = 0

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var perDrawableTarget = [LayerRenderer.Drawable.Target: DrawableTarget]()

    let meshes: [RenderMesh]
    let modelNormalizationTransform: matrix_float4x4
    let reflectiveObjectMeshes: [ReflectiveObjectKind: ReflectiveObjectMesh]
    let rayTracingScene: RayTracingScene
    let rayTracingAccelerationBuilder: RayTracingAccelerationBuilder
    let rayTracingHitDataBuffers: RayTracingHitDataBuffers

    let worldTracking: WorldTrackingProvider
    let handTracking: HandTrackingProvider
    let layerRenderer: LayerRenderer
    let appModel: AppModel
    let modelTransform: ModelTransformState
    let roomVisibility: RoomVisibilityState
    let reflectiveObjectSpawns: ReflectiveObjectSpawnState
    let reflectionBounceCountState: ReflectionBounceCountState
    let reflectiveObjectMaterialState: ReflectiveObjectMaterialState
    let modelURL: URL
    var savedRecord: AnchoredModelRecord?
    var reflectiveObjects: [ReflectiveObject] = []
    var pendingReflectiveObjectKinds: [ReflectiveObjectKind] = []
    var reflectiveObjectGrabController = ReflectiveObjectGrabController()
    var reflectiveObjectPlacement = ReflectiveObjectPlacement()
    var leftHandPinchTracker = HandPinchTracker(side: .left)
    var rightHandPinchTracker = HandPinchTracker(side: .right)
    var leftHandPinchFrame = HandPinchFrame.unavailable
    var rightHandPinchFrame = HandPinchFrame.unavailable
    var lastRayTracingRoomTransform: ModelTransformSnapshot?
    var isRayTracingRoomTransformDirty = true
    var isRayTracingBuildInFlight = false
    var areRayTracingResourcesBound = false
    var anchoredPlacementTransform: matrix_float4x4?
    var anchoredScale: Float = 1

    init(_ layerRenderer: LayerRenderer,
         appModel: AppModel,
         modelURL: URL,
         modelTransform: ModelTransformState,
         roomVisibility: RoomVisibilityState,
         reflectiveObjectSpawns: ReflectiveObjectSpawnState,
         reflectionBounceCountState: ReflectionBounceCountState,
         reflectiveObjectMaterialState: ReflectiveObjectMaterialState,
         restoredAnchor: AnchoredModelRecord?) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.appModel = appModel
        self.modelTransform = modelTransform
        self.roomVisibility = roomVisibility
        self.reflectiveObjectSpawns = reflectiveObjectSpawns
        self.reflectionBounceCountState = reflectionBounceCountState
        self.reflectiveObjectMaterialState = reflectiveObjectMaterialState
        self.modelURL = modelURL
        self.savedRecord = restoredAnchor

        let device = self.device
        self.reflectiveObjectMeshes = try! ReflectiveObjectMesh.makeAll(device: device)
        self.rayTracingScene = RayTracingScene(device: device)
        self.rayTracingAccelerationBuilder = try! RayTracingAccelerationBuilder(device: device)
        self.commandQueue = layerRenderer.commandQueue
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0...maxBuffersInFlight).map { _ in device.makeCommandAllocator()! }

        let argTableDesc = MTL4ArgumentTableDescriptor()
        argTableDesc.maxBufferBindCount = 4
        self.vertexArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)
        argTableDesc.maxBufferBindCount = BufferIndex.rayTracingSettings.rawValue + 1
        argTableDesc.maxTextureBindCount = TextureIndex.rayTracingBase.rawValue + maxRayTracingTextures
        self.fragmentArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)

        #if !targetEnvironment(simulator)
        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = 3 // color + depth + view projection buffer
        self.residencySets = (0...maxBuffersInFlight).map { _ in try! device.makeResidencySet(descriptor: residencySetDesc) }
        #endif

        self.endFrameEvent = device.makeSharedEvent()!
        // Start the signal value + committed frames index at
        // max buffers in flight to avoid negative values
        self.endFrameEvent.signaledValue = UInt64(maxBuffersInFlight)
        committedFrameIndex = UInt64(maxBuffersInFlight)

        let mtlVertexDescriptor = Self.buildMetalVertexDescriptor()

        do {
            let loadedModel = try Self.loadModel(device: device,
                                                 modelURL: modelURL,
                                                 mtlVertexDescriptor: mtlVertexDescriptor,
                                                 reflectiveObjectMeshes: reflectiveObjectMeshes)
            meshes = loadedModel.meshes
            modelNormalizationTransform = loadedModel.normalizationTransform
            rayTracingHitDataBuffers = try! RayTracingHitDataBuffers(device: device,
                                                                     data: loadedModel.rayTracingHitData)
            rayTracingScene.setHitDataBuffers(rayTracingHitDataBuffers)
        } catch {
            fatalError("Unable to load imported model. Error info: \(error)")
        }

        for (meshIndex, renderMesh) in meshes.enumerated() {
            let geometryID = RayTracingGeometryID.roomMesh(meshIndex)
            let geometrySource = try! RayTracingTriangleGeometrySource(id: geometryID,
                                                                       mesh: renderMesh.mesh)
            let buffers = renderMesh.mesh.vertexBuffers.map { $0.buffer }
                + renderMesh.mesh.submeshes.map { $0.indexBuffer.buffer }
            rayTracingScene.registerGeometrySource(geometrySource)
            rayTracingScene.registerGeometryBuffers(buffers, for: geometryID)
            rayTracingScene.markDirty(.geometryChanged(geometryID))
            rayTracingScene.markDirty(.instanceAdded(.roomMesh(meshIndex)))
        }
        for kind in ReflectiveObjectKind.allCases {
            let geometryID = RayTracingGeometryID.reflectiveObject(kind)
            guard let reflectiveObjectMesh = reflectiveObjectMeshes[kind] else { continue }
            let geometrySource = try! RayTracingTriangleGeometrySource(
                id: geometryID,
                mesh: reflectiveObjectMesh.mesh)
            let buffers = reflectiveObjectMesh.mesh.vertexBuffers.map { $0.buffer }
                + reflectiveObjectMesh.mesh.submeshes.map { $0.indexBuffer.buffer }
            rayTracingScene.registerGeometrySource(geometrySource)
            rayTracingScene.registerGeometryBuffers(buffers, for: geometryID)
            rayTracingScene.markDirty(.geometryChanged(geometryID))
        }
        uniformsPerFrameSize = alignedUniformsSize
            * max(meshes.count + ReflectiveObject.maximumCount, 1)
        let uniformBufferSize = uniformsPerFrameSize * maxBuffersInFlight
        self.dynamicUniformBuffer = self.device.makeBuffer(length: uniformBufferSize,
                                                           options: [MTLResourceOptions.storageModeShared])!
        self.dynamicUniformBuffer.label = "UniformBuffer"

        self.reflectiveMaterialBuffer = self.device.makeBuffer(
            length: MemoryLayout<PureReflectionMaterialUniforms>.stride,
            options: [MTLResourceOptions.storageModeShared])!
        self.reflectiveMaterialBuffer.label = "Pure Reflection Material"
        let materialPointer = self.reflectiveMaterialBuffer.contents()
            .bindMemory(to: PureReflectionMaterialUniforms.self, capacity: 1)
        let material = PureReflectionMaterial()
        materialPointer.pointee.reflectivity = material.reflectivity
        materialPointer.pointee.roughness = material.roughness
        materialPointer.pointee.metallic = material.metallic
        materialPointer.pointee.diffuseContribution = material.diffuseContribution
        materialPointer.pointee.baseColor = SIMD4<Float>(material.baseColor, 1)
        materialPointer.pointee.missColor = SIMD4<Float>(material.missColor, 1)

        self.rayTracingSettingsBuffer = self.device.makeBuffer(
            length: MemoryLayout<RayTracingSettings>.stride,
            options: [MTLResourceOptions.storageModeShared])!
        self.rayTracingSettingsBuffer.label = "Ray Tracing Settings"
        let raySettings = self.rayTracingSettingsBuffer.contents()
            .bindMemory(to: RayTracingSettings.self, capacity: 1)
        raySettings.pointee.maxBounces = reflectionBounceCountState.snapshot()
        raySettings.pointee.maxDistance = 30
        raySettings.pointee.rayEpsilon = 0.002
        raySettings.pointee.padding = 0

        do {
            pipelineState = try Self.buildRenderPipeline(device: device,
                                                         layerRenderer: layerRenderer,
                                                         mtlVertexDescriptor: mtlVertexDescriptor)
            reflectiveSpherePipelineState = try Self.buildReflectiveSpherePipeline(
                device: device,
                layerRenderer: layerRenderer,
                vertexDescriptor: ReflectiveSphereMesh.metalVertexDescriptor(),
                fragmentFunctionName: "reflectiveSphereFragment")
            rayTracedReflectiveSpherePipelineState = try Self.buildReflectiveSpherePipeline(
                device: device,
                layerRenderer: layerRenderer,
                vertexDescriptor: ReflectiveSphereMesh.metalVertexDescriptor(),
                fragmentFunctionName: "rayTracedReflectiveSphereFragment")
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }

        self.depthState = Self.buildDepthStencilState(device: device)

        let fallbackColorMap: MTLTexture
        do {
            fallbackColorMap = try Self.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            fatalError("Unable to load texture. Error info: \(error)")
        }
        self.colorMap = fallbackColorMap
        let rayTracingTextures = meshes.flatMap { renderMesh in
            renderMesh.baseColorTextures.map { $0 ?? fallbackColorMap }
        }
        rayTracingScene.setMaterialResources(buffer: reflectiveMaterialBuffer,
                                             textures: rayTracingTextures)
        for meshIndex in meshes.indices {
            rayTracingScene.markDirty(.materialChanged(.roomMesh(meshIndex)))
        }

        #if !targetEnvironment(simulator)
        // Add all persistent resources to the command queue residency set,
        // must be done after loading all resources.
        let vertexBuffers = meshes.flatMap { renderMesh in
            renderMesh.mesh.vertexBuffers.map { $0.buffer }
        }
        let indexBuffers = meshes.flatMap { renderMesh in
            renderMesh.mesh.submeshes.map { $0.indexBuffer.buffer }
        }
        let modelTextures = meshes.flatMap { renderMesh in
            renderMesh.baseColorTextures.compactMap { $0 }
        }
        let reflectiveObjectVertexBuffers = reflectiveObjectMeshes.values.flatMap {
            $0.mesh.vertexBuffers.map(\.buffer)
        }
        let reflectiveObjectIndexBuffers = reflectiveObjectMeshes.values.flatMap {
            $0.mesh.submeshes.map { $0.indexBuffer.buffer }
        }
        residencySetDesc.initialCapacity = vertexBuffers.count
            + indexBuffers.count
            + modelTextures.count
            + reflectiveObjectVertexBuffers.count
            + reflectiveObjectIndexBuffers.count
            + 9
        let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
        residencySet.addAllocations(vertexBuffers)
        residencySet.addAllocations(indexBuffers)
        residencySet.addAllocations(modelTextures)
        residencySet.addAllocations(reflectiveObjectVertexBuffers)
        residencySet.addAllocations(reflectiveObjectIndexBuffers)
        residencySet.addAllocations([
            colorMap,
            dynamicUniformBuffer,
            reflectiveMaterialBuffer,
            rayTracingSettingsBuffer,
            rayTracingHitDataBuffers.vertices,
            rayTracingHitDataBuffers.indices,
            rayTracingHitDataBuffers.geometries,
            rayTracingHitDataBuffers.instances,
            rayTracingHitDataBuffers.objects
        ])
        residencySet.commit()
        commandQueueResidencySet = residencySet
        commandQueue.addResidencySet(residencySet)
        #endif

        worldTracking = WorldTrackingProvider()
        handTracking = HandTrackingProvider()
    }

    private func startARSession(_ arSession: ARKitSession) async {
        do {
            try await arSession.run([worldTracking, handTracking])
            await restoreSavedAnchorIfAvailable()
            await MainActor.run {
                appModel.setAnchorAction { [self] in
                    try await anchorCurrentModel()
                }
            }
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }

    private func restoreSavedAnchorIfAvailable() async {
        guard let savedRecord,
              let anchors = await worldTracking.allAnchors,
              let worldAnchor = anchors.first(where: { $0.id == savedRecord.anchorID }) else {
            return
        }

        anchoredPlacementTransform = worldAnchor.originFromAnchorTransform
        anchoredScale = savedRecord.transform.scale
        isRayTracingRoomTransformDirty = true
    }

    private func anchorCurrentModel() async throws -> URL {
        guard worldTracking.state == .running else {
            throw ModelAnchorError.unavailable
        }

        let adjustment = modelTransform.snapshot()
        let anchorTransform = Self.placementTransform(for: adjustment, includeScale: false)
        let anchor = WorldAnchor(originFromAnchorTransform: anchorTransform)
        let replacedRecord = savedRecord
        let newRecord = try AnchoredModelStore.save(modelAt: modelURL,
                                                    anchorID: anchor.id,
                                                    transform: adjustment,
                                                    replacing: replacedRecord)
        savedRecord = newRecord
        anchoredPlacementTransform = nil
        print("[AnchoredModels] Saved model before world-anchor registration: \(newRecord.modelURL.path)")

        do {
            try await worldTracking.addAnchor(anchor)
            anchoredPlacementTransform = anchorTransform
            anchoredScale = adjustment.scale
            isRayTracingRoomTransformDirty = true
            if let replacedRecord, replacedRecord.anchorID != anchor.id {
                try? await worldTracking.removeAnchor(forID: replacedRecord.anchorID)
            }
            return newRecord.modelURL
        } catch {
            print("[AnchoredModels] Model was saved, but world-anchor registration failed: \(error.localizedDescription)")
            throw error
        }
    }

    @MainActor
    static func startRenderLoop(_ layerRenderer: LayerRenderer,
                                appModel: AppModel,
                                arSession: ARKitSession,
                                modelURL: URL) {
        let modelTransform = appModel.modelTransform
        let roomVisibility = appModel.roomVisibility
        let reflectiveObjectSpawns = appModel.reflectiveObjectSpawns
        let reflectionBounceCountState = appModel.reflectionBounceCountState
        let reflectiveObjectMaterialState = appModel.reflectiveObjectMaterialState
        let restoredAnchor = appModel.activeAnchoredModel
        Task(executorPreference: RendererTaskExecutor.shared) {
            let renderer = Renderer(layerRenderer,
                                    appModel: appModel,
                                    modelURL: modelURL,
                                    modelTransform: modelTransform,
                                    roomVisibility: roomVisibility,
                                    reflectiveObjectSpawns: reflectiveObjectSpawns,
                                    reflectionBounceCountState: reflectionBounceCountState,
                                    reflectiveObjectMaterialState: reflectiveObjectMaterialState,
                                    restoredAnchor: restoredAnchor)
            await renderer.startARSession(arSession)
            await renderer.renderLoop()
        }
    }

    static func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    static func buildRenderPipeline(device: MTLDevice,
                                    layerRenderer: LayerRenderer,
                                    mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.rasterSampleCount = device.rasterSampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    static func buildDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptor.isDepthWriteEnabled = true
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    static func buildReflectiveSpherePipeline(device: MTLDevice,
                                              layerRenderer: LayerRenderer,
                                              vertexDescriptor: MTLVertexDescriptor,
                                              fragmentFunctionName: String) throws -> MTLRenderPipelineState {
        let library = device.makeDefaultLibrary()
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Pure Reflection Sphere Pipeline"
        descriptor.vertexFunction = library?.makeFunction(name: "reflectiveSphereVertex")
        descriptor.fragmentFunction = library?.makeFunction(name: fragmentFunctionName)
        descriptor.vertexDescriptor = vertexDescriptor
        descriptor.rasterSampleCount = device.rasterSampleCount
        descriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        descriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat
        descriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }

    static func loadModel(device: MTLDevice,
                          modelURL: URL,
                          mtlVertexDescriptor: MTLVertexDescriptor,
                          reflectiveObjectMeshes: [ReflectiveObjectKind: ReflectiveObjectMesh]) throws -> LoadedModel {
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate

        let asset = MDLAsset(url: modelURL,
                             vertexDescriptor: mdlVertexDescriptor,
                             bufferAllocator: metalAllocator)
        asset.loadTextures()

        let convertedMeshes = try MTKMesh.newMeshes(asset: asset, device: device)
        guard !convertedMeshes.metalKitMeshes.isEmpty else {
            throw RendererError.badVertexDescriptor
        }

        let textureLoader = MTKTextureLoader(device: device)
        let textureOptions: [MTKTextureLoader.Option: Any] = [
            .generateMipmaps: true,
            .origin: MTKTextureLoader.Origin.bottomLeft,
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
        ]

        let renderMeshes = zip(convertedMeshes.modelIOMeshes,
                               convertedMeshes.metalKitMeshes).map { modelIOMesh, metalKitMesh in
            let modelIOSubmeshes = (modelIOMesh.submeshes as? [MDLSubmesh]) ?? []
            let baseColorTextures = modelIOSubmeshes.map { submesh in
                Self.loadBaseColorTexture(material: submesh.material,
                                          textureLoader: textureLoader,
                                          options: textureOptions)
            }

            return RenderMesh(mesh: metalKitMesh,
                              assetTransform: MDLTransform.globalTransform(with: modelIOMesh, atTime: 0),
                              baseColorTextures: baseColorTextures)
        }

        let bounds = asset.boundingBox
        let minimum = bounds.minBounds
        let maximum = bounds.maxBounds
        let size = maximum - minimum
        let largestDimension = max(size.x, max(size.y, size.z))
        let safeDimension = largestDimension.isFinite && largestDimension > 0.0001 ? largestDimension : 1
        let scale = 2.0 / safeDimension
        let center = (minimum + maximum) * 0.5

        let normalizationTransform = matrix4x4_scale(scale)
            * matrix4x4_translation(-center.x, -center.y, -center.z)

        let rayTracingHitData = try RayTracingCPUHitData.make(
            roomMeshes: convertedMeshes.modelIOMeshes,
            reflectiveObjectMeshes: reflectiveObjectMeshes,
            reflectiveObjectCapacity: ReflectiveObject.maximumCount)
        return LoadedModel(meshes: renderMeshes,
                           normalizationTransform: normalizationTransform,
                           rayTracingHitData: rayTracingHitData)
    }

    static func loadBaseColorTexture(material: MDLMaterial?,
                                     textureLoader: MTKTextureLoader,
                                     options: [MTKTextureLoader.Option: Any]) -> MTLTexture? {
        guard let property = material?.property(with: .baseColor) else { return nil }

        if let modelTexture = property.textureSamplerValue?.texture {
            return try? textureLoader.newTexture(texture: modelTexture, options: options)
        }
        if let textureURL = property.urlValue {
            return try? textureLoader.newTexture(URL: textureURL, options: options)
        }
        return nil
    }

    static func loadTexture(device: MTLDevice,
                            textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling

        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]

        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
    }

    private func updateDynamicBufferState(frameIndex: UInt64) {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = uniformsPerFrameSize * uniformBufferIndex

        /// Reset resources used in previous frame

        #if !targetEnvironment(simulator)
        residencySets[uniformBufferIndex].removeAllAllocations()
        residencySets[uniformBufferIndex].commit()
        #endif
        commandAllocators[uniformBufferIndex].reset()

        /// Remove all per drawable target resources that are older than 90 frames

        perDrawableTarget = perDrawableTarget.filter { $0.value.lastUsedFrameIndex + 90 > frameIndex }
    }

    private func updateGameState() {
        let material = reflectiveObjectMaterialState.snapshot()
        let materialPointer = reflectiveMaterialBuffer.contents()
            .bindMemory(to: PureReflectionMaterialUniforms.self, capacity: 1)
        materialPointer.pointee.reflectivity = material.reflectivity
        materialPointer.pointee.roughness = material.roughness
        materialPointer.pointee.metallic = material.metallic
        materialPointer.pointee.diffuseContribution = material.diffuseContribution
        materialPointer.pointee.baseColor = SIMD4<Float>(material.baseColor, 1)
        materialPointer.pointee.missColor = SIMD4<Float>(material.missColor, 1)

        let raySettings = rayTracingSettingsBuffer.contents()
            .bindMemory(to: RayTracingSettings.self, capacity: 1)
        raySettings.pointee.maxBounces = reflectionBounceCountState.snapshot()

        pendingReflectiveObjectKinds.append(contentsOf: reflectiveObjectSpawns.consumeAll())
        spawnPendingReflectiveObjectsIfPossible()

        let adjustment = modelTransform.snapshot()
        let roomTransformChanged = isRayTracingRoomTransformDirty
            || lastRayTracingRoomTransform != adjustment
        let placementTransform = if let anchoredPlacementTransform,
                                    let savedRecord,
                                    adjustment == savedRecord.transform {
            anchoredPlacementTransform * matrix4x4_scale(anchoredScale)
        } else {
            Self.placementTransform(for: adjustment, includeScale: true)
        }

        for (meshIndex, renderMesh) in meshes.enumerated() {
            let pointer = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()
                                                  + uniformBufferOffset
                                                  + alignedUniformsSize * meshIndex)
                .bindMemory(to: Uniforms.self, capacity: 1)
            pointer[0].modelMatrix = placementTransform
                * modelNormalizationTransform
                * renderMesh.assetTransform
            if roomTransformChanged {
                let instanceID = RayTracingInstanceID.roomMesh(meshIndex)
                rayTracingScene.setInstanceTransform(pointer[0].modelMatrix, for: instanceID)
                rayTracingHitDataBuffers.setObjectTransform(pointer[0].modelMatrix,
                                                            sphereRadius: 0,
                                                            at: meshIndex)
                rayTracingScene.markDirty(.transformChanged(instanceID))
            }
        }
        lastRayTracingRoomTransform = adjustment
        isRayTracingRoomTransformDirty = false

        for object in reflectiveObjects {
            let objectIndex = meshes.count + object.id.rawValue
            let objectPointer = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()
                                                        + uniformBufferOffset
                                                        + alignedUniformsSize * objectIndex)
                .bindMemory(to: Uniforms.self, capacity: 1)
            objectPointer[0].modelMatrix = reflectiveObjectTransform(for: object)
        }
    }

    private func spawnPendingReflectiveObjectsIfPossible() {
        guard !pendingReflectiveObjectKinds.isEmpty,
              let spawnTransform = reflectiveObjectPlacement.makeSpawnTransform() else {
            return
        }

        let availableCount = ReflectiveObject.maximumCount - reflectiveObjects.count
        guard availableCount > 0 else {
            print("[ReflectiveObjects] Maximum object count reached")
            pendingReflectiveObjectKinds.removeAll(keepingCapacity: true)
            return
        }

        let spawnKinds = pendingReflectiveObjectKinds.prefix(availableCount)
        pendingReflectiveObjectKinds.removeFirst(spawnKinds.count)
        for kind in spawnKinds {
            let id = ReflectiveObjectID(rawValue: reflectiveObjects.count)
            let object = ReflectiveObject(id: id,
                                          kind: kind,
                                          originFromObjectTransform: spawnTransform)
            reflectiveObjects.append(object)

            let instanceID = RayTracingInstanceID.reflectiveObject(id)
            let objectTransform = reflectiveObjectTransform(for: object)
            rayTracingScene.setInstanceTransform(objectTransform, for: instanceID)
            rayTracingHitDataBuffers.setObjectTransform(
                objectTransform,
                sphereRadius: ReflectiveObject.radius,
                at: meshes.count + id.rawValue)
            rayTracingHitDataBuffers.setReflectiveObjectKind(
                kind,
                at: meshes.count + id.rawValue)
            rayTracingScene.markDirty(.instanceAdded(instanceID))
        }

        if !pendingReflectiveObjectKinds.isEmpty {
            print("[ReflectiveObjects] Maximum object count reached")
            pendingReflectiveObjectKinds.removeAll(keepingCapacity: true)
        }
    }

    private func reflectiveObjectTransform(for object: ReflectiveObject) -> matrix_float4x4 {
        guard let reflectiveObjectMesh = reflectiveObjectMeshes[object.kind] else {
            return object.originFromObjectTransform
        }
        return object.originFromObjectTransform * reflectiveObjectMesh.localTransform
    }

    nonisolated private static func placementTransform(for adjustment: ModelTransformSnapshot,
                                                       includeScale: Bool) -> matrix_float4x4 {
        let rigidTransform = matrix4x4_translation(adjustment.translation.x,
                                                   adjustment.translation.y,
                                                   adjustment.translation.z - 2.5)
            * matrix4x4_rotation(radians: adjustment.yaw, axis: SIMD3<Float>(0, 1, 0))
        return includeScale ? rigidTransform * matrix4x4_scale(adjustment.scale) : rigidTransform
    }

    func renderFrame() {
        /// Per frame updates hare

        guard let frame = layerRenderer.queryNextFrame() else { return }

        guard self.endFrameEvent.wait(untilSignaledValue: committedFrameIndex - UInt64(maxBuffersInFlight), timeoutMS: 10000) else {
            return
        }

        frame.startUpdate()

        // Perform frame independent work

        self.updateDynamicBufferState(frameIndex: frame.frameIndex)

        self.updateGameState()
        rayTracingScene.schedulePendingEvents()
        self.scheduleRayTracingAccelerationStructureUpdateIfNeeded()

        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }

        frame.startSubmission()

        for drawable in drawables {
            render(drawable: drawable, frameIndex: frame.frameIndex)
        }

        committedFrameIndex += 1

        commandQueue.signalEvent(self.endFrameEvent, value: committedFrameIndex)

        frame.endSubmission()
    }

    private func scheduleRayTracingAccelerationStructureUpdateIfNeeded() {
        guard !isRayTracingBuildInFlight else { return }

        var instances: [RayTracingInstanceSource] = []
        for meshIndex in meshes.indices {
            let instanceID = RayTracingInstanceID.roomMesh(meshIndex)
            guard let transform = rayTracingScene.instanceTransforms[instanceID] else { return }
            instances.append(RayTracingInstanceSource(id: instanceID,
                                                      geometryID: .roomMesh(meshIndex),
                                                      transform: transform,
                                                      mask: 0x1))
        }
        for object in reflectiveObjects {
            let instanceID = RayTracingInstanceID.reflectiveObject(object.id)
            guard let transform = rayTracingScene.instanceTransforms[instanceID] else { return }
            instances.append(RayTracingInstanceSource(
                id: instanceID,
                geometryID: .reflectiveObject(object.kind),
                transform: transform,
                mask: 0x2))
        }

        guard let plan = rayTracingScene.consumeRebuildPlan() else { return }

        isRayTracingBuildInFlight = true
        Task { [self] in
            await performRayTracingAccelerationStructureUpdate(plan: plan,
                                                               instances: instances)
        }
    }

    private func performRayTracingAccelerationStructureUpdate(
        plan: RayTracingRebuildPlan,
        instances: [RayTracingInstanceSource]
    ) async {
        defer { isRayTracingBuildInFlight = false }
        do {
            if !plan.bottomLevelGeometry.isEmpty {
                let sources = plan.bottomLevelGeometry.compactMap {
                    rayTracingScene.geometrySources[$0]
                }
                let rebuiltStructures = try await rayTracingAccelerationBuilder
                    .buildBottomLevelStructures(for: sources)
                for (geometryID, structure) in rebuiltStructures {
                    rayTracingScene.setBottomLevelStructure(structure, for: geometryID)
                }
            }

            let needsFullTopLevelBuild = plan.rebuildTopLevel
                || !plan.bottomLevelGeometry.isEmpty
                || rayTracingScene.topLevelStructure == nil
            let topLevelBuild: RayTracingTopLevelBuild
            if !needsFullTopLevelBuild,
               plan.refitTopLevel,
               let existingTopLevel = rayTracingScene.topLevelStructure {
                topLevelBuild = try await rayTracingAccelerationBuilder.refitTopLevelStructure(
                    existingTopLevel,
                    instances: instances,
                    bottomLevelStructures: rayTracingScene.bottomLevelStructures)
            } else {
                topLevelBuild = try await rayTracingAccelerationBuilder.buildTopLevelStructure(
                    instances: instances,
                    bottomLevelStructures: rayTracingScene.bottomLevelStructures)
            }
            rayTracingScene.setTopLevelStructure(topLevelBuild.structure,
                                                 instanceBuffer: topLevelBuild.instanceBuffer)
            #if !targetEnvironment(simulator)
            for structure in rayTracingScene.bottomLevelStructures.values {
                commandQueueResidencySet.addAllocation(structure)
            }
            commandQueueResidencySet.addAllocation(topLevelBuild.structure)
            commandQueueResidencySet.addAllocation(topLevelBuild.instanceBuffer)
            commandQueueResidencySet.commit()
            #endif
            rayTracingScene.finishBuild()
        } catch {
            rayTracingScene.failBuild(error)
            print("[RayTracing] \(error.localizedDescription)")
        }
    }

    func render(drawable: LayerRenderer.Drawable, frameIndex: UInt64) {
        let time = drawable.frameTiming.presentationTime.timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)
        let handAnchors = handTracking.handAnchors(at: time)
        leftHandPinchFrame = leftHandPinchTracker.update(with: handAnchors.leftHand)
        rightHandPinchFrame = rightHandPinchTracker.update(with: handAnchors.rightHand)
        reflectiveObjectPlacement.updateTrackedHeadPose(from: deviceAnchor)

        let movedObjectIDs = [
            reflectiveObjectGrabController.update(hand: .left,
                                                  pinch: leftHandPinchFrame,
                                                  objects: &reflectiveObjects),
            reflectiveObjectGrabController.update(hand: .right,
                                                  pinch: rightHandPinchFrame,
                                                  objects: &reflectiveObjects)
        ].compactMap { $0 }
        for objectID in Set(movedObjectIDs) {
            guard let object = reflectiveObjects.first(where: { $0.id == objectID }) else { continue }
            let objectTransform = reflectiveObjectTransform(for: object)
            let instanceID = RayTracingInstanceID.reflectiveObject(objectID)
            rayTracingScene.setInstanceTransform(objectTransform, for: instanceID)
            rayTracingHitDataBuffers.setObjectTransform(objectTransform,
                                                        sphereRadius: ReflectiveObject.radius,
                                                        at: meshes.count + objectID.rawValue)
            rayTracingScene.markDirty(.transformChanged(instanceID))
        }

        drawable.deviceAnchor = deviceAnchor

        if let deviceAnchor {
            for object in reflectiveObjects {
                let objectPointer = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()
                                                            + uniformBufferOffset
                                                            + alignedUniformsSize
                                                                * (meshes.count + object.id.rawValue))
                    .bindMemory(to: Uniforms.self, capacity: 1)
                objectPointer[0].cameraPosition = deviceAnchor.originFromAnchorTransform.columns.3
            }
        }

        if perDrawableTarget[drawable.target] == nil {
            perDrawableTarget[drawable.target] = .init(drawable: drawable)
        }
        let drawableTarget = perDrawableTarget[drawable.target]!

        drawableTarget.updateBufferState(uniformBufferIndex: uniformBufferIndex, frameIndex: frameIndex)

        drawableTarget.updateViewProjectionArray(drawable: drawable)

        let renderPassDescriptor = MTL4RenderPassDescriptor()

        if device.supportsMSAA {
            let renderTargets = drawableTarget.memorylessTargets[uniformBufferIndex]
            assert(renderTargets.color.width == drawable.colorTextures[0].width)
            assert(renderTargets.color.height == drawable.colorTextures[0].height)

            renderPassDescriptor.colorAttachments[0].resolveTexture = drawable.colorTextures[0]
            renderPassDescriptor.colorAttachments[0].texture = renderTargets.color
            renderPassDescriptor.depthAttachment.resolveTexture = drawable.depthTextures[0]
            renderPassDescriptor.depthAttachment.texture = renderTargets.depth

            renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve
            renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
        } else {
            renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
            renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]

            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.depthAttachment.storeAction = .store
        }

        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
        if layerRenderer.configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = drawable.views.count
        }

        #if !targetEnvironment(simulator)
        let residencySet = self.residencySets[uniformBufferIndex]
        residencySet.addAllocations([
            drawable.colorTextures[0],
            drawable.depthTextures[0],
            drawableTarget.viewProjectionBuffer
        ])
        residencySet.commit()
        #endif

        let commandAllocator = self.commandAllocators[uniformBufferIndex]
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)
        commandBuffer.useResidencySet(residencySet)

        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.pushDebugGroup("Draw Imported Model")

        renderEncoder.setCullMode(.back)

        renderEncoder.setFrontFacing(.counterClockwise)

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setDepthStencilState(depthState)

        let viewports = drawable.views.map { $0.textureMap.viewport }

        renderEncoder.setViewports(viewports)

        if drawable.views.count > 1 {
            let viewMappings = (0..<drawable.views.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewMappings)
        }

        renderEncoder.setArgumentTable(self.vertexArgumentTable, stages: .vertex)
        renderEncoder.setArgumentTable(self.fragmentArgumentTable, stages: .fragment)

        self.vertexArgumentTable.setAddress(drawableTarget.viewProjectionBuffer.gpuAddress + UInt64(drawableTarget.viewProjectionBufferOffset), index: BufferIndex.viewProjection.rawValue)

        self.fragmentArgumentTable.setTexture(colorMap.gpuResourceID, index: TextureIndex.color.rawValue)

        let isRoomVisible = roomVisibility.snapshot()
        for (meshIndex, renderMesh) in meshes.enumerated() where isRoomVisible {
            let mesh = renderMesh.mesh
            self.vertexArgumentTable.setAddress(dynamicUniformBuffer.gpuAddress
                                                + UInt64(uniformBufferOffset)
                                                + UInt64(alignedUniformsSize * meshIndex),
                                                index: BufferIndex.uniforms.rawValue)

            for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
                guard let layout = element as? MDLVertexBufferLayout else {
                    fatalError("unsupported layout")
                }

                if layout.stride != 0 {
                    let buffer = mesh.vertexBuffers[index]
                    self.vertexArgumentTable.setAddress(buffer.buffer.gpuAddress + UInt64(buffer.offset),
                                                        index: index)
                }
            }

            for (submeshIndex, submesh) in mesh.submeshes.enumerated() {
                let baseColorTexture = renderMesh.baseColorTextures.indices.contains(submeshIndex)
                    ? renderMesh.baseColorTextures[submeshIndex]
                    : nil
                self.fragmentArgumentTable.setTexture((baseColorTexture ?? colorMap).gpuResourceID,
                                                      index: TextureIndex.color.rawValue)

                renderEncoder.drawIndexedPrimitives(primitiveType: submesh.primitiveType,
                                                    indexCount: submesh.indexCount,
                                                    indexType: submesh.indexType,
                                                    indexBuffer: submesh.indexBuffer.buffer.gpuAddress + UInt64(submesh.indexBuffer.offset),
                                                    indexBufferLength: submesh.indexBuffer.buffer.length)
            }
        }

        renderEncoder.popDebugGroup()

        if !reflectiveObjects.isEmpty {
            renderEncoder.pushDebugGroup("Draw Pure Reflection Objects")
            areRayTracingResourcesBound = bindRayTracingResourcesIfReady()
            renderEncoder.setRenderPipelineState(areRayTracingResourcesBound
                                                  ? rayTracedReflectiveSpherePipelineState
                                                  : reflectiveSpherePipelineState)
            self.fragmentArgumentTable.setAddress(reflectiveMaterialBuffer.gpuAddress,
                                                  index: BufferIndex.material.rawValue)

            for object in reflectiveObjects {
                let objectUniformAddress = dynamicUniformBuffer.gpuAddress
                    + UInt64(uniformBufferOffset)
                    + UInt64(alignedUniformsSize * (meshes.count + object.id.rawValue))
                self.vertexArgumentTable.setAddress(objectUniformAddress,
                                                    index: BufferIndex.uniforms.rawValue)
                self.fragmentArgumentTable.setAddress(objectUniformAddress,
                                                      index: BufferIndex.uniforms.rawValue)

                guard let reflectiveObjectMesh = reflectiveObjectMeshes[object.kind]?.mesh else {
                    fatalError("Missing reflective object mesh")
                }
                for (index, element) in reflectiveObjectMesh.vertexDescriptor.layouts.enumerated() {
                    guard let layout = element as? MDLVertexBufferLayout else {
                        fatalError("unsupported reflective object layout")
                    }

                    if layout.stride != 0 {
                        let buffer = reflectiveObjectMesh.vertexBuffers[index]
                        self.vertexArgumentTable.setAddress(
                            buffer.buffer.gpuAddress + UInt64(buffer.offset),
                            index: index)
                    }
                }

                for submesh in reflectiveObjectMesh.submeshes {
                    renderEncoder.drawIndexedPrimitives(
                        primitiveType: submesh.primitiveType,
                        indexCount: submesh.indexCount,
                        indexType: submesh.indexType,
                        indexBuffer: submesh.indexBuffer.buffer.gpuAddress
                            + UInt64(submesh.indexBuffer.offset),
                        indexBufferLength: submesh.indexBuffer.buffer.length)
                }
            }

            renderEncoder.popDebugGroup()
        }

        renderEncoder.endEncoding()

        commandBuffer.endCommandBuffer()

        self.commandQueue.commit([commandBuffer])

        drawable.encodePresent()
    }

    private func bindRayTracingResourcesIfReady() -> Bool {
        guard rayTracingScene.isReady,
              let topLevelStructure = rayTracingScene.topLevelStructure,
              let hitData = rayTracingScene.hitDataBuffers else {
            return false
        }

        fragmentArgumentTable.setResource(topLevelStructure.gpuResourceID,
                                          bufferIndex: BufferIndex.rayTracingScene.rawValue)
        fragmentArgumentTable.setAddress(hitData.vertices.gpuAddress,
                                         index: BufferIndex.rayTracingVertices.rawValue)
        fragmentArgumentTable.setAddress(hitData.indices.gpuAddress,
                                         index: BufferIndex.rayTracingIndices.rawValue)
        fragmentArgumentTable.setAddress(hitData.geometries.gpuAddress,
                                         index: BufferIndex.rayTracingGeometries.rawValue)
        fragmentArgumentTable.setAddress(hitData.instances.gpuAddress,
                                         index: BufferIndex.rayTracingInstances.rawValue)
        fragmentArgumentTable.setAddress(hitData.objects.gpuAddress,
                                         index: BufferIndex.rayTracingObjects.rawValue)
        fragmentArgumentTable.setAddress(rayTracingSettingsBuffer.gpuAddress,
                                         index: BufferIndex.rayTracingSettings.rawValue)

        for (textureOffset, texture) in rayTracingScene.textures
            .prefix(maxRayTracingTextures).enumerated() {
            fragmentArgumentTable.setTexture(texture.gpuResourceID,
                                             index: TextureIndex.rayTracingBase.rawValue + textureOffset)
        }
        return true
    }

    func renderLoop() async {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                Task { @MainActor in
                    appModel.immersiveSpaceState = .closed
                    appModel.setAnchorAction(nil)
                }
                return
            } else if layerRenderer.state == .paused {
                Task { @MainActor in
                    appModel.immersiveSpaceState = .inTransition
                }
                layerRenderer.waitUntilRunning()
                continue
            } else {
                Task { @MainActor in
                    if appModel.immersiveSpaceState != .open {
                        appModel.immersiveSpaceState = .open
                    }
                }
                autoreleasepool {
                    self.renderFrame()
                }
                await Task.yield()
            }
        }
    }
}

extension Renderer {
    class DrawableTarget {
        var lastUsedFrameIndex: UInt64

        let memorylessTargets: [(color: MTLTexture, depth: MTLTexture)]

        let viewProjectionBuffer: MTLBuffer

        var viewProjectionBufferOffset = 0

        var viewProjectionArray: UnsafeMutablePointer<ViewProjectionArray>

        nonisolated init(drawable: LayerRenderer.Drawable) {
            lastUsedFrameIndex = 0

            let device = drawable.colorTextures[0].device
            nonisolated func renderTarget(resolveTexture: MTLTexture) -> MTLTexture {
                assert(device.supportsMSAA)

                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: resolveTexture.pixelFormat,
                                                                          width: resolveTexture.width,
                                                                          height: resolveTexture.height,
                                                                          mipmapped: false)
                descriptor.usage = .renderTarget
                descriptor.textureType = .type2DMultisampleArray
                descriptor.sampleCount = device.rasterSampleCount
                descriptor.storageMode = .memoryless
                descriptor.arrayLength = resolveTexture.arrayLength
                return device.makeTexture(descriptor: descriptor)!
            }

            if device.supportsMSAA {
                memorylessTargets = .init(repeating: (renderTarget(resolveTexture: drawable.colorTextures[0]),
                                                      renderTarget(resolveTexture: drawable.depthTextures[0])),
                                          count: maxBuffersInFlight)
            } else {
                memorylessTargets = []
            }

            let bufferSize = alignedViewProjectionArraySize * maxBuffersInFlight

            viewProjectionBuffer = device.makeBuffer(length: bufferSize,
                                                     options: [MTLResourceOptions.storageModeShared])!
            viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)
        }
    }
}

extension Renderer.DrawableTarget {
    nonisolated func updateBufferState(uniformBufferIndex: Int, frameIndex: UInt64) {
        viewProjectionBufferOffset = alignedViewProjectionArraySize * uniformBufferIndex

        viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)

        lastUsedFrameIndex = frameIndex
    }

    nonisolated func updateViewProjectionArray(drawable: LayerRenderer.Drawable) {
        let simdDeviceAnchor = drawable.deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        nonisolated func viewProjection(forViewIndex viewIndex: Int) -> float4x4 {
            let view = drawable.views[viewIndex]
            let viewMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = drawable.computeProjection(viewIndex: viewIndex)

            return projectionMatrix * viewMatrix
        }

        viewProjectionArray[0].viewProjectionMatrix.0 = viewProjection(forViewIndex: 0)
        if drawable.views.count > 1 {
            viewProjectionArray[0].viewProjectionMatrix.1 = viewProjection(forViewIndex: 1)
        }
    }
}

// Generic matrix math utility functions
nonisolated func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return .init(columns: (vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                           vector_float4(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
                           vector_float4(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
                           vector_float4(                  0, 0, 0, 1)))
}

nonisolated func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return .init(columns: (vector_float4(1, 0, 0, 0),
                           vector_float4(0, 1, 0, 0),
                           vector_float4(0, 0, 1, 0),
                           vector_float4(translationX, translationY, translationZ, 1)))
}

nonisolated func matrix4x4_scale(_ scale: Float) -> matrix_float4x4 {
    .init(columns: (vector_float4(scale, 0, 0, 0),
                    vector_float4(0, scale, 0, 0),
                    vector_float4(0, 0, scale, 0),
                    vector_float4(0, 0, 0, 1)))
}
