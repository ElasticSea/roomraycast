//
//  ShaderTypes.h
//  roomraycast
//
//  Created by Elastic Sea on 7/22/26.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions  = 0,
    BufferIndexMeshGenerics   = 1,
    BufferIndexUniforms       = 2,
    BufferIndexViewProjection = 3,
    BufferIndexMaterial       = 4,
    BufferIndexRayTracingScene      = 5,
    BufferIndexRayTracingVertices   = 6,
    BufferIndexRayTracingIndices    = 7,
    BufferIndexRayTracingGeometries = 8,
    BufferIndexRayTracingInstances  = 9,
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition   = 0,
    VertexAttributeTexcoord   = 1,
    VertexAttributeNormal     = 2,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor         = 0,
    TextureIndexRayTracingBase = 1,
};

typedef NS_ENUM(EnumBackingType, RayTracingMaterialKind)
{
    RayTracingMaterialKindRoom           = 0,
    RayTracingMaterialKindPureReflection = 1,
};

typedef struct
{
    matrix_float4x4 viewProjectionMatrix[2];
} ViewProjectionArray;

typedef struct
{
    matrix_float4x4 modelMatrix;
    vector_float4 cameraPosition;
} Uniforms;

typedef struct
{
    float reflectivity;
    float roughness;
    float metallic;
    float diffuseContribution;
} PureReflectionMaterialUniforms;

typedef struct
{
    vector_float4 position;
    vector_float4 normal;
    vector_float4 texCoordAndPadding;
} RayTracingVertex;

typedef struct
{
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t textureIndex;
    uint32_t triangleCount;
} RayTracingGeometry;

typedef struct
{
    uint32_t geometryOffset;
    uint32_t geometryCount;
    uint32_t materialKind;
    uint32_t padding;
} RayTracingInstance;

#endif /* ShaderTypes_h */
