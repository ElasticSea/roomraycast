//
//  ReflectiveSphereShaders.metal
//  roomraycast
//

#include <metal_stdlib>
#include <metal_raytracing>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;
using namespace raytracing;

struct ReflectiveSphereVertex {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal [[attribute(VertexAttributeNormal)]];
};

struct ReflectiveSphereVaryings {
    float4 position [[position]];
    float3 worldPosition;
    float3 worldNormal;
};

vertex ReflectiveSphereVaryings reflectiveSphereVertex(
    ReflectiveSphereVertex in [[stage_in]],
    ushort ampID [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant ViewProjectionArray &viewProjectionArray [[buffer(BufferIndexViewProjection)]])
{
    ReflectiveSphereVaryings out;
    float4 worldPosition = uniforms.modelMatrix * float4(in.position, 1.0);
    out.position = viewProjectionArray.viewProjectionMatrix[ampID] * worldPosition;
    out.worldPosition = worldPosition.xyz;
    float3x3 normalMatrix = float3x3(uniforms.modelMatrix[0].xyz,
                                     uniforms.modelMatrix[1].xyz,
                                     uniforms.modelMatrix[2].xyz);
    out.worldNormal = normalize(normalMatrix * in.normal);
    return out;
}

fragment float4 reflectiveSphereFragment(
    ReflectiveSphereVaryings in [[stage_in]],
    constant PureReflectionMaterialUniforms &material [[buffer(BufferIndexMaterial)]])
{
    float3 normal = normalize(in.worldNormal);
    float skyAmount = saturate(normal.y * 0.5 + 0.5);
    float horizonBand = pow(saturate(1.0 - abs(normal.y)), 8.0);
    float3 darkChrome = float3(0.025, 0.035, 0.055);
    float3 brightChrome = float3(0.75, 0.86, 1.0);
    float3 chrome = mix(darkChrome, brightChrome, skyAmount);
    chrome += horizonBand * 0.35;
    chrome *= material.reflectivity;
    return float4(chrome, 1.0);
}

fragment float4 rayTracedReflectiveSphereFragment(
    ReflectiveSphereVaryings in [[stage_in]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant PureReflectionMaterialUniforms &material [[buffer(BufferIndexMaterial)]],
    instance_acceleration_structure scene [[buffer(BufferIndexRayTracingScene)]],
    device const RayTracingVertex *vertices [[buffer(BufferIndexRayTracingVertices)]],
    device const uint32_t *indices [[buffer(BufferIndexRayTracingIndices)]],
    device const RayTracingGeometry *geometries [[buffer(BufferIndexRayTracingGeometries)]],
    device const RayTracingInstance *instances [[buffer(BufferIndexRayTracingInstances)]],
    array<texture2d<half>, 64> roomTextures [[texture(TextureIndexRayTracingBase)]])
{
    float3 normal = normalize(in.worldNormal);
    float3 incidentDirection = normalize(in.worldPosition - uniforms.cameraPosition.xyz);

    ray reflectionRay;
    reflectionRay.origin = in.worldPosition + normal * 0.002;
    reflectionRay.direction = normalize(reflect(incidentDirection, normal));
    reflectionRay.min_distance = 0.001;
    reflectionRay.max_distance = 100.0;

    intersector<triangle_data, instancing> sceneIntersector;
    sceneIntersector.assume_geometry_type(geometry_type::triangle);
    auto intersection = sceneIntersector.intersect(reflectionRay, scene, 0xFF);

    if (intersection.type == intersection_type::triangle) {
        uint instanceIndex = intersection.instance_id;
        RayTracingInstance hitInstance = instances[instanceIndex];
        if (hitInstance.materialKind == RayTracingMaterialKindRoom) {
            uint geometryIndex = hitInstance.geometryOffset + intersection.geometry_id;
            RayTracingGeometry geometry = geometries[geometryIndex];
            uint triangleIndex = geometry.indexOffset + intersection.primitive_id * 3;
            uint vertexIndex0 = indices[triangleIndex];
            uint vertexIndex1 = indices[triangleIndex + 1];
            uint vertexIndex2 = indices[triangleIndex + 2];

            float2 barycentric = intersection.triangle_barycentric_coord;
            float weight0 = 1.0 - barycentric.x - barycentric.y;
            float2 uv0 = vertices[vertexIndex0].texCoordAndPadding.xy;
            float2 uv1 = vertices[vertexIndex1].texCoordAndPadding.xy;
            float2 uv2 = vertices[vertexIndex2].texCoordAndPadding.xy;
            float2 uv = uv0 * weight0 + uv1 * barycentric.x + uv2 * barycentric.y;

            constexpr sampler roomSampler(mip_filter::linear,
                                          mag_filter::linear,
                                          min_filter::linear,
                                          address::repeat);
            half4 roomColor = roomTextures[min(geometry.textureIndex, 63u)]
                .sample(roomSampler, uv);
            return float4(roomColor) * material.reflectivity;
        }

        return float4(0.75, 0.86, 1.0, 1.0) * material.reflectivity;
    }

    return float4(0.025, 0.035, 0.055, 1.0);
}
