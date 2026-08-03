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

static float3 applyObjectMaterial(float3 reflectionColor,
                                  constant PureReflectionMaterialUniforms &material)
{
    float metallic = saturate(material.metallic);
    float roughness = saturate(material.roughness);
    float3 baseColor = saturate(material.baseColor.rgb);
    float3 specularTint = mix(float3(1.0), baseColor, metallic);
    float specularStrength = mix(0.55, 1.0, metallic) * (1.0 - roughness * 0.2);
    float3 specular = reflectionColor * specularTint * specularStrength;
    float3 diffuse = baseColor * material.diffuseContribution * 0.65;
    return specular + diffuse;
}

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
    chrome = mix(chrome, float3(dot(chrome, float3(0.2126, 0.7152, 0.0722))),
                 saturate(material.roughness) * 0.65);
    return float4(applyObjectMaterial(chrome, material), 1.0);
}

static float3 rayMissColor(constant PureReflectionMaterialUniforms &material)
{
    return saturate(material.missColor.rgb);
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
    device const RayTracingObject *objects [[buffer(BufferIndexRayTracingObjects)]],
    constant RayTracingSettings &settings [[buffer(BufferIndexRayTracingSettings)]],
    array<texture2d<half>, 64> roomTextures [[texture(TextureIndexRayTracingBase)]])
{
    float3 normal = normalize(in.worldNormal);
    float3 incidentDirection = normalize(in.worldPosition - uniforms.cameraPosition.xyz);

    ray reflectionRay;
    reflectionRay.origin = in.worldPosition + normal * settings.rayEpsilon;
    reflectionRay.direction = normalize(reflect(incidentDirection, normal));
    reflectionRay.min_distance = settings.rayEpsilon * 0.5;
    reflectionRay.max_distance = settings.maxDistance;

    intersector<triangle_data, instancing> sceneIntersector;
    sceneIntersector.assume_geometry_type(geometry_type::triangle);
    constexpr sampler roomSampler(mip_filter::linear,
                                  mag_filter::linear,
                                  min_filter::linear,
                                  address::repeat);
    float3 throughput = float3(material.reflectivity);
    uint ignoredInstanceIndex = uniforms.rayTracingData.x;

    uint bounceLimit = min(settings.maxBounces, 10u);
    for (uint bounce = 0; bounce < bounceLimit; ++bounce) {
        auto intersection = sceneIntersector.intersect(reflectionRay, scene, 0xFF);
        for (uint skippedIntersection = 0;
             skippedIntersection < 16u
                 && intersection.type == intersection_type::triangle
                 && intersection.instance_id == ignoredInstanceIndex;
             ++skippedIntersection) {
            float advance = intersection.distance + settings.rayEpsilon;
            reflectionRay.origin += reflectionRay.direction * advance;
            intersection = sceneIntersector.intersect(reflectionRay, scene, 0xFF);
        }
        if (intersection.type != intersection_type::triangle) {
            return float4(applyObjectMaterial(rayMissColor(material) * throughput, material),
                          1.0);
        }

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

            uint textureIndex = min(geometry.textureIndex, 63u);
            uint mipCount = roomTextures[textureIndex].get_num_mip_levels();
            float maxMipLevel = float(max(mipCount, 1u) - 1u);
            float mipLevel = saturate(material.roughness) * maxMipLevel;
            half4 roomColor = roomTextures[textureIndex]
                .sample(roomSampler, uv, level(mipLevel));
            return float4(applyObjectMaterial(float3(roomColor.rgb) * throughput, material),
                          1.0);
        }

        uint geometryIndex = hitInstance.geometryOffset + intersection.geometry_id;
        RayTracingGeometry geometry = geometries[geometryIndex];
        uint triangleIndex = geometry.indexOffset + intersection.primitive_id * 3;
        uint vertexIndex0 = indices[triangleIndex];
        uint vertexIndex1 = indices[triangleIndex + 1];
        uint vertexIndex2 = indices[triangleIndex + 2];
        float2 barycentric = intersection.triangle_barycentric_coord;
        float weight0 = 1.0 - barycentric.x - barycentric.y;
        float3 localNormal = vertices[vertexIndex0].normal.xyz * weight0
            + vertices[vertexIndex1].normal.xyz * barycentric.x
            + vertices[vertexIndex2].normal.xyz * barycentric.y;

        RayTracingObject hitObject = objects[instanceIndex];
        float3 hitPosition = reflectionRay.origin
            + reflectionRay.direction * intersection.distance;
        float3x3 worldToObject = float3x3(hitObject.worldToObject[0].xyz,
                                          hitObject.worldToObject[1].xyz,
                                          hitObject.worldToObject[2].xyz);
        float3 hitNormal = normalize(transpose(worldToObject) * localNormal);
        if (dot(hitNormal, reflectionRay.direction) > 0.0) {
            hitNormal = -hitNormal;
        }
        ignoredInstanceIndex = instanceIndex;
        reflectionRay.origin = hitPosition + hitNormal * settings.rayEpsilon;
        reflectionRay.direction = normalize(reflect(reflectionRay.direction, hitNormal));
        throughput *= material.reflectivity;
    }

    return float4(applyObjectMaterial(throughput * rayMissColor(material), material), 1.0);
}
