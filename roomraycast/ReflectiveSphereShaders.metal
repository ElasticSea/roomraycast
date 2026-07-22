//
//  ReflectiveSphereShaders.metal
//  roomraycast
//

#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

struct ReflectiveSphereVertex {
    float3 position [[attribute(VertexAttributePosition)]];
    float3 normal [[attribute(VertexAttributeNormal)]];
};

struct ReflectiveSphereVaryings {
    float4 position [[position]];
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
    out.worldNormal = normalize((float3x3)uniforms.modelMatrix * in.normal);
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
