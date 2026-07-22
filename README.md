# Room Raycast

Room Raycast is a visionOS app that imports a 3D room model into an immersive Metal scene. The room is rendered normally using its meshes, materials, and textures.

The app spawns shiny objects throughout the room. Using Metal ray tracing, every reflective pixel can see the room and the other spawned objects, creating real-time object-to-object and object-to-room reflections.

The renderer uses a hybrid approach: normal rasterization for the main image and ray tracing for reflections. One reflection bounce shows the room and other objects; additional bounces can create mirror-within-mirror reflections at a higher performance cost.

## Vision Pro Performance

This is possible on both M2 and M5 Vision Pro. M5 has hardware-accelerated ray tracing, so a reflection-heavy scene could run roughly **3–5× faster** than on M2; the complete renderer will more likely be around **2–3× faster**, depending on model complexity, texture resolution, object count, and reflection quality. These are estimates—the final difference must be measured on both devices.
