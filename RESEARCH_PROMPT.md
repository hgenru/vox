# Research Project: Scalable Voxel Physics Engine

We're building a real-time 3D voxel physics engine — think "Noita but 3D" or "Minecraft with real physics". Every voxel is a physical particle: stone, water, lava, wood, ice, gas. Materials melt, freeze, burn, explode, flow. Destruction is core gameplay.

Current engine uses GPU MPM (Material Point Method) in Vulkan compute shaders (Rust). It works for small scenes but doesn't scale: a 256³ grid eats 768MB VRAM and simulates every particle every frame, even static mountains.

**The core challenge:** how to make a Minecraft-scale world where 99% is static rock, but any part can dynamically melt, fracture, flow, or explode — without simulating everything all the time.

We need a unified architecture for: particle sleeping, rigid body behavior for solid chunks (falling boulders), fracture/destruction, fluid simulation, and heat/chemistry — all in one GPU pipeline, scaling to massive worlds.

See RESEARCH_BRIEF.md for full technical context, current implementation details, and 9 specific research directions.
