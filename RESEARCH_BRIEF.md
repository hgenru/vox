# Deep Research Brief: Unified Voxel Physics for Massive Worlds

## What we're building

A real-time 3D voxel physics engine ("Noita 3D") — every voxel is a physical particle.
Materials: stone, water, lava, ice, wood, gunpowder, steam. Full phase transitions and chemistry.
GPU-only simulation (Vulkan compute shaders, Rust + rust-gpu).
Target: Minecraft-scale world (thousands of chunks), RTX 4090 primary, but must scale to 6-8GB GPUs.

## Current implementation

**Physics:** MLS-MPM (Moving Least Squares Material Point Method) on GPU.
- Every voxel = one MPM particle (position, velocity, deformation gradient F, temperature, material_id, phase)
- Per-frame pipeline: Clear Grid → P2G (scatter) → Grid Update → G2P (gather) → Voxelize → Render
- Solid: fixed corotated elasticity (SVD polar decomposition, McAdams 2011)
- Liquid: equation of state + viscosity
- Gas: weak EOS
- Phase transitions: temperature-driven (stone↔lava at 1500K, water↔steam at 100K, water↔ice at 0K)
- Chemical reactions: neighbor-contact based (water+lava→stone+steam, wood burns)
- Grid: 256³ dense (768 MB VRAM just for grid!)
- Particles: up to 4M, 144 bytes each

**Optimization already tried:**
- Brick-based sleeping (8³ bricks, activity tracking, multi-rate scheduling)
- Sparse hash grid — FAILED, particles stop moving, reverted to dense
- Counting sort for cache coherence
- Per-brick particle dispatch
- Dirty-tile rendering
- Far-field voxel rendering for distant chunks

**What works well:**
- MPM itself is solid — fluids flow, solids deform, phase transitions work
- Brick sleeping gives some savings
- Rendering pipeline is decent

**What's broken / problematic:**
- Dense grid: 768MB for 256³ = 12.8m world. Way too small, way too much VRAM.
- Every solid particle runs full P2G/G2P even when sitting still as part of a mountain
- No concept of "rigid pieces" — a boulder is 1000s of independent particles, all expensive
- Sparse grid attempt failed (correctness issues with hash collisions or boundary conditions)
- Chunk streaming exists but async loading is disabled
- World is tiny (12.8m) — need 100x-1000x bigger

## The core problem

MPM computes P2G/G2P for EVERY particle EVERY frame. In a world that's 99% static rock, this is catastrophically wasteful. We need a way to:

1. **Not simulate static stuff** — a mountain should cost ~0 FLOPS until something disturbs it
2. **Handle rigid motion cheaply** — a falling boulder should NOT need per-particle MPM. It's rigid until it impacts.
3. **Fracture naturally** — when a boulder hits ground, it should crack into pieces, and THOSE pieces should behave correctly
4. **Keep fluids working** — water, lava, gas still need full MPM (or something equivalent)
5. **Scale to huge worlds** — only simulate physics where "interesting" things happen

## What we've considered so far

### Approach 1: Multi-domain MPM
Multiple small (64³-128³) MPM grids, each independent. Active grids around player + activity zones.
- Pro: each domain is cheap
- Con: particle transfer between domains is hard, boundary artifacts

### Approach 2: Sparse active-set
Fix the hash grid so cells only exist where particles are.
- Pro: single domain, no boundaries
- Con: hash collisions, random memory access kills cache, we already tried and it broke

### Approach 3: "Active zones" hybrid
Static terrain stored as voxel chunks (no MPM). MPM only for active zones (fluid, damage, heat).
Zones activate when disturbed.
- Pro: huge world cheap, physics only where needed
- Con: transition between "static voxel" and "MPM particle" needs careful design

### Approach 4: Layered activity system (current best idea)
All voxels are particles, but with different simulation tiers:
- **Sleeping**: 0 cost, just voxel data
- **Rigid islands** (shape matching): one SVD per chunk, not per particle. Detected via GPU CCL.
- **Thermal/chemical only**: just heat diffusion + reaction checks, no P2G/G2P
- **Active MPM**: full simulation, only for fluids and deforming materials

Lifecycle: sleeping → (disturbed) → rigid island falling → (impact) → active MPM → (settled) → sleeping

## What we need researched

### 1. Unified particle systems with variable-cost simulation
Are there papers/engines that run MPM (or similar) with per-particle activity levels?
Specifically: sleeping particles that skip P2G/G2P, with correct wake-up propagation.
Houdini's MPM solver has "sleep" — how exactly does it work? What are the boundary conditions at the sleep/active interface?

### 2. Shape matching as rigid body substitute within particle systems
Müller et al. 2005 "Meshless Deformations Based on Shape Matching" and its follow-ups.
Can shape matching coexist with MPM in one pipeline? Specifically:
- Shape-matched cluster interacts with MPM fluid (rigid rock falls into water)
- Cluster-to-cluster collision
- Transition: cluster exceeds strain → breaks into MPM particles
- How to do this on GPU efficiently (one SVD per cluster, not per particle)

### 3. GPU Connected Component Labeling for voxel grids
After destruction, how to find disconnected pieces in real-time on GPU?
- Block-based union-find on 3D grid
- Amortized over multiple frames (Teardown approach)
- Can this run on the same grid we already have?

### 4. Bond/lattice-based fracture for voxels
How to model bonds between adjacent solid voxels that break under stress?
- Peridynamics-inspired bond breaking in MPM context
- Strain-based fracture criteria for voxel bonds
- How does this interact with MPM's deformation gradient F?
- Papers on MPM + fracture/damage mechanics

### 5. Position-Based MPM (PB-MPM, EA SEED)
Their reformulation allows huge timesteps. Could this help us?
- Can PB-MPM be combined with sleeping/rigid clustering?
- Does it change the P2G/G2P cost structure?
- Is it more numerically stable for our phase transition edge cases?

### 6. Alternatives to MPM for our use case
Maybe MPM isn't the best fit for a game engine where 99% of the world is static solid?
- FLIP/APIC for fluids only, something simpler for solids?
- Lattice Boltzmann for fluids + discrete element for solids?
- Any hybrid approach specifically designed for "mostly static, locally dynamic" voxel worlds?
- What does Noita actually use? (they claim "falling sand + rigid body + custom fluid sim")

### 7. Massive world simulation architectures
How do engines handle physics in worlds much larger than the simulation domain?
- Minecraft: only simulates loaded chunks, redstone propagation is tick-based
- Teardown: voxel destruction + rigid body islands + flood-fill CCL
- Noita: pixel physics on visible screen + simplified simulation off-screen
- Any papers on "simulation Level of Detail" for physics?

### 8. GPU memory efficiency for sparse voxel physics
Our dense grid is 768MB for 256³. How to go bigger?
- VDB / NanoVDB for sparse voxel storage on GPU
- Hash grids that actually work (what went wrong with ours?)
- Brick-map approaches (like NVIDIA GVDB)
- Two-level grid: coarse dense + fine sparse

### 9. Temperature and chemistry at scale
Heat diffusion currently requires grid neighbors (P2G/G2P).
For sleeping particles, how to do thermal simulation without the full MPM grid?
- Separate thermal grid (cheaper, no velocity)?
- Per-particle heat exchange with direct neighbors?
- Can we run chemistry on a coarser grid?

## Constraints

- Must run real-time (60fps physics, max fps render) on RTX 4090
- Must be implementable in Vulkan compute shaders (Rust + rust-gpu → SPIR-V)
- Must handle water, lava, stone, gas in one unified system (no separate engines per material type)
- Fracture and destruction are core gameplay, not optional
- World should be explorable (Minecraft-scale), not just a small arena

## Output format

For each topic, we'd like:
1. **Best papers/sources** (with key insight from each)
2. **What's practical for real-time GPU** (vs what's only for offline simulation)
3. **Concrete architecture recommendations** — not just "use X", but HOW it fits with our existing MPM pipeline
4. **Open questions and risks** — what could go wrong, what's unproven
5. **Priority ranking** — what gives the most bang for the buck for our specific case
