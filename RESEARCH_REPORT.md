# Implementation Plan: Massive Voxel World POC

## Context

We have a working GPU MLS-MPM prototype (Vulkan compute, Rust + rust-gpu → SPIR-V). It works for small scenes but doesn't scale — 256³ dense grid eats 768MB VRAM and simulates every particle every frame. We need to prove that a Minecraft-scale world (thousands of chunks) can have real-time physics, chemistry, and destruction.

**This POC proves the core architecture**: cellular automata substrate + on-demand PB-MPM particle zones, with chunk streaming. We're NOT building a full game — we're proving the system works at scale.

## Target Hardware

- Primary: RTX 5070 / RTX 4090 class (12-24 GB VRAM)
- Must run on: RTX 4060+ / RX 7700 XT+ (8+ GB VRAM)
- API: Vulkan 1.3 compute shaders via rust-gpu → SPIR-V

## Existing Codebase

- Rust + rust-gpu for GPU compute shaders
- Vulkan compute pipeline (already working)
- MLS-MPM solver (P2G, grid update, G2P — all on GPU)
- Basic voxel renderer
- Server/client separation (in progress)
- Brick-based sleeping (partially working)
- Counting sort for particle cache coherence

---

## Architecture Overview

Two-layer system:

```
Layer 1: CA Substrate (entire loaded world)
├── 4 bytes per voxel (material 10b, temperature 8b, bonds 6b, oxygen 4b, flags 4b)
├── Chunks: 32³ voxels = 128 KB each, fixed-size pool on GPU
├── Processes: thermal diffusion, chemical reactions, falling sand, phase transitions
├── Cost: ~0 for static regions, only dirty chunks processed
└── Triggers PB-MPM activation when CA can't handle the physics

Layer 2: PB-MPM Particle Zones (local, on-demand)
├── Spawned from voxels when triggered (explosion, collapse, complex fluid)
├── 2-4 simultaneous zones, 32³-128³ each
├── Full physics: fluids, deformation, fracture, rigid bodies via shape matching
├── Particles sleep back into voxels when settled
└── If overloaded: world time slows down (graceful degradation, no hard caps)
```

---

## Phase 1: Voxel World Foundation

**Goal**: Get a massive world of chunks on GPU with streaming, before any physics.

### 1.1 Voxel Data Format

```rust
// 4 bytes per voxel, packed into u32
struct Voxel {
    material_id: u10,  // bits 0-9: 1024 material types
    temperature: u8,   // bits 10-17: 0-255 game units (NOT Kelvin)
    bonds: u6,         // bits 18-23: connections to ±X, ±Y, ±Z neighbors
    oxygen: u4,        // bits 24-27: 0-15 oxygen level
    flags: u4,         // bits 28-31: dirty, burning, wet, poisoned
}
```

Packing/unpacking via bitwise ops in both Rust (CPU) and rust-gpu (GPU) code. All values are game-balance units, not physical units.

### 1.2 Chunk Structure

```rust
// CPU side
struct Chunk {
    voxels: [u32; 32 * 32 * 32],  // 128 KB — main data
    dirty_bitmask: [u32; 1024],    // 4 KB — 1 bit per voxel (32768 bits)
    metadata: ChunkMetadata,
}

struct ChunkMetadata {
    world_pos: IVec3,           // chunk position in world coordinates
    activity: ActivityLevel,     // Sleeping | CA_Only | Physics
    dirty: bool,                 // any voxel changed since last tick
    neighbor_ids: [Option<u32>; 6],  // indices of neighbor chunks in pool
    gpu_buffer_offset: u32,      // offset in gigabuffer
    is_homogeneous: bool,        // true = only metadata stored, no voxel array
    homogeneous_material: u32,   // if homogeneous: the single voxel value
}

enum ActivityLevel {
    Sleeping,   // 0 cost, not dispatched
    CAOnly,     // CA automaton runs on dirty voxels
    Physics,    // contains PB-MPM zone
}
```

### 1.3 GPU Memory Layout — Gigabuffer Pattern

Allocate one large buffer at startup. Use VMA (Vulkan Memory Allocator) for sub-allocation.

```
Gigabuffer (~512 MB):
├── Chunk Pool:       2048 slots × 132 KB = 264 MB
│   (each slot: 128 KB voxels + 4 KB dirty bitmask)
│   NOTE: 34³ with ghost cells if needed later = 157 KB, plan for this
│
├── Chunk Metadata:   2048 × 64 B = 128 KB
├── Dirty Chunk List: u32[2048] + dispatch args = 8 KB + 12 B
├── Material Table:   1024 × 32 B = 32 KB (material properties)
├── Reaction Table:   sparse, ~4096 entries × 16 B = 64 KB
└── Free Slot Stack:  u32[2048] = 8 KB
```

Use **32-bit offsets** into the gigabuffer, not 64-bit pointers — saves bandwidth in shaders.

Key pattern: `VkBufferDeviceAddress` or bind the whole buffer as one SSBO, index by `chunk_pool[slot_id * CHUNK_SIZE_U32 + voxel_index]`.

### 1.4 Chunk Streaming

```
CPU Thread: Streaming Manager
├── Priority queue of chunks to load/unload
│   Priority = 1/(distance² + ε) × dot(move_dir, chunk_dir)
│   Chunks ahead of player movement get higher priority
│
├── Load path:
│   1. Decompress/generate chunk data on CPU
│   2. Pop free slot from free_slot_stack
│   3. Write to staging buffer (double-buffered, 32 MB each)
│   4. Signal GPU upload via timeline semaphore
│
├── GPU Upload (per frame):
│   vkCmdCopyBuffer(staging → gigabuffer[slot])
│   Update metadata
│   Mark chunk dirty
│
├── Unload path:
│   1. vkCmdCopyBuffer(gigabuffer[slot] → staging) if chunk was modified
│   2. CPU compresses and writes to disk
│   3. Push slot to free_slot_stack
│
└── Predictive loading:
    Load chunks at player_pos + velocity * 2.0 seconds ahead
    Sphere of radius = render_distance around player
    Budget: max 20 chunk uploads per frame (2.5 MB at 128 KB each)
```

Use `VK_KHR_timeline_semaphore` (core in Vulkan 1.3) for upload/compute/render synchronization without CPU stalls.

### 1.5 Homogeneous Chunk Optimization

Most chunks deep underground are solid stone. Don't allocate 128 KB for them.

```rust
// When loading a chunk, check if all voxels are identical
if chunk.is_homogeneous {
    // Store only metadata, no GPU buffer slot needed
    // On first modification: allocate slot, fill with homogeneous value, then apply change
}
```

This can save 80%+ of VRAM for a typical world.

### 1.6 Milestone 1 Deliverable

A world of 100×100×10 chunks (32M voxels) that the player can fly through at high speed. Chunks stream in/out smoothly. Voxels render (even as simple colored cubes). No physics yet — just prove the data layer works at scale.

---

## Phase 2: Cellular Automata

**Goal**: Thermal diffusion, chemical reactions, falling sand on the CA substrate.

### 2.1 Dirty Chunk Tracking + Stream Compaction

Before dispatching CA shaders, compact the list of dirty chunks:

```
Pass 0: Compact Dirty Chunks
  Input:  metadata[2048].dirty (bool per chunk)
  Output: dirty_list[N] (contiguous array of dirty chunk IDs)
          dirty_count (u32)
          VkDispatchIndirectCommand { x: ceil(dirty_count/1), y: 1, z: 1 }

  Implementation: parallel prefix sum + scatter
  Reference: Raph Levien's decoupled look-back (from Vello, Apache 2.0)
  Fallback: simple 3-pass tree reduction if subgroup memory model not available
```

All subsequent CA dispatches use `vkCmdDispatchIndirect` with this compacted list. Sleeping world = 0 dispatches.

### 2.2 Margolus Partitioning for Falling Sand

Race-condition-free parallel CA on GPU. The world is divided into 2×2×2 blocks. Each thread processes one block atomically.

```
Two passes per tick:
  Pass A: blocks at offset (0,0,0) — even partition
  Pass B: blocks at offset (1,1,1) — odd partition

Within each block (8 voxels), one thread decides:
  - Which voxels swap (sand falls down, gas rises up)
  - Which reactions happen (adjacent materials interact)
  - All writes are local to the block — no conflicts

Important: chunk size 32 is even, so 2×2×2 blocks never cross chunk boundaries.
```

Workgroup layout: `local_size(4, 4, 4)` = 64 threads per workgroup. Each thread handles one 2×2×2 block. One workgroup covers a 8×8×8 region = part of one chunk.

Dispatch: `dispatchIndirect(ca_pass_a, dirty_count)` where each workgroup processes one dirty chunk (multiple workgroups per chunk for full coverage: 16×16×16 blocks per 32³ chunk = 4096 blocks, so 64 workgroups per chunk).

### 2.3 Thermal Diffusion Pass

Separate pass before Margolus, using a stencil pattern:

```glsl
// For each dirty voxel:
uint my_temp = get_temperature(voxel);
uint avg_temp = average_neighbor_temperatures(chunk, x, y, z); // 6 neighbors
int conductivity = material_table[get_material(voxel)].conductivity; // 0-10

// Integer arithmetic for determinism:
// new_temp = my_temp + (conductivity * (avg_temp - my_temp)) / 256
// Division by 256 = right shift, always rounds toward zero
int delta = (conductivity * (avg_temp as i32 - my_temp as i32)) >> 8;
set_temperature(voxel, clamp(my_temp as i32 + delta, 0, 255) as u8);
```

All integer math. Deterministic across GPUs. Temperature unit is arbitrary game units (not Kelvin).

**Chunk boundary reads**: for voxels at chunk edges, read from neighbor chunk's buffer. Neighbor chunk IDs stored in metadata. If neighbor not loaded → assume default (ambient temperature). Alternative: ghost cells (34³ chunks, +23% memory, +0 random reads).

Decision: **start without ghost cells** (simpler), add them later if profiling shows boundary reads are a bottleneck.

### 2.4 Chemical Reaction Pass

```rust
// Reaction table entry (stored in GPU buffer)
struct Reaction {
    input_a: u16,         // material ID
    input_b: u16,         // material ID
    condition: u8,        // 0=none, 1=needs_oxygen, 2=needs_high_temp, ...
    output_a: u16,        // new material for A (0xFFFF = no change)
    output_b: u16,        // new material for B
    heat_delta: i8,       // temperature change (-128 to +127)
    impulse: u8,          // 0=none, 1=weak, 2=medium, 3=strong (triggers PB-MPM)
    probability: u8,      // 0-255, reaction happens with this probability per tick
}
```

Reaction lookup: each voxel checks its 6 neighbors. If `(my_material, neighbor_material)` or `(neighbor_material, my_material)` exists in the reaction table → apply. Use a GPU hash map or sorted array with binary search for the table.

**Reactions are evaluated during the Margolus pass**, not separately — this ensures that a reaction between two adjacent voxels is handled by the thread owning their shared block, avoiding race conditions.

### 2.5 Phase Transitions

Part of the reaction/temperature system:

```
Material table entry:
  melt_temp: u8        // above this → becomes melt_into material
  freeze_temp: u8      // below this → becomes freeze_into material  
  boil_temp: u8        // above this → becomes boil_into material
  melt_into: u16       // material ID when melting
  freeze_into: u16
  boil_into: u16
```

Check in thermal pass: if `temperature > melt_temp` → change material to `melt_into`, update bonds (liquids have no bonds). Simple, deterministic, per-voxel.

### 2.6 Oxygen Diffusion

Same pattern as thermal diffusion but on the 4-bit oxygen field:

```
- Open sky voxels: oxygen = 15 (source)
- Fire consumes: oxygen -= fire_consumption_rate * dt
- Diffusion: oxygen moves toward neighbors with lower values
- If oxygen == 0 and voxel is fire → fire becomes smoke/ash
```

No flood fill. No connectivity checks. Just diffusion. Closed room → oxygen depletes naturally because no source. Open window → oxygen flows in from sky-connected chain. Emergent behavior from simple rules.

### 2.7 Falling Sand Specifics

Material types have a `phase` property: Solid, Powder, Liquid, Gas.

Within a Margolus 2×2×2 block:
```
Priority: heavier materials sink, lighter rise
  - Powder: tries to move down, or diag-down
  - Liquid: tries to move down, then sideways (with fill-level later)
  - Gas: tries to move up, or diag-up
  - Solid: doesn't move (only MPM moves solids)

Displacement: if liquid meets powder below it → swap (powder sinks in liquid)
Density ordering: stone > sand > water > oil > gas
```

### 2.8 Milestone 2 Deliverable

Massive world where:
- Heat diffuses visually (place lava next to ice → ice melts into water → water heats up → steam)
- Sand falls, water flows (basic CA, not fill-level yet)
- Chemical reactions work (wood + fire → fire spreads → ash)
- Oxygen depletes in closed spaces → fire goes out
- All at 60fps with 200+ dirty chunks

---

## Phase 3: PB-MPM Integration

**Goal**: Prove that PB-MPM zones can coexist with the CA world.

### 3.1 PB-MPM Solver (replace existing MLS-MPM)

Reference: EA SEED paper (SIGGRAPH 2024), open-source WebGPU implementation at `github.com/electronicarts/pbmpm` (BSD-3).

Key difference from our current MLS-MPM: instead of explicit time integration (which requires tiny dt for stability), PB-MPM iteratively projects constraints. Each iteration = one P2G + grid solve + G2P cycle, but the result is **unconditionally stable at any timestep**.

```
PB-MPM per frame (2-4 iterations):
  for iter in 0..NUM_ITERATIONS:
    clear_grid()
    p2g_scatter()        // particles → grid: mass + momentum
    grid_update()        // apply gravity, boundary conditions
    g2p_gather()         // grid → particles: update positions + velocities
    apply_constraints()  // material-specific: density for liquid, elasticity for solid
```

Material models (constitutive laws) are simpler than in standard MPM:
- **Liquid**: density constraint (particles push apart if too close)
- **Elastic solid**: co-rotational constraint via polar decomposition (same SVD we already have)
- **Sand/granular**: Drucker-Prager yield surface (clamp shear stress)
- **Gas**: weak density constraint + buoyancy

Port from the WebGPU reference to Vulkan compute / rust-gpu. The core algorithm is ~200 lines of shader code.

### 3.2 Activation Triggers (CA → PB-MPM)

After CA tick, a check pass scans dirty chunks for conditions that require particle physics:

```rust
enum ActivationTrigger {
    Explosion { center: Vec3, radius: f32, impulse: f32 },
    StructuralCollapse { chunk_id: u32 },  // bonds broken, disconnected piece detected
    LargeFluidVolume { chunk_id: u32 },    // too much liquid for CA to handle well
    PlayerAction { pos: Vec3, action: ActionType },
}
```

When triggered:
1. Allocate PB-MPM zone (grid + particle buffer from pre-allocated pool)
2. Convert voxels → particles in the zone (spawn pass)
3. Run PB-MPM each frame until zone deactivates

**Zone limit**: start with max 4 simultaneous zones. If more needed → time_scale decreases.

### 3.3 Voxel → Particle Spawning

```
For each voxel in activation zone:
  if voxel is not air:
    spawn 1 particle (or 2³=8 for high quality, configurable)
    particle.position = voxel_center + small_jitter
    particle.velocity = vec3(0) (or inherited from trigger impulse)
    particle.mass = material_table[material].density * voxel_volume / particles_per_voxel
    particle.volume = voxel_volume / particles_per_voxel
    particle.F = mat3::identity()  // MUST be identity — no deformation history from static voxel
    particle.C = mat3::zero()      // APIC affine momentum
    particle.material = voxel.material_id
    particle.temperature = voxel.temperature

  mark voxel as "owned by physics zone" (don't CA-update it)
```

### 3.4 Particle → Voxel Sleeping

```
For each particle:
  if |velocity| < SLEEP_THRESHOLD for SLEEP_FRAMES consecutive frames:
    // Find nearest voxel position
    voxel_pos = round(particle.position / voxel_size)
    
    // Write back to chunk
    set_voxel(voxel_pos, particle.material, particle.temperature)
    
    // Restore bonds with solid neighbors
    for each neighbor direction:
      if neighbor is solid of compatible material:
        set_bond(voxel_pos, direction, true)
    
    // Remove particle (add to freelist)
    particle.active = false

// When all particles in a zone are sleeping → deallocate zone
```

**Visual transition**: fade particle rendering out and voxel rendering in over 4-8 frames to avoid popping.

### 3.5 Bond Breaking + Connected Component Labeling (CCL)

When bonds break (from impact, reaction, or CA trigger):
1. Mark broken bonds in the voxel data
2. Run incremental CCL in the local area to find disconnected pieces
3. Each disconnected piece becomes a shape-matching cluster (rigid body) or individual particles

CCL algorithm: **block-based union-find** on GPU. Run on a local region (AABB around damage + margin), not the whole world.

For POC: start with a simple CPU-side flood fill for CCL. GPU CCL is an optimization for later.

### 3.6 Milestone 3 Deliverable

- Place an explosion in the CA world → PB-MPM zone activates → debris flies → particles settle → sleep back into voxels → zone deactivates
- Lava flow (CA) meets explosion zone (PB-MPM) → particles interact with CA boundary
- Multiple simultaneous zones work without crashes
- Time scale adapts to load

---

## Phase 4: Structural Integrity

**Goal**: Buildings collapse when support is removed.

### 4.1 BFS Weight Propagation

Event-driven, not per-frame. Trigger: when voxels are destroyed or bonds break.

```
For each grounded block (touching bedrock or other grounded block):
  support_remaining = material.max_load

BFS outward from grounded blocks:
  for each neighbor:
    cost = neighbor.mass
    if support_remaining >= cost:
      neighbor.grounded = true
      neighbor.support_remaining = support_remaining - cost
      add neighbor to BFS queue
    else:
      neighbor.grounded = false  // unsupported!

Unsupported blocks: break bonds → CCL → pieces fall (PB-MPM or shape-matching)
```

Material support values (game-balance, not physics):
```
Stone:   mass=5,  max_load=100  → supports 20 blocks horizontally
Wood:    mass=3,  max_load=30   → supports 10 blocks
Metal:   mass=8,  max_load=200  → supports 25 blocks
Dirt:    mass=4,  max_load=10   → supports 2 blocks (collapses easily)
```

### 4.2 Incremental Recalculation

Don't recalculate the entire world. Only recalculate within a region:
1. Find AABB of destroyed blocks
2. Expand AABB by max_support_distance
3. Run BFS only within expanded AABB
4. Budget: max N blocks per frame, spread across multiple frames if needed

### 4.3 Milestone 4 Deliverable

- Build a bridge → walk on it → it holds
- Remove support pillars → bridge collapses section by section
- Tunnel under a building → building falls into the hole
- Cascading collapse: remove one load-bearing wall → multiple floors fall

---

## Phase 5: Kinematic Mechanisms

**Goal**: Rotor blocks, hinges, doors that interact with the voxel world.

### 5.1 Mechanism Blocks

Special voxel types that create mechanical connections:

```rust
enum MechanismType {
    Hinge { axis: Axis },       // doors, drawbridges
    Bearing { axis: Axis },     // rotors, wheels
    Piston { axis: Axis },      // linear actuators
    Motor { axis: Axis, torque: f32 },  // powered rotation
}
```

When player places a mechanism block → system detects connected groups on each side → creates a kinematic cluster.

### 5.2 Kinematic Cluster Update

```
Per frame, for each mechanism:
  1. Compute torque/force balance:
     input_torque = motor_power + external_forces (wind, water, player)
     resistance = friction + load_from_attached_blocks
     angular_acceleration = (input_torque - resistance) / moment_of_inertia
     angular_velocity += angular_acceleration * dt
     angular_velocity *= damping  // 0.98-0.99, prevents Lord Clang
  
  2. Rotate/translate attached block group:
     new_transform = old_transform * rotation_matrix(angular_velocity * dt)
  
  3. Check interactions with world:
     For each voxel overlapped by the moving group:
       if drill/saw: damage the voxel (reduce bonds/health → break → debris)
       if collision: apply resistance force, possibly break mechanism if overloaded
```

Cost: one matrix multiply + one intersection test per mechanism per frame. Hundreds of mechanisms = trivial.

### 5.3 Mechanism Breaking

If load exceeds mechanism strength → break the mechanism → both sides become independent physics objects → PB-MPM zone if needed.

### 5.4 Milestone 5 Deliverable

- Build a windmill with a bearing block → it rotates from wind
- Attach a drill to a motor → it drills through stone
- Overload a mechanism → it breaks spectacularly

---

## GPU Pipeline: Per-Frame Dispatch Order

```
Frame Start
│
├── vkCmdPipelineBarrier (if uploads pending)
│
├── Pass 0: Stream Compaction (dirty chunks → compacted list)
│   dispatch(compact_dirty, ceil(2048/256))
│   barrier(compute→compute)
│
├── Pass 1: Thermal Diffusion
│   dispatchIndirect(ca_thermal, &dirty_dispatch_args)
│   barrier(compute→compute)
│
├── Pass 2: Oxygen Diffusion  
│   dispatchIndirect(ca_oxygen, &dirty_dispatch_args)
│   barrier(compute→compute)
│
├── Pass 3a: Margolus CA Pass A (even partition)
│   dispatchIndirect(ca_margolus_even, &dirty_dispatch_args)
│   barrier(compute→compute)
│
├── Pass 3b: Margolus CA Pass B (odd partition)
│   dispatchIndirect(ca_margolus_odd, &dirty_dispatch_args)
│   barrier(compute→compute)
│
├── Pass 4: Activation Check (does any dirty chunk need PB-MPM?)
│   dispatchIndirect(check_activations, &dirty_dispatch_args)
│   barrier(compute→compute)
│
├── Pass 5: PB-MPM (only if active zones exist)
│   ├── 5a: Spawn particles for new zones
│   │   dispatch(spawn_particles, new_zone_particle_count/256)
│   │   barrier(compute→compute)
│   │
│   ├── 5b: PB-MPM iterations (×2-4)
│   │   for iter in 0..NUM_ITERS:
│   │     dispatch(clear_grid, grid_cells/256)
│   │     barrier
│   │     dispatch(p2g, active_particles/256)
│   │     barrier
│   │     dispatch(grid_update, grid_cells/256)
│   │     barrier
│   │     dispatch(g2p, active_particles/256)
│   │     barrier
│   │
│   └── 5c: Sleep check (particles → voxels)
│       dispatch(sleep_check, active_particles/256)
│       barrier(compute→compute)
│
├── Pass 6: Mechanism Update
│   dispatch(update_mechanisms, mechanism_count)
│   barrier(compute→compute)
│
├── Pass 7: Structural Integrity (if damage occurred)
│   dispatch(si_bfs, affected_region_size/256)
│   barrier(compute→compute)
│
├── Pass 8: Update Dirty Flags + Ghost Cells (if used)
│   dispatchIndirect(update_flags, &dirty_dispatch_args)
│   barrier(compute→transfer)
│
├── Pass 9: Chunk Upload/Download (streaming)
│   vkCmdCopyBuffer for pending uploads/downloads
│
└── Render (separate queue or after compute)
```

### Time Management

```rust
// Adaptive time scaling — world slows down under load, never breaks
let target_physics_ms = 10.0; // leave 6ms for render at 60fps
let actual_physics_ms = gpu_timestamp_query_result();

if actual_physics_ms > target_physics_ms {
    time_scale *= 0.95; // slow down
} else {
    time_scale = (time_scale * 1.02).min(1.0); // speed back up
}

let world_dt = real_dt * time_scale;

// CA uses fixed sub-steps
let ca_steps = (accumulated_ca_time / CA_FIXED_DT).floor() as u32;
accumulated_ca_time -= ca_steps as f32 * CA_FIXED_DT;

// PB-MPM uses world_dt directly (unconditionally stable)
```

---

## Material System

### Material Table (GPU buffer, 1024 entries)

```rust
struct MaterialProperties {
    // Phase & movement
    phase: u8,              // 0=solid, 1=powder, 2=liquid, 3=gas
    density: u8,            // 0-255, for sinking/floating order
    viscosity: u8,          // 0-255, how fast liquid flows (0=water, 200=lava, 255=tar)
    
    // Thermal
    conductivity: u8,       // 0-10, heat transfer speed
    heat_capacity: u8,      // 0-10, resistance to temperature change
    melt_temp: u8,          // temperature threshold → liquid
    boil_temp: u8,          // temperature threshold → gas
    freeze_temp: u8,        // temperature threshold → solid
    melt_into: u16,         // material ID when melting
    boil_into: u16,         // material ID when boiling
    freeze_into: u16,       // material ID when freezing
    
    // Combustion
    flammability: u8,       // 0=fireproof, 255=flash paper
    ignition_temp: u8,      // min temperature to catch fire
    burn_into: u16,         // material when burned out (ash, nothing)
    burn_heat: u8,          // heat output while burning
    
    // Structural
    strength: u8,           // 0-255, bond break threshold
    max_load: u16,          // structural support capacity
    mass: u8,               // weight for SI calculations
    
    // Chemistry
    acid_resistance: u8,    // 0=dissolves instantly, 255=immune
    
    // Physics (for PB-MPM)
    bulk_modulus: f32,       // liquid compressibility (only used in PB-MPM)
    elastic_mu: f32,         // shear modulus for solids (only used in PB-MPM)
    elastic_lambda: f32,     // Lamé parameter (only used in PB-MPM)
    
    // Visual
    color: u32,             // RGBA packed
    emission: u8,           // 0-255, glow intensity
}
```

### Initial Material Set (POC)

```
ID  Name        Phase    Notes
0   Air         -        empty voxel
1   Stone       Solid    strong, high melt temp, low conductivity
2   Dirt        Solid    weak, low melt temp
3   Sand        Powder   falls, turns to glass at high temp
4   Water       Liquid   standard fluid behavior
5   Lava        Liquid   very hot, high viscosity, melts stone
6   Ice         Solid    melts into water at low temp
7   Wood        Solid    flammable, medium strength
8   Metal       Solid    strong, high conductivity, high melt temp
9   Glass       Solid    brittle (low strength, high hardness)
10  Fire        Gas      timer-based, heats neighbors, consumes oxygen
11  Smoke       Gas      rises, fades after timer
12  Steam       Gas      hot gas, condenses to water when cooled
13  Gunpowder   Powder   very flammable, explodes (triggers PB-MPM)
14  Oil         Liquid   flammable, floats on water (lower density)
15  Acid        Liquid   dissolves materials based on acid_resistance
16  Ash         Powder   result of burning
17  Obsidian    Solid    very strong, formed when lava cools fast
18  Coal        Solid    flammable, burns longer than wood
19  Gravel      Powder   heavier than sand, falls
```

### Reaction Table (POC subset)

```
A          B          Condition   → A        → B        Heat  Impulse
Water      Lava       -           Steam      Stone      +100  weak
Water      Fire       -           Steam      Ash        -50   -
Fire       Wood       oxygen>2    Fire       Fire       +30   -
Fire       Wood       oxygen<=2   Smoke      Coal       +5    -
Fire       Gunpowder  -           (removed)  (removed)  +200  EXPLOSION
Lava       Ice        -           Stone      Water      +50   weak
Lava       Wood       -           Lava       Fire       +40   -
Lava       Stone      temp>230    Lava       Lava       -10   -
Acid       Metal      -           Acid       (removed)  +10   -
Acid       Stone      -           Acid       Sand       +5    -
Acid       Glass      -           (removed)  (removed)  -     -
Water      Sand       -           Water      Clay*      -     -
Oil        Fire       oxygen>2    Fire       Fire       +50   -
```

*Clay is material ID 20, to be added.

---

## Testing & Validation Strategy

### Stress Tests (run these regularly)

1. **Scale test**: 200×200×20 chunks loaded, fly through at max speed. Must not crash, VRAM must stay in budget.

2. **Thermal cascade**: Place lava on top of ice mountain. Watch chain reaction: lava→melts ice→water→flows down→cools lava→stone. Must propagate correctly without stalling.

3. **Explosion chain**: 100 gunpowder voxels in a line. One ignites. Chain of explosions must propagate, PB-MPM zones must spawn and despawn correctly, no memory leaks.

4. **Structural collapse**: Build a 50-block bridge, remove center support. Must collapse in sections, not all at once and not stay floating.

5. **Oxygen depletion**: Seal a room, light a fire. Fire must eventually go out. Open a hole → fire restarts.

6. **Time scaling**: Trigger 10 explosions simultaneously. Frame rate must stay above 15fps (time scaling kicks in), no crashes.

7. **Long-running**: Leave fire spreading through a forest (1000+ burning voxels) for 5 minutes. Must not grow unboundedly, must not run out of memory.

### Determinism Test (for future multiplayer)

Run the same world state with the same inputs twice (on the same GPU). Compare CA state after 1000 ticks. Must be bit-identical (because all CA uses integer math).

---

## Key Technical Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| 4 bytes per voxel (not 8) | Memory is the bottleneck for scale. 4B fits the essential data. |
| uint8 temperature (not float) | Deterministic, saves 3 bytes, 256 levels is enough for game design |
| Integer CA math | Cross-GPU determinism for multiplayer. Float only in PB-MPM. |
| Margolus 2-pass (not 4-pass checkerboard) | Simpler, proven, natural conservation laws. |
| 32³ chunks (not 16³ or 64³) | 32 = warp size on NVIDIA. 128KB fits in L2. 64³ = 1MB too big. |
| Gigabuffer (not per-chunk VkBuffer) | Eliminates descriptor set management, enables simple indexing. |
| Graceful time dilation (not hard caps) | Core principle: every rule works 100% of the time, scale via slowdown. |
| PB-MPM (not MLS-MPM) | Unconditional stability means fewer iterations, larger dt, no CFL limit. |
| BFS structural integrity (not FEM) | Simple, fast, good enough for game. FEM is correct but too expensive. |
| No fill-level water in POC | Defer complexity. Basic falling-sand water proves the architecture. |

---

## Open Questions (to resolve during implementation)

1. **Ghost cells or direct neighbor reads?** Start without ghost cells, profile boundary access patterns, add if needed.

2. **Particles per voxel at spawn?** Start with 1. If PB-MPM quality is poor, try 4 or 8. More particles = more compute per zone.

3. **CCL on CPU or GPU?** Start on CPU (simpler). Move to GPU block-based union-find if it becomes a bottleneck.

4. **Reaction table lookup**: hash map vs sorted array? Profile both. Table is small (~4K entries), binary search might be faster due to cache.

5. **PB-MPM iteration count**: 2 or 4? Start with 2, increase if quality is poor. Each iteration is one full P2G+G2P cycle.

6. **Chunk size with ghost cells**: 32³ data + 1-voxel border = 34³ = 157KB. Still fits in L2 but uses 23% more memory. Worth it?

---

## References

- **PB-MPM**: EA SEED SIGGRAPH 2024. Paper: `media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf`. Code: `github.com/electronicarts/pbmpm`
- **Margolus CA**: Hahm 2023 "A Framework of Artificial Matter", Temple University. Code: `github.com/ccrock4t/3DCellularWorld`
- **GPU Falling Sand**: `github.com/GelamiSalami/GPU-Falling-Sand-CA`, `meatbatgames.com/blog/falling-sand-gpu/`
- **Rust GPU Falling Sand**: `github.com/ARez2/sandengine`
- **Fill-Level Water** (future): W-Shadow algorithm: `w-shadow.com/blog/2009/09/01/simple-fluid-simulation/`
- **Pipe Model Water** (future): Mei, Decaudin, Hu "Fast Hydraulic Erosion Simulation and Visualization on GPU" 2007
- **VRAGE3 Volumetric Water**: `blog.marekrosa.org/2022/04/vrage-volumetric-water.html`
- **Stream Compaction**: Raph Levien's decoupled look-back: `raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html`
- **Teardown Multiplayer**: `blog.voxagon.se/2026/03/13/teardown-multiplayer.html`
- **GPU Determinism**: `shaderfun.com/2020/10/25/understanding-determinism-part-1-intro-and-floating-points/`
- **VMA Ring Buffers**: `gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/custom_memory_pools.html`
- **Vulkan Timeline Semaphores**: `github.com/nvpro-samples/vk_timeline_semaphore`
- **Tom Forsyth CA**: `tomforsyth1000.github.io/papers/cellular_automata_for_physical_modelling.html`
- **Structural Integrity**: 7 Days to Die block SI system, Space Engineers VRage SI
- **Noita GDC Talk**: "Exploring the Tech and Design of Noita" (GDC 2019)
