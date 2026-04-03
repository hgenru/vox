# Performance Refactor v2: O(active) Simulation

## Problem

Current pipeline is O(total): every shader dispatches for ALL particles and ALL grid cells,
even though 80-90% of the world is typically sleeping. Sleep is "soft" (early return in shader),
GPU still pays for scheduling every thread.

Rendering is 50% of GPU time — brute-force ray march for every pixel every frame.

Dense grid clear writes ~2GB/frame of zeros.

## Goals

1. True O(active) simulation — zero GPU cost for sleeping regions
2. Render only what changed (temporal reuse)
3. Sparse memory — don't pay for empty space
4. Multi-rate physics — different processes at different frequencies
5. Prepare architecture for air-as-material (high particle counts)
6. **Work on modest GPUs (6-8GB VRAM)** — not just RTX 4090

## Target Hardware

The system MUST scale down to GPUs with 6-8GB VRAM (GTX 1060, RTX 3060, etc).
RTX 4090 (24GB) is the dev machine, not the target.

This means:
- Flat 768MB grid (256³) is already too much for 6GB GPUs
- Sparse data structures are ESSENTIAL, not optional optimizations
- Dynamic VRAM budget: query available memory, adapt limits at startup
- World size limited by content, not by fixed buffer allocations
- Streaming/LOD is core architecture

## Design Principles

- Physics tied to absolute time (seconds), NOT tick rate
- Tick rate is just "how often we compute" — can be dynamic
- Brick (8x8x8) is the fundamental scheduling unit
- "If it looks right, it IS right" — perceptual quality over physical accuracy for distant regions

---

## Phase 1: Quick Wins (no architecture change)

### 1.1 Conditional Render
If no active bricks AND camera hasn't moved → skip render, reuse previous frame.
- CPU-side check before recording render dispatch
- Readback a single `any_active` flag from GPU (set in update_sleep via atomic OR)
- **Expected win: ~50% GPU for static scenes**
- **Effort: 1 day**

### 1.2 Voxelize Sleep Skip
Add `should_skip_brick()` check to voxelize shader. Sleeping particles don't move,
their voxels don't change.
- **Expected win: ~5% GPU**
- **Effort: 2 hours**

### 1.3 Multi-Rate Scheduling
Different physics systems run at different real-time intervals:

| System | Interval | Current | Savings |
|--------|----------|---------|---------|
| Mechanics (P2G/G2P) | every substep | every substep | 0 (baseline) |
| Reactions (react) | 100ms (0.1s) | every frame | ~5x less |
| Temperature diffusion | 200ms | every substep | ~12x less |
| Sleep/activity update | 100ms | every substep | ~6x less |
| Brick occupancy | 200ms | every substep | ~12x less |

Implementation: `elapsed_since_last_X` timer on CPU, skip recording dispatch if not due.
All accumulators use real seconds (from `Instant::now()`), not frame counts.
- **Expected win: ~10-15% GPU**
- **Effort: 1 day**

### 1.4 Sparse Voxel Clear
Instead of dense clear of entire voxel grid (256MB), clear only bricks that were dirty.
Same pattern as existing `clear_grid_sparse` — maintain dirty brick list.
- **Expected win: ~5-8% GPU**
- **Effort: 2-3 days**

### 1.5 Remove Dense Grid Clear
We already have `clear_grid_sparse` + `mark_active` pipeline. The dense clear_grid pass
is redundant but was kept "for safety". Switch fully to sparse clear.
Requires careful validation that no stale data leaks.
- **Expected win: ~7% GPU**
- **Effort: 1 day + testing**

**Phase 1 total: ~40-50% GPU savings, 1-2 weeks work**

---

## Phase 2: Per-Brick Dispatch (the big one)

### Core idea
Instead of dispatching ALL particles and doing early-return for sleeping ones,
sort particles by brick and dispatch ONLY active bricks' particle ranges.

### 2.1 Counting Sort by Brick ID

Each frame (or every N frames when world is mostly static):

```
Pass 1: Count particles per brick
  - brick_count[brick_id(particle.pos)] += 1  (atomic)
  
Pass 2: Prefix sum over brick_count → brick_offset table
  - brick_offset[0] = 0
  - brick_offset[i] = brick_offset[i-1] + brick_count[i-1]
  
Pass 3: Scatter particles to sorted positions
  - sorted_index = atomicAdd(brick_offset_copy[brick_id], 1)
  - sorted_particles[sorted_index] = particle
```

3 compute passes. PB-MPM (EA SEED) reports <1ms for 1M particles.

**Open question:** sort particle data itself (move 144 bytes) or sort indices (move 4 bytes)?
Index indirection = less memory bandwidth for sort, more for P2G/G2P (random access).
Probably sort full data for memory coherence in P2G/G2P — they're the bottleneck.

### 2.2 Active Brick Compaction

After counting sort:
```
Pass 4: Compact active brick list
  - If brick_count[i] > 0 AND !sleeping[i]:
    active_bricks[atomicAdd(active_count, 1)] = i
```

### 2.3 Per-Brick Dispatch for P2G/G2P

**Option A: Multi indirect dispatch**
One indirect dispatch per active brick. Each dispatch: workgroups = ceil(brick_particle_count / 64).
VK_EXT_multi_draw_indirect or manual indirect chain.

**Option B: wgsparkl-style workgroup-per-brick**  
One workgroup per active brick. Inside workgroup: loop over brick's particles.
Use shared memory for grid cell accumulation (no global atomics!).
Brick = 8x8x8 = 512 cells. Workgroup = 64 threads. Each thread handles ~8 cells.

Option B is better: eliminates global atomics in P2G, better memory access pattern,
natural for brick-level parallelism. This is what Dimforge does in wgsparkl.

**Key challenge:** P2G writes to 27 neighboring cells. Particles near brick boundary
write to adjacent bricks' cells. Solutions:
- a) Halo cells: each brick's shared memory includes 1-cell border from neighbors
- b) Global atomics only for cross-brick writes (rare, ~surface particles only)
- c) Expand active region to include neighbor bricks of active bricks

### 2.4 Sparse Grid (hash map)

Replace flat `grid[256^3]` with hash-map grid:
- Only cells touched by active particles exist
- Insert during P2G, remove after G2P
- Zero clear cost (just drop entries)
- Memory: O(active_cells) instead of O(grid_size^3)

Implementation: open-addressing hash table on GPU.
Key = packed (x,y,z), value = GridCell (48 bytes).
Table size = next_power_of_2(max_active_cells * 2) for low collision rate.

**This is needed for air.** If air is explicit particles, grid goes from 5% occupied
to potentially 100%. Sparse grid keeps memory proportional to actual content.

### 2.5 Skip Sort When Stable

Counting sort is O(N) but not free. When world is mostly sleeping:
- Track `any_particle_changed_brick` flag during G2P
- If no particle crossed a brick boundary → skip sort next frame
- Stale sort is still valid if particles stay in their bricks

**Phase 2 total: 60-80% GPU savings for mixed active/sleeping scenes, 2-3 weeks work**

---

## Phase 3: Smart Rendering

### 3.1 Dirty-Tile Rendering

Divide screen into tiles (e.g. 16x16 pixels). Track which bricks are dirty.
Project dirty bricks to screen space → mark dirty tiles.
Only re-trace rays for dirty tiles, reuse previous frame for clean tiles.

Requires:
- Previous frame color buffer (already have render_output_buffer)
- Per-tile dirty flag (small buffer, ~5K entries for 1280x720)
- Modified render shader: check dirty flag, early-out if clean

**Expected win: 30-50% render cost when <50% of world is active**

### 3.2 Hierarchical Brick Skip (render acceleration)

Current: flat occupancy check per 8^3 brick.
Better: 2-level hierarchy:
- Level 0: 8^3 brick occupancy (existing)
- Level 1: 64^3 super-brick occupancy (8 bricks per axis = 4^3 = 64 super-bricks for 256^3)

Ray skips entire 64^3 super-brick if empty. Cuts ray steps by ~8x for empty regions.
Trivial to compute from existing brick occupancy (OR over 8 bricks).

### 3.3 Temporal Reprojection

When camera moves slightly: reproject previous frame using motion vectors.
Only re-trace rays where reprojection fails (disocclusion, moved voxels).

More complex, probably not needed until we have larger worlds or higher res.

**Phase 3 total: 30-50% render savings, 1-2 weeks work**

---

## Phase 4: Predictive & LOD (ongoing)

### 4.1 Neighbor Wake-Up

When a brick is active, also wake face-adjacent neighbors (6 neighbors).
Prevents 1-frame penetration when falling object approaches sleeping surface.

Optional: velocity-directed wake-up (particle moving down → wake brick below).

### 4.2 Camera-Distance LOD

Bricks further from camera get higher tick_period:
```
distance < 4 bricks:   full rate
distance 4-16 bricks:  half rate
distance > 16 bricks:  quarter rate

effective_period = max(sleep_period, distance_period)
```

### 4.3 Player Movement Prediction

If player is flying in direction V with speed S:
- Pre-wake bricks along trajectory V * lookahead_time
- Pre-compute render data for upcoming view frustum

### 4.4 Simplified Far-Field Physics

Far from camera: skip stress tensor in P2G, use simplified gravity-only advection.
Different shader variant for "far" bricks. Cheaper per-particle cost.

---

## Air Material Considerations

Air as explicit particles creates a problem: particle count explodes.
256^3 grid fully filled = 16.7M particles (MAX_PARTICLES is currently much less).

Options:
1. **Implicit air:** "no particle = air." Chemistry checks absence of particles.
   Cheapest, but can't model wind, convection, air pressure.
   
2. **Sparse air:** Air particles only where needed — near surfaces, in enclosed spaces,
   where wind/convection matters. Rest is implicit vacuum/air.
   
3. **Grid-based air:** Air isn't particles — it's a separate fluid grid (Euler).
   MPM particles interact with air grid. Classic coupled Euler-Lagrange.
   Most physically correct but doubles the grid work.

4. **Coarse air grid:** Air on a coarser grid (e.g. 64^3 instead of 256^3).
   Each air cell = 4^3 voxels. Sufficient for wind/convection, cheap to compute.

Recommendation: start with option 2 (sparse air) for chemistry, upgrade to option 4
(coarse air grid) for wind/convection later.

---

## Review Questions for Sub-Agents

### For GPU Agent (gpu-core owner):
- Sparse hash-map grid: open addressing vs chaining? Max load factor?
- Prefix sum implementation: single-pass CUB-style or multi-pass?
- Indirect dispatch chain performance: one dispatch per brick vs batched?
- Memory budget: how much VRAM do we actually use now vs available?

### For Render Agent (shaders owner):
- Per-brick P2G with shared memory: how to handle boundary particles?
- Counting sort: sort full particle structs (144B) or sort indices (4B)?
- Dirty-tile rendering: integration with current render pipeline?
- Hierarchical brick skip: perf impact on ray march inner loop?

### For Sim Agent (shared/sim-cpu owner):
- Multi-rate scheduling: how does variable-rate temperature affect stability?
- Air material: implicit vs explicit — impact on reaction system design?
- Sparse grid CPU reference implementation for testing?
- Phase transition correctness with reduced-rate temperature updates?

---

## Success Metrics

- **Static scene:** <2ms GPU per frame (currently ~16ms)
- **10% active:** <5ms GPU per frame
- **50% active:** <10ms GPU per frame  
- **Particle budget:** support 500K+ particles at 60fps with 10% active
- **Memory:** GPU memory proportional to world content, not grid size
