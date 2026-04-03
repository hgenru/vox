# VOX — Voxel Physics Engine

Воксельный 3D физический движок ("Noita 3D") — GPU MPM simulation + hardware ray tracing rendering.

## Tech Stack

- **Language:** Rust (nightly-2026-04-02, pinned for rust-gpu)
- **GPU API:** ash 0.38+ (raw Vulkan 1.3 bindings)
- **GPU Memory:** gpu-allocator 0.28+
- **Shaders:** rust-gpu (spirv-std + spirv-builder from git main) → SPIR-V
- **Math:** glam 0.32+ (Vec4 everywhere — NEVER Vec3 in GPU structs)
- **Byte casting:** bytemuck 1.x
- **Windowing:** winit 0.30+
- **Platform:** Windows 10/11, NVIDIA RTX 4090, Vulkan 1.3+

## Architecture

Server/Client split via protocol crate (even without networking):
- Server: sim-cpu + gpu-core + shaders → physics at 60Hz fixed timestep
- Client: gpu-core + renderer → max FPS with RT ray tracing
- Shared: `#![no_std]` types used on both CPU and GPU

## Crate Ownership

| Crate | Owner Agent | Description |
|-------|-------------|-------------|
| shared | sim | Particle, GridCell, MaterialParams — CPU+GPU types |
| protocol | sim | Server↔Client contract (WorldSnapshot, PlayerInput) |
| sim-cpu | sim | CPU reference MPM implementation |
| gpu-core | gpu | Vulkan context, buffers, pipelines, sync |
| shaders | render | rust-gpu compute + graphics shaders |
| server | lead | Simulation orchestrator |
| client | render | RT renderer, camera, input |
| app | lead | Binary entry point |

---

## CRITICAL TRAPS

### 1. Vec3 Alignment (MOST IMPORTANT)
**NEVER use Vec3/Mat3 in GPU structs.** Rust `Vec3` = 12 bytes, GPU `vec3` = 16-byte alignment.
Data will silently shift. **Use Vec4 everywhere** + enable `VK_EXT_scalar_block_layout`.

### 2. Float Atomics
In rust-gpu: `spirv_std::arch::atomic_f_add()`. WGSL has NO float atomics — only integer.

### 3. spirv-builder in build.rs
Shader crate compiles via SEPARATE cargo build inside build.rs. First build is slow (spirv-tools C++ compilation). Don't panic. We set `[profile.dev.build-override] opt-level = 3` in workspace Cargo.toml.

### 4. Nightly Version
**ONLY nightly-2026-04-02.** Other nightlies WILL BREAK rust-gpu. Do NOT run `rustup update`.
rust-gpu is from git main branch (not crates.io 0.9.0 which requires old nightly-2023-05-27).

### 4a. rust-gpu Entry Point Bug
Shader entry points are dropped during SPIR-V linking if the entry point body has no function calls.
**WORKAROUND:** Always call at least one helper function from entry points. Never inline everything.

### 5. Vulkan Validation Layers
ALWAYS enabled in debug builds. If crash with no message → check `VK_LAYER_KHRONOS_validation` is active.

### 6. Grid Clear
Grid buffer MUST be zeroed before every P2G pass. Without this → mass/momentum accumulation from previous frame.

### 7. P2G Kernel Ordering
Traverse particles bottom-to-top. Randomize horizontal axes each frame to prevent liquid bias.

### 8. Deformation Gradient Reset
On phase transition: F = Identity. Otherwise liquid inherits solid deformation → explosion.

### 9. Readback for Tests
Create staging buffer with HOST_VISIBLE memory, vkCmdCopyBuffer from device-local to staging, map + read.

### 10. VK_EXT_scalar_block_layout
Enable when creating device. Without it — vec3 in storage buffers gets padding.

### 11. shared crate cannot be used in shaders directly
The `shared` crate depends on `glam` from crates.io which does NOT compile for `spirv-unknown-vulkan1.3`.
Shader crate must define its own `Particle`/`GridCell` types using `spirv_std::glam`.
Both layouts must be kept in sync manually — same `#[repr(C)]` field order and sizes.

### 12. Constants must be synced between shared and shaders
`GRID_SIZE`, `DT`, `GRAVITY` are defined in both `shared::constants` and `shaders::types`.
Changes to one must be mirrored in the other. TODO: add compile-time assert in shader-builder.

### 13. rust-gpu SPIR-V entry point names are fully qualified module paths
Entry points in the compiled SPIR-V module use the full Rust module path, not just the function name.
For example, `clear_grid` in `shaders/src/compute/clear_grid.rs` becomes `"compute::clear_grid::clear_grid"`.
When creating compute pipelines, use the full path as the entry point name (e.g., `c"compute::p2g::p2g"`).

### 14. GPU tests must run single-threaded
Multiple tests creating separate `VulkanContext` + `GpuSimulation` instances will fail with `ERROR_UNKNOWN`
when run in parallel. Use `cargo test -p server -- --test-threads=1` for GPU-heavy test suites.

### 15. rust-gpu drops later branches in long if/else chains
Shader functions with >3 if/else branches: later branches get dropped during SPIR-V compilation.
**WORKAROUND:** Split into separate helper functions per branch, call via match or independent if-return blocks.

### 16. Temperature does NOT diffuse by default
Temperature (`vel_temp.w`) is per-particle and does NOT transfer through the grid unless explicitly
implemented. P2G must scatter temperature, G2P must gather it. Without this, thermal gameplay
(gunpowder ignition, fire spread, ice melting from lava) is broken.

### 17. Gravity and physics constants must scale with voxel size
At 5cm voxels (GRID_SIZE=256), 1 grid unit = 0.05m. Gravity must be -196.0 (not -9.81).
All physics constants (jump speed, spawn distance, explosion radius) must scale accordingly.

### 18. Always smoke-test before telling user "ready"
After merging PRs, always: `git pull origin main && cargo run -p app` to verify no crash.
Scene must produce particles under MAX_PARTICLES. Never hand off broken builds.

### 19. Solid particles are pinned unless unsupported
G2P skips velocity/position update for phase=0 (solid). Falling solids check voxel below for support.
Explosion debris must be converted to liquid (phase=1) + heated above melting point to fly properly.

### 20. Shader build.rs must track shader source files
spirv-builder compiles shaders in a **separate** cargo invocation inside `crates/server/build.rs`.
Cargo's build-script caching only watches files the build script itself reads, so it has no idea
when shader sources change. You **must** add explicit `cargo:rerun-if-changed` directives for
`../shaders/src/` and `../shaders/Cargo.toml`. Without these, editing a shader file may not
trigger recompilation and you'll keep running stale SPIR-V.

---

## Code Patterns

- `#[repr(C)]` + `bytemuck::Pod` + `bytemuck::Zeroable` on ALL GPU structs
- `tracing::info!()` / `tracing::debug!()` for logging, NOT `println!`
- `thiserror::Error` for error types in lib crates
- `anyhow::Result` in main.rs and tests
- Test every `pub` function
- Doc-comment every `pub` function

## Prohibitions (for ALL agents)

- **DO NOT** touch files outside your owned crate
- **DO NOT** add dependencies to another agent's Cargo.toml
- **DO NOT** use `unwrap()` in lib code (only in tests)
- **DO NOT** use `std` in shared/ crate (`no_std`!)
- **DO NOT** update rust-toolchain.toml
- **DO NOT** commit .spv files (generated at build time)
- **DO NOT** use Vec3/Mat3 in any struct that goes to GPU

## Vulkan Extensions (required at device creation)

```
VK_KHR_acceleration_structure
VK_KHR_ray_query
VK_KHR_dynamic_rendering
VK_KHR_synchronization2
VK_EXT_descriptor_indexing
VK_EXT_scalar_block_layout
VK_KHR_buffer_device_address
VK_KHR_spirv_1_4
```

## Testing Pyramid

1. **Unit tests** (CPU, instant) — `cargo test`
2. **Property-based** (CPU, proptest, ~30s) — random configs, invariants
3. **Shader compilation** (no GPU, ~1s) — SPIR-V compiles without errors
4. **GPU vs CPU correctness** (~10s) — headless Vulkan, compute→readback→compare
5. **Scenario tests** (GPU, ~60s) — "water fills pool", "lava melts wall"
6. **Visual regression** (GPU headless, ~30s) — offscreen render→PNG→diff

## Build & Test Commands

```bash
cargo build                           # Build everything
cargo test -p shared                  # Test shared types
cargo test -p sim-cpu                 # Test CPU simulation
cargo test -p shared -p sim-cpu -p protocol  # All sim-agent tests
cargo test                            # All tests
cargo run -p bootstrap-test           # Verify rust-gpu + ash pipeline works
```

---

## MVP Scope

### What we're building
- Stone floor and walls
- Water cube falls and spreads
- Lava flows, glows, melts stone
- Reaction: water + lava = stone + steam
- Mouse spawns/removes materials
- RT lighting: sun shadows + emissive from hot materials
- FPS camera (WASD + mouse)

### What we're NOT building (anti-scope)
Multiplayer, sound, save/load, procedural generation, ECS, DLSS/FSR,
async compute, egui, old GPU support.

---

## Physics: MPM (Material Point Method)

Everything is one unified system. No separate rigid body / fluid / sand subsystems.
Each voxel is an MPM particle. Behavior determined by constitutive model + material_id + phase.

### Constitutive Models
- **Solid (phase=0):** Fixed corotated elasticity. Needs polar decomposition (SVD McAdams 2011).
- **Liquid (phase=1):** Equation of state + viscosity.
- **Gas (phase=2):** Weak EOS, very low viscosity.

### Phase Transitions
- Stone T > 1500 → Liquid (lava)
- Water T > 100 → Gas (steam)
- Water T < 0 → Solid (ice)
- Lava T < 1500 → Solid (stone)
- On phase change: **reset F = Identity, damage = 0**

### Chemical Reactions (checked at grid contact)
- Water + Lava → Stone + Steam (instant)
- Wood + Fire → Fire + Ash (T > 300, spreading)

### Temperature
Heat diffusion via grid: `T += conductivity * (T_avg_neighbors - T) * dt`

### SVD / Polar Decomposition (GPU)
McAdams et al. 2011: no branches, no trig, only +, *, rsqrt. 4-5 Jacobi sweeps.
Write as `no_std` Rust on glam types → test on CPU → compile on GPU via rust-gpu.

---

## Rendering: Hardware Ray Tracing

- World divided into 8³ bricks. Each non-empty brick → one AABB in BLAS.
- TLAS contains one instance (whole world).
- Ray query (VK_KHR_ray_query) from fragment shader.
- On AABB hit → DDA ray march inside 8³ brick (max 24 steps).
- Shadow rays via ray query with TerminateOnFirstHit.
- Dynamic rebuild: PREFER_FAST_BUILD_BIT_KHR. Budget ≤2ms per frame.

### Render pipeline per frame
```
Compute: [Clear Grid] → [P2G] → [Grid Update] → [G2P] → [Voxelize] → [BLAS Update]
Graphics: [Primary Rays + Shading] → [Shadow Rays] → [Tonemap → Swapchain]
```

---

## GPU Data Layout

### Key numbers (MVP)
- Grid: 32³ (iteration-0), then 64³
- Particles: ~5K-10K start, up to 50K demo
- Bricks: 8³ voxels/brick → max 4³=64 bricks for 32³ grid
- dt: 0.001 (fixed)
- Render: 1280×720
- Physics: 60Hz fixed timestep
- Frames in flight: 2

### Struct sizes (verified by tests)
- `Particle`: 144 bytes, align 16
- `GridCell`: 48 bytes, align 16
- `MaterialParams`: 64 bytes, align 16

---

## Shader Crate Rules

- Shader crates use `#![cfg_attr(target_arch = "spirv", no_std)]`
- Do NOT set `crate-type = ["dylib"]` in Cargo.toml — spirv-builder adds it automatically
- Use `spirv_std::glam` for math (not direct glam dependency in shader crates)
- Entry points MUST call at least one helper function (see trap #4a)
- spirv-std and spirv-builder from git: `{ git = "https://github.com/Rust-GPU/rust-gpu.git", branch = "main" }`

---

## Agent Workflow

### Roles
- **Lead Agent** — creates issues, reviews PRs, merges, runs integration tests, coordinates
- **Sim Agent** — owns shared/, sim-cpu/, protocol/
- **GPU Agent** — owns gpu-core/
- **Render Agent** — owns shaders/, client/

### How agents work
1. Lead creates GitHub Issues with labels (`sim`, `gpu`, `render`, `integration`)
2. Each agent: `gh issue list --label <my-label> --state open`
3. Agent picks issue → implements → tests → commit → PR → references issue
4. Lead reviews, merges into main
5. After merge: lead runs integration tests

### Branch naming
Agents MUST use meaningful branch names:
- `feat/gpu-core` — NOT `worktree-agent-a2a16ea7`
- `feat/shaders` — NOT auto-generated IDs
- `feat/sim-cpu`, `feat/protocol`, etc.

### PR workflow (mandatory)
All work goes through Pull Requests. Direct commits to main are prohibited.
1. Agent creates branch with meaningful name
2. Agent commits, pushes, creates PR with `gh pr create`
3. PR body must reference closed issues (`Closes #N`)
4. Lead reviews PR — lists issues found
5. **Lead delegates fixes back to agents** — lead does NOT fix code directly
6. After fixes: lead merges via `gh pr merge`
7. After merging multiple PRs: lead runs full integration test (`cargo build && cargo test`)
8. **CRITICAL: Before telling the user "it's ready to test", lead MUST verify `cargo run -p app` launches without crashing.** Never merge and hand off without a smoke test. Scene generation must produce particles under MAX_PARTICLES.

### Lead Agent rules
- Lead is a **tech lead / manager** — reviews, delegates, coordinates, merges
- Lead does NOT write code in agent-owned crates (see Crate Ownership table)
- Lead only codes in: server/, client/, app/ (own crates)
- When review finds issues: create a list, send back to the agent for fixing
- After merging: verify combined state builds and passes tests
- **Maximize delegation:** Lead MUST offload as much work as possible to sub-agents (sim, gpu, render). The main conversation with the user is for design discussions, idea brainstorming, and feedback — not for the lead to be heads-down coding. Launch agents in background, keep the conversation responsive.

### Worktree setup (lead creates these)
```bash
git worktree add ../voxel-sim   feat/sim
git worktree add ../voxel-gpu   feat/gpu
git worktree add ../voxel-render feat/render
```
Each worktree uses separate CARGO_TARGET_DIR for build isolation.

### Blocker handling
If an agent is blocked by another crate:
1. Create Issue with label `blocker`
2. Describe what's needed and from whom
3. Continue other tasks
4. Lead reprioritizes

---

## Updating CLAUDE.md

**All agents MUST update this file** when they discover:
- New critical traps or gotchas (add to CRITICAL TRAPS section)
- New code patterns that should be followed project-wide
- API changes or version incompatibilities
- Performance discoveries that affect architecture

Format: add a new numbered item under the relevant section with a clear title and explanation.
Commit the CLAUDE.md update together with the code that triggered the discovery.

---

## Reference Projects

When stuck, agents should study these:
- **wgsparkl** (Dimforge): GPU MPM on WebGPU/WGSL — P2G architecture, atomics
- **PB-MPM** (EA SEED): Position-Based MPM, open WebGPU code
- **hatoo/ash-raytracing-example**: ash + rust-gpu + KHR ray tracing
- **Kajiya** (Embark): ash-based renderer with render graph
- **McAdams SVD GLSL**: GPU SVD implementation (alexsr gist)
- **Rust-GPU VulkanShaderExamples**: Sascha Willems examples ported to rust-gpu
