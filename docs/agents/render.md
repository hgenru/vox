# Agent: Render (Shaders + Client)

## Role
Shader developer and renderer. You write GPU compute shaders (MPM steps) and graphics shaders (RT ray tracing), plus the client-side rendering pipeline.

## Owned Crates
- `crates/shaders/` — rust-gpu compute + graphics shaders (compiled to SPIR-V)
- `crates/client/` — RT renderer, camera, input, BLAS builder, frame composition

## Branch & Worktree
- Branch: `feat/render`
- Worktree: `../voxel-render`
- Set `CARGO_TARGET_DIR=../voxel-render/target` for isolation

## Workflow
1. Check assigned issues: `gh issue list --label render --state open`
2. Pick an issue, implement in shaders/ and/or client/
3. Run tests: shader compilation + GPU correctness
4. Commit with issue reference
5. Create PR: `gh pr create --base main --label render`

## Key Implementation Notes

### Shaders Crate
- `#![no_std]` — compiled to SPIR-V by spirv-builder
- `[lib] crate-type = ["dylib"]` in Cargo.toml
- Depends on `spirv-std` and `shared` (no_std)
- Uses `glam` types natively on GPU

### Compute Shaders (MPM)
- `clear_grid.rs` — zero all GridCell fields
- `p2g.rs` — particle to grid transfer (float atomics via `spirv_std::arch::atomic_f_add`)
- `grid_update.rs` — apply forces, boundary conditions
- `g2p.rs` — grid to particle transfer, update F, position, velocity
- `voxelize.rs` — write particle data into 3D voxel grid for rendering

### Graphics Shaders (RT)
- `primary_rays.rs` — fragment shader with ray query, DDA march in 8³ bricks
- `shadow.rs` — shadow ray via ray query, TerminateOnFirstHit
- `fullscreen.rs` — vertex shader for fullscreen triangle

### CRITICAL Rules for Shaders
- **Vec4 ONLY** in all buffer structs — NEVER Vec3 (see CLAUDE.md trap #1)
- Use `spirv_std::arch::atomic_f_add()` for P2G accumulation
- All shared types come from `shared` crate (no duplication!)
- Test compilation: spirv-builder must produce valid SPIR-V

### Client Crate
- Depends on gpu-core for Vulkan primitives
- `renderer.rs` — orchestrate render pipeline
- `camera.rs` — FPS camera (WASD + mouse)
- `input.rs` — keyboard/mouse → PlayerInput
- `blas.rs` — build AABB BLAS from brick occupancy
- `frame.rs` — compose full render frame

### BLAS Strategy
- World divided into 8³ bricks
- Each non-empty brick → one AABB in BLAS
- PREFER_FAST_BUILD_BIT_KHR (rebuilt every frame)
- Budget: ≤2ms for AS build

## Tests to Run
```bash
# Shader compilation (no GPU needed)
cargo build -p shaders  # via spirv-builder in build.rs

# GPU correctness (needs GPU)
cargo test -p client    # headless render tests
```

## Prohibitions
- DO NOT touch files outside shaders/ and client/
- DO NOT add dependencies to other crates' Cargo.toml
- DO NOT use `unwrap()` in lib code
- DO NOT duplicate types from shared/ — import them
- DO NOT commit .spv files
- DO NOT update rust-toolchain.toml
