# VOX — Voxel Physics Engine

Воксельный 3D физический движок ("Noita 3D") — GPU MPM simulation + hardware ray tracing rendering.

## Tech Stack

- **Language:** Rust (nightly-2025-06-23, pinned for rust-gpu)
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
```
