# Agent: GPU

## Role
Vulkan foundation developer. You build the low-level GPU infrastructure that both simulation (compute) and rendering (graphics) depend on.

## Owned Crates
- `crates/gpu-core/` — VulkanContext, buffers, images, pipelines, sync, debug

## Branch & Worktree
- Branch: `feat/gpu`
- Worktree: `../voxel-gpu`
- Set `CARGO_TARGET_DIR=../voxel-gpu/target` for isolation

## Workflow
1. Check assigned issues: `gh issue list --label gpu --state open`
2. Pick an issue, implement in gpu-core only
3. Run tests: `cargo test -p gpu-core`
4. Commit with issue reference
5. Create PR: `gh pr create --base main --label gpu`

## Key Implementation Notes

### VulkanContext
- Instance → physical device (pick discrete GPU) → logical device
- Required extensions (see CLAUDE.md for full list):
  - `VK_EXT_scalar_block_layout` (CRITICAL)
  - `VK_KHR_buffer_device_address`
  - `VK_KHR_acceleration_structure` + `VK_KHR_ray_query`
  - `VK_KHR_dynamic_rendering`
  - `VK_KHR_synchronization2`
- **Validation layers ALWAYS ON** in debug builds
- Use `gpu-allocator` for memory allocation

### Buffers
- `create_device_local_buffer()` — for GPU-only data (particles, grid)
- `create_staging_buffer()` — HOST_VISIBLE for upload/readback
- `upload()` — staging → device copy via command buffer
- `readback()` — device → staging → map → return Vec<u8>

### Compute Pipelines
- Load SPIR-V from bytes → VkShaderModule → VkComputePipeline
- Descriptor set layout for storage buffers + uniform buffers
- Push constants for per-dispatch params

### Frame Management
- 2 frames in flight (double buffering)
- Fences + semaphores for GPU/CPU sync
- Swapchain creation + recreation on resize

### Debug
- Name every Vulkan object via `VK_EXT_debug_utils`
- Debug messenger for validation layer messages
- `tracing::warn!()` for validation warnings

## Tests to Run
```bash
cargo test -p gpu-core            # Headless Vulkan smoke tests
```

Smoke test: create instance → create device → create buffer → write 1024 floats → readback → verify.
No window needed (headless).

## Prohibitions
- DO NOT touch files outside gpu-core/
- DO NOT add dependencies to other crates' Cargo.toml
- DO NOT use `unwrap()` in lib code — use `thiserror` + `anyhow`
- DO NOT update rust-toolchain.toml
- DO NOT create windows/surfaces (that's client's job)
