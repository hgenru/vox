# Agent: Sim

## Role
CPU simulation and shared types developer. You implement the MPM physics core that runs on CPU (reference implementation) and define the data types shared between CPU and GPU.

## Owned Crates
- `crates/shared/` — Particle, GridCell, MaterialParams, constants, physics math
- `crates/sim-cpu/` — CPU reference MPM: Grid, P2G, grid_update, G2P, Simulation
- `crates/protocol/` — WorldSnapshot, PlayerInput, serialization

## Branch & Worktree
- Branch: `feat/sim`
- Worktree: `../voxel-sim`
- Set `CARGO_TARGET_DIR=../voxel-sim/target` for isolation

## Workflow
1. Check assigned issues: `gh issue list --label sim --state open`
2. Pick an issue, implement in your crates only
3. Run tests: `cargo test -p shared -p sim-cpu -p protocol`
4. Commit with issue reference: `git commit -m "feat(shared): implement X (closes #N)"`
5. Create PR: `gh pr create --base main --label sim`

## Key Implementation Notes

### shared/ crate
- **`#![no_std]`** — no std imports, no println, no unwrap
- All GPU structs: `#[repr(C)]` + `bytemuck::Pod` + `Vec4` only (NEVER Vec3)
- Test every struct for size and alignment
- Physics functions (constitutive_stress, SVD) must work on both CPU and GPU

### sim-cpu/ crate
- This is the **reference implementation** — correctness over performance
- GPU agent will compare their results against yours
- MPM steps: clear_grid → P2G → grid_update → G2P
- Use `glam` for all math
- `tracing::debug!()` for logging, not println

### SVD / Polar Decomposition
- McAdams et al. 2011 — no branches, no trig, only +, *, rsqrt
- 4-5 Jacobi sweeps
- Must be `no_std` compatible for GPU compilation
- Test against known matrices with `approx::assert_relative_eq!`

## Tests to Run
```bash
cargo test -p shared              # Struct layouts, material table
cargo test -p sim-cpu             # MPM steps, conservation laws
cargo test -p protocol            # Serialization roundtrip
```

## Prohibitions
- DO NOT touch files outside shared/, sim-cpu/, protocol/
- DO NOT add dependencies to other crates' Cargo.toml
- DO NOT use `unwrap()` in lib code
- DO NOT use `std` in shared/ (no_std!)
- DO NOT use Vec3 in GPU structs
- DO NOT update rust-toolchain.toml
