//! Toolbar overlay compute shader.
//!
//! Draws a Minecraft-style material toolbar at the bottom-center of the render
//! output buffer. Each material slot is rendered as a colored square with a gap
//! between slots. The currently selected slot gets a bright white border.
//!
//! Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
//! Runs AFTER the main render pass, compositing on top of existing pixels.

use spirv_std::glam::{UVec3, Vec4};
use spirv_std::spirv;

/// Maximum number of material slots supported by the toolbar.
const MAX_MATERIALS: usize = 8;

/// Size of each material slot square in pixels.
const SLOT_SIZE: u32 = 48;

/// Gap between slots in pixels.
const SLOT_GAP: u32 = 4;

/// Distance from the bottom edge of the screen to the toolbar.
const BOTTOM_MARGIN: u32 = 16;

/// Border width for the selected slot highlight (pixels).
const BORDER_WIDTH: u32 = 3;

/// Border width for unselected slot outlines (pixels).
const OUTLINE_WIDTH: u32 = 1;

/// Half-length of each crosshair arm in pixels.
const CROSSHAIR_ARM: u32 = 10;

/// Thickness of each crosshair line in pixels (each side from center).
const CROSSHAIR_HALF_THICK: u32 = 1;

/// Push constants for the toolbar overlay shader.
///
/// Contains screen dimensions, selection state, and material colors.
/// Total size: 16 bytes (header) + 128 bytes (colors) = 144 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ToolbarParams {
    /// Render target width in pixels.
    pub screen_width: u32,
    /// Render target height in pixels.
    pub screen_height: u32,
    /// Index of the currently selected material slot (0-based).
    pub selected_index: u32,
    /// Number of material slots to display.
    pub material_count: u32,
    /// RGBA colors for each material slot (up to 8).
    pub colors: [Vec4; MAX_MATERIALS],
}

/// Pack RGBA color components (0.0-1.0 float) into a u32 for B8G8R8A8 format.
///
/// Byte order matches the render shader's `pack_bgra`: B in lowest byte,
/// then G, R, A in little-endian u32.
fn pack_bgra_f32(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let ri = ((r * 255.0) as u32).min(255);
    let gi = ((g * 255.0) as u32).min(255);
    let bi = ((b * 255.0) as u32).min(255);
    let ai = ((a * 255.0) as u32).min(255);
    bi | (gi << 8) | (ri << 16) | (ai << 24)
}

/// Alpha-blend a foreground color over a background color.
///
/// Both colors are packed B8G8R8A8 u32 values. Returns the blended result.
/// Uses standard "source over" compositing: out = fg * fg_a + bg * (1 - fg_a).
fn alpha_blend(bg: u32, fg_r: f32, fg_g: f32, fg_b: f32, fg_a: f32) -> u32 {
    let bg_b = (bg & 0xFF) as f32 / 255.0;
    let bg_g = ((bg >> 8) & 0xFF) as f32 / 255.0;
    let bg_r = ((bg >> 16) & 0xFF) as f32 / 255.0;

    let out_r = fg_r * fg_a + bg_r * (1.0 - fg_a);
    let out_g = fg_g * fg_a + bg_g * (1.0 - fg_a);
    let out_b = fg_b * fg_a + bg_b * (1.0 - fg_a);

    pack_bgra_f32(out_r, out_g, out_b, 1.0)
}

/// Draw a crosshair (+ symbol) at the center of the screen.
///
/// The crosshair is a white semi-transparent plus sign with configurable arm
/// length and thickness. It helps the player aim when placing or removing
/// materials.
pub fn draw_crosshair_pixel(
    px: u32,
    py: u32,
    push: &ToolbarParams,
    output: &mut [u32],
) {
    let width = push.screen_width;
    let height = push.screen_height;

    if px >= width || py >= height {
        return;
    }

    let cx = width / 2;
    let cy = height / 2;

    // Check horizontal bar: center_y +/- thickness, center_x +/- arm length
    // Note: saturating_sub is not available in SPIR-V, use manual underflow guard
    let in_h_bar = py + CROSSHAIR_HALF_THICK >= cy
        && py <= cy + CROSSHAIR_HALF_THICK
        && px + CROSSHAIR_ARM >= cx
        && px <= cx + CROSSHAIR_ARM;

    // Check vertical bar: center_x +/- thickness, center_y +/- arm length
    let in_v_bar = px + CROSSHAIR_HALF_THICK >= cx
        && px <= cx + CROSSHAIR_HALF_THICK
        && py + CROSSHAIR_ARM >= cy
        && py <= cy + CROSSHAIR_ARM;

    if in_h_bar || in_v_bar {
        let pixel_idx = (py * width + px) as usize;
        let bg = output[pixel_idx];
        // White with moderate transparency so it doesn't obscure the scene
        output[pixel_idx] = alpha_blend(bg, 1.0, 1.0, 1.0, 0.75);
    }
}

/// Determine what to draw at a given pixel and write to the output buffer.
///
/// This is the main per-pixel logic, separated from the entry point as required
/// by the rust-gpu linker bug workaround (CLAUDE.md trap #4a).
pub fn draw_toolbar_pixel(
    px: u32,
    py: u32,
    push: &ToolbarParams,
    output: &mut [u32],
) {
    let width = push.screen_width;
    let height = push.screen_height;

    if px >= width || py >= height {
        return;
    }

    let mat_count = push.material_count.min(MAX_MATERIALS as u32);
    if mat_count == 0 {
        return;
    }

    // Toolbar total dimensions
    let toolbar_width = mat_count * SLOT_SIZE + (mat_count - 1) * SLOT_GAP;
    let toolbar_height = SLOT_SIZE;

    // Toolbar position (centered horizontally, near bottom)
    let toolbar_x = (width - toolbar_width) / 2;
    let toolbar_y = height - BOTTOM_MARGIN - toolbar_height;

    // Check if pixel is in the toolbar bounding box (with some padding for border)
    let pad = BORDER_WIDTH;
    if px + pad < toolbar_x || px >= toolbar_x + toolbar_width + pad {
        return;
    }
    if py + pad < toolbar_y || py >= toolbar_y + toolbar_height + pad {
        return;
    }

    let pixel_idx = (py * width + px) as usize;

    // Determine which slot this pixel is in (if any)
    let rel_x = px as i32 - toolbar_x as i32;
    let rel_y = py as i32 - toolbar_y as i32;

    // Check each slot
    let mut slot_idx: u32 = 0;
    while slot_idx < mat_count {
        let slot_left = (slot_idx * (SLOT_SIZE + SLOT_GAP)) as i32;
        let slot_right = slot_left + SLOT_SIZE as i32;
        let slot_top = 0i32;
        let slot_bottom = SLOT_SIZE as i32;

        let is_selected = slot_idx == push.selected_index;
        let bw = if is_selected {
            BORDER_WIDTH as i32
        } else {
            OUTLINE_WIDTH as i32
        };

        // Extended slot rect including border
        let ext_left = slot_left - bw;
        let ext_right = slot_right + bw;
        let ext_top = slot_top - bw;
        let ext_bottom = slot_bottom + bw;

        if rel_x >= ext_left && rel_x < ext_right && rel_y >= ext_top && rel_y < ext_bottom {
            // Inside this slot's extended area
            let in_inner = rel_x >= slot_left
                && rel_x < slot_right
                && rel_y >= slot_top
                && rel_y < slot_bottom;

            if in_inner {
                // Draw material color with semi-transparent background panel
                let color = push.colors[slot_idx as usize];
                let bg = output[pixel_idx];
                // Dark background panel behind the color
                let panel_r = color.x * 0.85;
                let panel_g = color.y * 0.85;
                let panel_b = color.z * 0.85;
                output[pixel_idx] = alpha_blend(bg, panel_r, panel_g, panel_b, 0.9);
            } else {
                // Border region
                let bg = output[pixel_idx];
                if is_selected {
                    // Bright white border for selected slot
                    output[pixel_idx] = alpha_blend(bg, 1.0, 1.0, 1.0, 0.95);
                } else {
                    // Subtle dark outline for unselected slots
                    output[pixel_idx] = alpha_blend(bg, 0.15, 0.15, 0.15, 0.7);
                }
            }
            return;
        }

        slot_idx += 1;
    }

    // Pixel is in the toolbar bounding box padding area but not in any slot
    // Draw a semi-transparent dark background panel behind the whole toolbar
    let bg_left = toolbar_x as i32 - (BORDER_WIDTH as i32) - 4;
    let bg_right = (toolbar_x + toolbar_width) as i32 + BORDER_WIDTH as i32 + 4;
    let bg_top = toolbar_y as i32 - (BORDER_WIDTH as i32) - 4;
    let bg_bottom = (toolbar_y + toolbar_height) as i32 + BORDER_WIDTH as i32 + 4;

    let ipx = px as i32;
    let ipy = py as i32;

    if ipx >= bg_left && ipx < bg_right && ipy >= bg_top && ipy < bg_bottom {
        let bg = output[pixel_idx];
        output[pixel_idx] = alpha_blend(bg, 0.1, 0.1, 0.1, 0.5);
    }
}

/// Compute shader entry point: draw toolbar overlay on the render output buffer.
///
/// Dispatched with `(ceil(width/8), ceil(height/8), 1)` workgroups.
/// Descriptor set 0, binding 0: storage buffer of `u32` (pixel output, read-write).
/// Push constants: `ToolbarParams`.
#[spirv(compute(threads(8, 8, 1)))]
pub fn toolbar_overlay(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] push: &ToolbarParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] output: &mut [u32],
) {
    draw_crosshair_pixel(id.x, id.y, push, output);
    draw_toolbar_pixel(id.x, id.y, push, output);
}
