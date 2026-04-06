//! Packed u32 voxel type for the CA substrate.
//!
//! Each voxel is exactly 4 bytes, packed as:
//! - `material_id`: bits 0-9 (10 bits, 0-1023)
//! - `temperature`: bits 10-17 (8 bits, 0-255 game units)
//! - `bonds`: bits 18-23 (6 bits, +X/-X/+Y/-Y/+Z/-Z neighbor bonds)
//! - `oxygen`: bits 24-27 (4 bits, 0-15)
//! - `flags`: bits 28-31 (4 bits: dirty, burning, wet, poisoned)

use bytemuck::{Pod, Zeroable};

/// Packed voxel representation for the Cellular Automata substrate.
///
/// Layout: `[material_id:10][temperature:8][bonds:6][oxygen:4][flags:4]` = 32 bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Pod, Zeroable)]
#[repr(transparent)]
pub struct Voxel(pub u32);

// Bit layout constants
const MATERIAL_ID_BITS: u32 = 10;
const TEMPERATURE_BITS: u32 = 8;
const BONDS_BITS: u32 = 6;
const OXYGEN_BITS: u32 = 4;

const MATERIAL_ID_SHIFT: u32 = 0;
const TEMPERATURE_SHIFT: u32 = MATERIAL_ID_SHIFT + MATERIAL_ID_BITS; // 10
const BONDS_SHIFT: u32 = TEMPERATURE_SHIFT + TEMPERATURE_BITS; // 18
const OXYGEN_SHIFT: u32 = BONDS_SHIFT + BONDS_BITS; // 24
const FLAGS_SHIFT: u32 = OXYGEN_SHIFT + OXYGEN_BITS; // 28

const MATERIAL_ID_MASK: u32 = (1 << MATERIAL_ID_BITS) - 1; // 0x3FF
const TEMPERATURE_MASK: u32 = (1 << TEMPERATURE_BITS) - 1; // 0xFF
const BONDS_MASK: u32 = (1 << BONDS_BITS) - 1; // 0x3F
const OXYGEN_MASK: u32 = (1 << OXYGEN_BITS) - 1; // 0xF
const FLAGS_MASK: u32 = (1 << 4) - 1; // 0xF

// Bond bit indices within the bonds field
const BOND_POS_X: u32 = 0;
const BOND_NEG_X: u32 = 1;
const BOND_POS_Y: u32 = 2;
const BOND_NEG_Y: u32 = 3;
const BOND_POS_Z: u32 = 4;
const BOND_NEG_Z: u32 = 5;

// Flag bit indices within the flags field
const FLAG_DIRTY: u32 = 0;
const FLAG_BURNING: u32 = 1;
const FLAG_WET: u32 = 2;
const FLAG_POISONED: u32 = 3;

impl Voxel {
    /// Creates a new voxel with the given field values.
    ///
    /// Values are masked to their respective bit widths.
    pub fn new(material_id: u16, temperature: u8, bonds: u8, oxygen: u8, flags: u8) -> Self {
        let mut v = 0u32;
        v |= (material_id as u32 & MATERIAL_ID_MASK) << MATERIAL_ID_SHIFT;
        v |= (temperature as u32 & TEMPERATURE_MASK) << TEMPERATURE_SHIFT;
        v |= (bonds as u32 & BONDS_MASK) << BONDS_SHIFT;
        v |= (oxygen as u32 & OXYGEN_MASK) << OXYGEN_SHIFT;
        v |= (flags as u32 & FLAGS_MASK) << FLAGS_SHIFT;
        Self(v)
    }

    /// Creates an air voxel (all zeros).
    pub fn air() -> Self {
        Self(0)
    }

    /// Returns `true` if this voxel is air (material_id == 0).
    pub fn is_air(&self) -> bool {
        self.material_id() == 0
    }

    // --- Getters ---

    /// Returns the material ID (0-1023).
    pub fn material_id(&self) -> u16 {
        ((self.0 >> MATERIAL_ID_SHIFT) & MATERIAL_ID_MASK) as u16
    }

    /// Returns the temperature (0-255 game units).
    pub fn temperature(&self) -> u8 {
        ((self.0 >> TEMPERATURE_SHIFT) & TEMPERATURE_MASK) as u8
    }

    /// Returns the bond flags (6 bits).
    pub fn bonds(&self) -> u8 {
        ((self.0 >> BONDS_SHIFT) & BONDS_MASK) as u8
    }

    /// Returns the oxygen level (0-15).
    pub fn oxygen(&self) -> u8 {
        ((self.0 >> OXYGEN_SHIFT) & OXYGEN_MASK) as u8
    }

    /// Returns the flag bits (4 bits).
    pub fn flags(&self) -> u8 {
        ((self.0 >> FLAGS_SHIFT) & FLAGS_MASK) as u8
    }

    // --- Setters ---

    /// Sets the material ID (0-1023).
    pub fn set_material_id(&mut self, v: u16) {
        self.0 = (self.0 & !(MATERIAL_ID_MASK << MATERIAL_ID_SHIFT))
            | ((v as u32 & MATERIAL_ID_MASK) << MATERIAL_ID_SHIFT);
    }

    /// Sets the temperature (0-255).
    pub fn set_temperature(&mut self, v: u8) {
        self.0 = (self.0 & !(TEMPERATURE_MASK << TEMPERATURE_SHIFT))
            | ((v as u32 & TEMPERATURE_MASK) << TEMPERATURE_SHIFT);
    }

    /// Sets the bond flags (6 bits).
    pub fn set_bonds(&mut self, v: u8) {
        self.0 = (self.0 & !(BONDS_MASK << BONDS_SHIFT))
            | ((v as u32 & BONDS_MASK) << BONDS_SHIFT);
    }

    /// Sets the oxygen level (0-15).
    pub fn set_oxygen(&mut self, v: u8) {
        self.0 = (self.0 & !(OXYGEN_MASK << OXYGEN_SHIFT))
            | ((v as u32 & OXYGEN_MASK) << OXYGEN_SHIFT);
    }

    /// Sets the flag bits (4 bits).
    pub fn set_flags(&mut self, v: u8) {
        self.0 = (self.0 & !(FLAGS_MASK << FLAGS_SHIFT))
            | ((v as u32 & FLAGS_MASK) << FLAGS_SHIFT);
    }

    // --- Bond helpers ---

    fn get_bond(&self, bit: u32) -> bool {
        (self.bonds() >> bit) & 1 != 0
    }

    fn set_bond(&mut self, bit: u32, v: bool) {
        let mut b = self.bonds();
        if v {
            b |= 1 << bit;
        } else {
            b &= !(1 << bit);
        }
        self.set_bonds(b);
    }

    /// Returns `true` if the +X neighbor bond is set.
    pub fn has_bond_pos_x(&self) -> bool {
        self.get_bond(BOND_POS_X)
    }

    /// Sets or clears the +X neighbor bond.
    pub fn set_bond_pos_x(&mut self, v: bool) {
        self.set_bond(BOND_POS_X, v);
    }

    /// Returns `true` if the -X neighbor bond is set.
    pub fn has_bond_neg_x(&self) -> bool {
        self.get_bond(BOND_NEG_X)
    }

    /// Sets or clears the -X neighbor bond.
    pub fn set_bond_neg_x(&mut self, v: bool) {
        self.set_bond(BOND_NEG_X, v);
    }

    /// Returns `true` if the +Y neighbor bond is set.
    pub fn has_bond_pos_y(&self) -> bool {
        self.get_bond(BOND_POS_Y)
    }

    /// Sets or clears the +Y neighbor bond.
    pub fn set_bond_pos_y(&mut self, v: bool) {
        self.set_bond(BOND_POS_Y, v);
    }

    /// Returns `true` if the -Y neighbor bond is set.
    pub fn has_bond_neg_y(&self) -> bool {
        self.get_bond(BOND_NEG_Y)
    }

    /// Sets or clears the -Y neighbor bond.
    pub fn set_bond_neg_y(&mut self, v: bool) {
        self.set_bond(BOND_NEG_Y, v);
    }

    /// Returns `true` if the +Z neighbor bond is set.
    pub fn has_bond_pos_z(&self) -> bool {
        self.get_bond(BOND_POS_Z)
    }

    /// Sets or clears the +Z neighbor bond.
    pub fn set_bond_pos_z(&mut self, v: bool) {
        self.set_bond(BOND_POS_Z, v);
    }

    /// Returns `true` if the -Z neighbor bond is set.
    pub fn has_bond_neg_z(&self) -> bool {
        self.get_bond(BOND_NEG_Z)
    }

    /// Sets or clears the -Z neighbor bond.
    pub fn set_bond_neg_z(&mut self, v: bool) {
        self.set_bond(BOND_NEG_Z, v);
    }

    // --- Flag helpers ---

    fn get_flag(&self, bit: u32) -> bool {
        (self.flags() >> bit) & 1 != 0
    }

    fn set_flag(&mut self, bit: u32, v: bool) {
        let mut f = self.flags();
        if v {
            f |= 1 << bit;
        } else {
            f &= !(1 << bit);
        }
        self.set_flags(f);
    }

    /// Returns `true` if the dirty flag is set.
    pub fn is_dirty(&self) -> bool {
        self.get_flag(FLAG_DIRTY)
    }

    /// Sets or clears the dirty flag.
    pub fn set_dirty(&mut self, v: bool) {
        self.set_flag(FLAG_DIRTY, v);
    }

    /// Returns `true` if the burning flag is set.
    pub fn is_burning(&self) -> bool {
        self.get_flag(FLAG_BURNING)
    }

    /// Sets or clears the burning flag.
    pub fn set_burning(&mut self, v: bool) {
        self.set_flag(FLAG_BURNING, v);
    }

    /// Returns `true` if the wet flag is set.
    pub fn is_wet(&self) -> bool {
        self.get_flag(FLAG_WET)
    }

    /// Sets or clears the wet flag.
    pub fn set_wet(&mut self, v: bool) {
        self.set_flag(FLAG_WET, v);
    }

    /// Returns `true` if the poisoned flag is set.
    pub fn is_poisoned(&self) -> bool {
        self.get_flag(FLAG_POISONED)
    }

    /// Sets or clears the poisoned flag.
    pub fn set_poisoned(&mut self, v: bool) {
        self.set_flag(FLAG_POISONED, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn test_air_is_zero() {
        assert_eq!(Voxel::air().0, 0);
        assert!(Voxel::air().is_air());
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let v = Voxel::new(42, 200, 0b101010, 13, 0b1010);
        assert_eq!(v.material_id(), 42);
        assert_eq!(v.temperature(), 200);
        assert_eq!(v.bonds(), 0b101010);
        assert_eq!(v.oxygen(), 13);
        assert_eq!(v.flags(), 0b1010);
    }

    #[test]
    fn test_material_id_max() {
        let v = Voxel::new(1023, 0, 0, 0, 0);
        assert_eq!(v.material_id(), 1023);
    }

    #[test]
    fn test_temperature_max() {
        let v = Voxel::new(0, 255, 0, 0, 0);
        assert_eq!(v.temperature(), 255);
    }

    #[test]
    fn test_bonds_all_set() {
        let v = Voxel::new(0, 0, 0b111111, 0, 0);
        assert_eq!(v.bonds(), 0b111111);
        assert!(v.has_bond_pos_x());
        assert!(v.has_bond_neg_x());
        assert!(v.has_bond_pos_y());
        assert!(v.has_bond_neg_y());
        assert!(v.has_bond_pos_z());
        assert!(v.has_bond_neg_z());
    }

    #[test]
    fn test_flags_all_set() {
        let v = Voxel::new(0, 0, 0, 0, 0b1111);
        assert!(v.is_dirty());
        assert!(v.is_burning());
        assert!(v.is_wet());
        assert!(v.is_poisoned());
    }

    #[test]
    fn test_set_field_preserves_others() {
        let mut v = Voxel::new(100, 50, 0b110011, 7, 0b0101);
        v.set_temperature(200);
        assert_eq!(v.material_id(), 100);
        assert_eq!(v.temperature(), 200);
        assert_eq!(v.bonds(), 0b110011);
        assert_eq!(v.oxygen(), 7);
        assert_eq!(v.flags(), 0b0101);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let v: Voxel = Voxel::zeroed();
        assert!(v.is_air());
        assert_eq!(v.0, 0);
    }

    #[test]
    fn test_size_align() {
        assert_eq!(size_of::<Voxel>(), 4);
        assert_eq!(align_of::<Voxel>(), 4);
    }
}
