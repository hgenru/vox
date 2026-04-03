//! GPU-side hash grid operations for sparse MPM grid.
//!
//! Provides lock-free open-addressing hash map with linear probing.
//! The hash grid replaces the dense flat grid array (256^3 = 768MB)
//! with a compact hash table (~52MB at 1M capacity).
//!
//! Two parallel arrays form the hash map:
//! - `keys: [u32; capacity]` — packed (x,y,z) coordinates or EMPTY sentinel
//! - `values: [GridCell; capacity]` — the actual grid cell data
//!
//! All operations are lock-free using atomic compare-exchange for inserts.

/// Empty sentinel for hash grid keys (must match shared::HASH_GRID_EMPTY_KEY).
pub const HASH_GRID_EMPTY_KEY: u32 = 0xFFFF_FFFF;

/// Maximum number of linear probes before giving up.
pub const HASH_GRID_MAX_PROBES: u32 = 128;

/// Pack (x, y, z) grid coordinates into a single u32 key.
///
/// For grid sizes up to 1024, packs into 10+10+10 = 30 bits.
/// Bits: [29:20] = z, [19:10] = y, [9:0] = x.
/// The top 2 bits are zero, so packed keys never equal EMPTY (0xFFFFFFFF).
pub fn pack_key(x: u32, y: u32, z: u32) -> u32 {
    (z & 0x3FF) << 20 | (y & 0x3FF) << 10 | (x & 0x3FF)
}

/// Unpack a u32 key back into (x, y, z) grid coordinates.
///
/// Inverse of `pack_key`. Returns (x, y, z).
pub fn unpack_key(key: u32) -> (u32, u32, u32) {
    let x = key & 0x3FF;
    let y = (key >> 10) & 0x3FF;
    let z = (key >> 20) & 0x3FF;
    (x, y, z)
}

/// Hash function for packed 3D coordinate keys.
///
/// Uses a multiply-shift hash with good avalanche properties for 3D coords.
/// The result is masked to [0, capacity) where capacity must be a power of 2.
pub fn hash_key(key: u32, capacity: u32) -> u32 {
    // Murmur3-style finalizer: good distribution for sequential 3D coords
    let mut h = key;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h & (capacity - 1) // capacity must be power of 2
}

/// Insert a key into the hash grid. Returns the slot index.
///
/// Uses open addressing with linear probing and atomic compare-exchange.
/// If the key already exists, returns the existing slot index.
/// If the table is full after MAX_PROBES, returns HASH_GRID_EMPTY_KEY.
///
/// # Safety
/// The caller must ensure concurrent atomic access is properly synchronized
/// via SPIR-V semantics. The keys buffer must be initialized to EMPTY.
pub unsafe fn hash_grid_insert(keys: &mut [u32], key: u32, capacity: u32) -> u32 {
    let mut slot = hash_key(key, capacity);
    let mut probes = 0u32;
    while probes < HASH_GRID_MAX_PROBES {
        let existing = atomic_compare_exchange_key(keys, slot as usize, HASH_GRID_EMPTY_KEY, key);
        // We inserted successfully (slot was empty)
        if existing == HASH_GRID_EMPTY_KEY {
            return slot;
        }
        // Key already exists at this slot
        if existing == key {
            return slot;
        }
        // Collision: linear probe to next slot
        slot = (slot + 1) & (capacity - 1);
        probes += 1;
    }
    // Table is too full — should not happen with proper sizing
    HASH_GRID_EMPTY_KEY
}

/// Lookup a key in the hash grid. Returns slot index or EMPTY if not found.
///
/// Uses open addressing with linear probing. Stops at the first EMPTY slot
/// or after MAX_PROBES attempts.
pub fn hash_grid_lookup(keys: &[u32], key: u32, capacity: u32) -> u32 {
    let mut slot = hash_key(key, capacity);
    let mut probes = 0u32;
    while probes < HASH_GRID_MAX_PROBES {
        let existing = keys[slot as usize];
        if existing == key {
            return slot;
        }
        if existing == HASH_GRID_EMPTY_KEY {
            return HASH_GRID_EMPTY_KEY;
        }
        slot = (slot + 1) & (capacity - 1);
        probes += 1;
    }
    HASH_GRID_EMPTY_KEY
}

/// Atomic compare-exchange on a u32 in a storage buffer.
///
/// On SPIR-V targets uses `spirv_std::arch::atomic_compare_exchange`.
/// On CPU (for testing) uses a simple non-atomic compare-exchange.
///
/// Returns the previous value at the slot.
///
/// # Safety
/// The caller must ensure `keys[index]` is a valid location and that
/// concurrent atomic access is properly synchronized via SPIR-V semantics.
#[inline]
unsafe fn atomic_compare_exchange_key(keys: &mut [u32], index: usize, compare: u32, value: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        // SCOPE=1 (Device), SEMANTICS_EQUAL=0x0 (Relaxed), SEMANTICS_UNEQUAL=0x0
        unsafe {
            spirv_std::arch::atomic_compare_exchange::<u32, 1u32, 0x0u32, 0x0u32>(
                &mut keys[index],
                compare,
                value,
            )
        }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let old = keys[index];
        if old == compare {
            keys[index] = value;
        }
        old
    }
}

#[cfg(not(target_arch = "spirv"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        for z in [0, 1, 255, 512, 1023] {
            for y in [0, 1, 127, 256, 1023] {
                for x in [0, 1, 63, 128, 1023] {
                    let key = pack_key(x, y, z);
                    let (ux, uy, uz) = unpack_key(key);
                    assert_eq!((ux, uy, uz), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn test_pack_key_never_equals_empty() {
        // All valid coordinates produce keys != EMPTY
        let key = pack_key(1023, 1023, 1023);
        assert_ne!(key, HASH_GRID_EMPTY_KEY);
        // Max possible packed value: 30 bits set = 0x3FFFFFFF
        assert!(key <= 0x3FFF_FFFF);
    }

    #[test]
    fn test_hash_key_in_range() {
        let capacity = 1024u32; // power of 2
        for i in 0..1000 {
            let key = pack_key(i % 256, (i / 256) % 256, i / 65536);
            let h = hash_key(key, capacity);
            assert!(h < capacity);
        }
    }

    #[test]
    fn test_insert_and_lookup() {
        let capacity = 64u32;
        let mut keys = vec![HASH_GRID_EMPTY_KEY; capacity as usize];

        let key1 = pack_key(10, 20, 30);
        let key2 = pack_key(5, 15, 25);

        // Insert key1
        let slot1 = unsafe { hash_grid_insert(&mut keys, key1, capacity) };
        assert_ne!(slot1, HASH_GRID_EMPTY_KEY);

        // Insert key2
        let slot2 = unsafe { hash_grid_insert(&mut keys, key2, capacity) };
        assert_ne!(slot2, HASH_GRID_EMPTY_KEY);

        // Slots should be different
        assert_ne!(slot1, slot2);

        // Lookup should find them
        assert_eq!(hash_grid_lookup(&keys, key1, capacity), slot1);
        assert_eq!(hash_grid_lookup(&keys, key2, capacity), slot2);

        // Lookup missing key should return EMPTY
        let key3 = pack_key(100, 100, 100);
        assert_eq!(hash_grid_lookup(&keys, key3, capacity), HASH_GRID_EMPTY_KEY);
    }

    #[test]
    fn test_insert_idempotent() {
        let capacity = 64u32;
        let mut keys = vec![HASH_GRID_EMPTY_KEY; capacity as usize];
        let key = pack_key(7, 8, 9);

        let slot1 = unsafe { hash_grid_insert(&mut keys, key, capacity) };
        let slot2 = unsafe { hash_grid_insert(&mut keys, key, capacity) };
        assert_eq!(slot1, slot2); // Same key -> same slot
    }
}
