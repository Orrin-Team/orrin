//! The hasher the ECS uses for its own type- and entity-keyed maps.
//!
//! `HashMap`'s default is SipHash-1-3, which is the right default for keys that
//! arrive from outside the program. These keys do not: a `TypeId` is
//! compiler-generated and already well distributed, and an [`Entity`] is a slot
//! index. Against those, SipHash buys only HashDoS resistance for an attacker
//! who cannot choose the keys in the first place — and it is charged on every
//! [`World::get`], [`get_mut`](World::get_mut), and [`has`](World::has), which
//! the scripting FFI calls per component per entity per frame.
//!
//! This is rustc's FxHash accumulator with a splitmix64 finalizer on top — the
//! same trade-off the compiler makes for its interned-key maps, plus the
//! avalanche step plain FxHash omits. See [`FxHasher::finish`] for why the
//! finalizer is not optional here.
//!
//! [`Entity`]: crate::Entity
//! [`World::get`]: crate::World::get

use std::hash::{BuildHasherDefault, Hasher};

/// Multiplier from rustc's `FxHasher`: the fractional bits of the golden ratio.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// A fast, non-cryptographic hasher for keys the program generates itself.
///
/// Deliberately **not** HashDoS-resistant. Use it for `TypeId`s, [`Entity`]
/// handles, and other internal keys — never for anything an untrusted party
/// chooses.
///
/// Distribution is a real requirement, not a nicety: the engine's hottest maps
/// are keyed by small sequential integers, which is exactly the input a weakly
/// mixed hash spreads badly. See [`finish`](FxHasher::finish).
///
/// [`Entity`]: crate::Entity
#[derive(Clone, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    /// Rotate-xor-multiply. The rotate carries the previous word's high bits
    /// back into range before the xor, so word order matters; the multiply then
    /// spreads each word across the whole output.
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

// `TypeId`'s `Hash` impl has changed shape across rustc releases — a `u64`, a
// `u128`, with and without a trailing length prefix. Every `write_*` is covered
// here, so which one a given toolchain picks never matters.
impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = <[u8; 8]>::try_from(chunk).expect("chunks_exact(8) yields 8 bytes");
            self.add(u64::from_ne_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_ne_bytes(tail));
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_u128(&mut self, n: u128) {
        self.add(n as u64);
        self.add((n >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // splitmix64's finalizer. Plain FxHash returns the accumulator as-is,
        // which leaves structured keys poorly spread in the bits `hashbrown`
        // actually uses — a pair of small entity indices lands near its
        // neighbours, and the probe sequences that follow cost more than the
        // cheap hash saved. Measured on `collision_run`: without this, 5000
        // bodies were ~4% *slower* than SipHash while 100 were ~6% faster.
        let mut z = self.hash;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The [`BuildHasher`](std::hash::BuildHasher) for [`FxHasher`], for use as a
/// `HashMap`'s third type parameter.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// A `HashMap` over program-generated keys, hashed with [`FxHasher`].
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = FxHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn distinct_type_ids_do_not_collide() {
        struct A;
        struct B;
        struct C;
        let ids = [TypeId::of::<A>(), TypeId::of::<B>(), TypeId::of::<C>()];
        let hashes: Vec<u64> = ids.iter().map(hash_of).collect();
        for (i, a) in hashes.iter().enumerate() {
            for b in &hashes[i + 1..] {
                assert_ne!(a, b, "TypeId hashes collided");
            }
        }
    }

    #[test]
    fn hashing_is_deterministic() {
        // `BuildHasherDefault` is unseeded on purpose; two maps in one process
        // must agree, which the sparse-set indices below rely on.
        assert_eq!(hash_of(&TypeId::of::<u32>()), hash_of(&TypeId::of::<u32>()));
        assert_eq!(hash_of(&(1u32, 2u64)), hash_of(&(1u32, 2u64)));
    }

    /// Byte streams shorter than, equal to, and longer than one word all have
    /// to reach `add` — a `write` that dropped the trailing partial word would
    /// make every short string hash alike.
    ///
    /// Bytes start at 1 deliberately. With a zero seed, an all-zero leading
    /// word is a no-op (`(0.rotate_left(5) ^ 0) * SEED == 0`), so a stream of
    /// zeros genuinely does collide with the empty one. That is FxHash as
    /// rustc ships it and it costs nothing here — `TypeId`s and `Entity`
    /// handles are not zero-prefixed, and a `HashMap` absorbs collisions
    /// anyway. This test checks that the remainder is consumed, not that.
    #[test]
    fn write_covers_partial_words() {
        let mut seen = Vec::new();
        for len in 0..24usize {
            let bytes: Vec<u8> = (0..len as u8).map(|b| b + 1).collect();
            let mut hasher = FxHasher::default();
            hasher.write(&bytes);
            seen.push(hasher.finish());
        }
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before, "different lengths hashed to the same value");
    }

    #[test]
    fn order_of_words_matters() {
        assert_ne!(hash_of(&(1u64, 2u64)), hash_of(&(2u64, 1u64)));
    }
}
