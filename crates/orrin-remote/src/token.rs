//! The secret a client proves it read from the discovery file.

/// Bytes on the wire; twice that in hex in the discovery file.
pub const TOKEN_LEN: usize = 32;

/// 244 random bits from the OS generator: two version-4 UUIDs, which is how
/// `EntityId::new` draws its own, so the remote brings no source of
/// randomness of its own.
///
/// Deliberately not `PartialEq`: the one comparison is [`matches`](Self::matches),
/// which takes as long whatever the bytes are, so a peer cannot learn a prefix
/// from how quickly it was refused. `Debug` never prints it.
#[derive(Clone)]
pub struct Token([u8; TOKEN_LEN]);

impl Token {
    pub fn generate() -> Self {
        let mut bytes = [0; TOKEN_LEN];
        bytes[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        bytes[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
        Self(bytes)
    }

    pub fn from_bytes(bytes: [u8; TOKEN_LEN]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; TOKEN_LEN] {
        &self.0
    }

    /// Lower-case hex, as the discovery file holds it.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// `None` for anything but exactly [`TOKEN_LEN`] bytes of hex.
    pub fn from_hex(text: &str) -> Option<Self> {
        let text = text.as_bytes();
        if text.len() != TOKEN_LEN * 2 {
            return None;
        }
        let mut bytes = [0; TOKEN_LEN];
        for (byte, pair) in bytes.iter_mut().zip(text.as_chunks::<2>().0) {
            let digit = |c: u8| (c as char).to_digit(16);
            *byte = (digit(pair[0])? * 16 + digit(pair[1])?) as u8;
        }
        Some(Self(bytes))
    }

    /// Whether `other` is this token, in time that does not depend on where
    /// they first differ.
    pub fn matches(&self, other: &Token) -> bool {
        let difference = self
            .0
            .iter()
            .zip(&other.0)
            .fold(0u8, |acc, (a, b)| acc | (a ^ b));
        std::hint::black_box(difference) == 0
    }
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(..)")
    }
}
