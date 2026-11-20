//! The editor's remote: a request/response protocol of the engine's own, on
//! loopback, by which a process on this machine reads — and later edits —
//! the session the developer has open. `orrin mcp` is its first client and
//! translates MCP into it; the editor (`orrin_core::editor::remote`) is its
//! server. Both read this crate, so a session is driven exactly one way, the
//! argument `orrin-project` and `orrin-build` make.
//!
//! **Not MCP.** MCP revises several times a year, and a revision must not
//! rebuild the engine; a client with no use for MCP should not have to speak
//! it. Everything MCP lives in the CLI.
//!
//! # What a connection may do
//!
//! **Loopback, and a token.** The editor binds `127.0.0.1` on a port the OS
//! picks, never a wildcard address, and writes the port and a fresh random
//! [`Token`] to `<project>/.orrin/remote` ([`Discovery`]), readable by its
//! owner alone on Unix and removed when the remote stops. A client proves it
//! read that file. The link to Blender needs no token; this does, because a
//! connection here will be able to change the open scene. A file a killed or
//! crashed editor left behind is removed by whoever next finds it stale
//! ([`client::sweep`]): its port refuses, or what listens there hangs up on
//! its token. A listener that is merely slow to answer is not taken for stale.
//!
//! **Its bytes are untrusted.** A length over [`MAX_FRAME`] is refused from the
//! header, before anything is allocated for it, and ends the connection.
//! Decoding is total, and a request that does not parse is answered with an
//! error that changes nothing.
//!
//! # The wire
//!
//! Version [`VERSION`], on TCP to `127.0.0.1`. Integers are little-endian.
//!
//! **Handshake.** The client speaks first, with 44 bytes that are not a frame:
//! the 8 bytes `ORRNRMTE` ([`MAGIC`]), its protocol version as a `u32`, and the
//! token's 32 bytes. The magic is read alone first, and a peer that opens with
//! anything else — a browser, a port scanner — is dropped having been told
//! nothing; so is one whose token is wrong, and one that has not sent the
//! preamble within five seconds. A peer of another version is sent one `error`
//! frame and dropped, before its token is read. Otherwise the editor answers
//! with `hello`, and from then on the client sends requests and the editor
//! answers them, each in its own frame.
//!
//! **Frames.** A `u32` payload length, a `u8` kind, then the payload. A peer
//! that receives a kind it does not know skips that many bytes and carries on,
//! which is how kinds are added without a new version.
//!
//! | kind | name | from | payload |
//! | --- | --- | --- | --- |
//! | 1 | `hello` | editor | JSON [`Hello`]: version, project name, process id |
//! | 2 | `error` | editor | UTF-8: why the connection is ending |
//! | 3 | `request` | client | JSON [`Request`]: `id` (a `u64`), `method`, `params` |
//! | 4 | `response` | editor | JSON [`Response`]: `id`, and `result` or `error` |
//! | 5 | `blob` | editor | `u64` request id, then bytes belonging to that request's response, sent just before it |
//!
//! A response carries its request's id. Requests that reach the editor's
//! frame are answered in the order they were sent; one refused before it gets
//! there — it does not decode, or the editor is too busy to queue it — is
//! answered at once, and may overtake them.
//!
//! A value from the scene — an entity, a component's default — travels as
//! text in the scene grammar inside the JSON, never as a JSON mapping of
//! `Value`: one grammar, the one in every scene file and every `git diff`.
//!
//! [`Token`]: token::Token
//! [`Discovery`]: discovery::Discovery
//! [`MAX_FRAME`]: wire::MAX_FRAME
//! [`VERSION`]: wire::VERSION
//! [`MAGIC`]: wire::MAGIC
//! [`Hello`]: wire::Hello
//! [`Request`]: wire::Request
//! [`Response`]: wire::Response

pub mod client;
pub mod discovery;
pub mod permissions;
pub mod token;
pub mod wire;

pub use client::{Client, Error};
pub use discovery::Discovery;
pub use token::Token;
