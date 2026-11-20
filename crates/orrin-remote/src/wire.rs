//! The remote's bytes: the preamble, frames, and the JSON a request, a
//! response and the greeting carry. The layout is written down once, in the
//! [crate] header; this is its one reader and writer.
//!
//! Every decoder here is total over its input: no byte sequence panics one,
//! and none allocates more than the peer actually sent.

use std::io::Read;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::token::{TOKEN_LEN, Token};

/// What a client sends before anything else, then its version and the token.
pub const MAGIC: [u8; 8] = *b"ORRNRMTE";

/// The protocol this engine speaks. Bumped for any change an older peer would
/// misread; a new frame kind or a new method is not one.
pub const VERSION: u32 = 1;

/// The largest frame payload either side accepts, in bytes. Step three's
/// screenshot is the largest thing the remote will carry: a 3840x2160 RGBA
/// PNG stored rather than compressed is 33 MB, so twice that refuses whatever
/// a confused peer announces past it before a byte is allocated for it.
pub const MAX_FRAME: u32 = 64 << 20;

/// A frame's length and kind, ahead of its payload.
pub const HEADER: usize = 5;

/// Engine to client, once, after a good preamble: a [`Hello`].
pub const HELLO: u8 = 1;
/// Engine to client: UTF-8 text saying why the connection is ending.
pub const ERROR: u8 = 2;
/// Client to engine: a [`Request`].
pub const REQUEST: u8 = 3;
/// Engine to client: a [`Response`].
pub const RESPONSE: u8 = 4;
/// Either way: a `u64` request id, then raw bytes belonging to that request's
/// response. Reserved; nothing sends one yet.
pub const BLOB: u8 = 5;

/// What a peer opened with.
#[derive(Debug)]
pub enum Opening {
    /// The magic, this version, and a token, which is not yet checked.
    Ours(Token),
    /// The magic, and another version. Its token is not read: another version
    /// may not send one where this one expects it.
    Version(u32),
    /// Anything else, or nothing: a browser, a port scanner.
    Stranger,
}

/// Read a client's preamble. The magic is read alone first, so a peer that is
/// not one is turned away before it is asked for anything more.
pub fn opening(reader: &mut impl Read) -> Opening {
    let mut magic = [0; MAGIC.len()];
    if reader.read_exact(&mut magic).is_err() || magic != MAGIC {
        return Opening::Stranger;
    }
    let mut version = [0; 4];
    if reader.read_exact(&mut version).is_err() {
        return Opening::Stranger;
    }
    match u32::from_le_bytes(version) {
        VERSION => {
            let mut token = [0; TOKEN_LEN];
            match reader.read_exact(&mut token) {
                Ok(()) => Opening::Ours(Token::from_bytes(token)),
                Err(_) => Opening::Stranger,
            }
        }
        other => Opening::Version(other),
    }
}

/// What a client sends to open a connection with `token`.
pub fn preamble(token: &Token) -> Vec<u8> {
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + TOKEN_LEN);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(token.as_bytes());
    out
}

/// Why the next frame could not be read.
#[derive(Debug, PartialEq)]
pub enum ReadError {
    /// The peer went, or the connection was shut down.
    Closed,
    /// It announced a payload over [`MAX_FRAME`]. Nothing was read past the
    /// header, so the stream is out of step and the connection must end.
    TooLarge(u32),
}

/// The next frame's kind, its payload in `payload`. The payload grows as its
/// bytes arrive rather than being sized from the header, so a peer that
/// announces a large frame and sends nothing holds nothing.
pub fn read_frame(reader: &mut impl Read, payload: &mut Vec<u8>) -> Result<u8, ReadError> {
    let mut header = [0; HEADER];
    reader
        .read_exact(&mut header)
        .map_err(|_| ReadError::Closed)?;
    let [a, b, c, d, kind] = header;
    let len = u32::from_le_bytes([a, b, c, d]);
    if len > MAX_FRAME {
        return Err(ReadError::TooLarge(len));
    }
    payload.clear();
    match reader.take(u64::from(len)).read_to_end(payload) {
        Ok(read) if read == len as usize => Ok(kind),
        _ => Err(ReadError::Closed),
    }
}

/// A frame of `kind` carrying `payload`, or why it cannot be sent.
pub fn frame(kind: u8, payload: &[u8]) -> Result<Vec<u8>, String> {
    let len = u32::try_from(payload.len())
        .ok()
        .filter(|&len| len <= MAX_FRAME)
        .ok_or_else(|| {
            format!(
                "{} bytes is over the remote's limit of {MAX_FRAME} a frame",
                payload.len()
            )
        })?;
    let mut out = Vec::with_capacity(HEADER + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    Ok(out)
}

/// The greeting, which says the token was accepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub version: u32,
    /// The project the editor has open, by its manifest name.
    pub project: String,
    pub pid: u32,
}

/// One call. `params` is an object, or absent for a method that takes none.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub method: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

/// The answer to the request with `id`: exactly one of `result` and `error`.
/// `id` is `None` only when the request was too malformed to carry one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Response {
    pub fn answer(id: Option<u64>, answer: Result<Value, String>) -> Self {
        match answer {
            Ok(result) => Self {
                id,
                result: Some(result),
                error: None,
            },
            Err(error) => Self {
                id,
                result: None,
                error: Some(error),
            },
        }
    }
}

/// A request that could not be read: the id, when one could be found, and why.
#[derive(Debug, PartialEq)]
pub struct Unreadable {
    pub id: Option<u64>,
    pub reason: String,
}

/// A request frame's payload as the request it describes.
pub fn decode_request(payload: &[u8]) -> Result<Request, Unreadable> {
    serde_json::from_slice::<Request>(payload).map_err(|error| {
        let id = serde_json::from_slice::<Value>(payload)
            .ok()
            .and_then(|value| value.get("id")?.as_u64());
        Unreadable {
            id,
            reason: format!("the request is not one the remote can read: {error}"),
        }
    })
}

pub fn decode_response(payload: &[u8]) -> Result<Response, String> {
    let response: Response = serde_json::from_slice(payload)
        .map_err(|error| format!("the editor's response could not be read: {error}"))?;
    match (&response.result, &response.error) {
        (Some(_), None) | (None, Some(_)) => Ok(response),
        _ => Err("the editor's response carries neither a result nor an error, or both".into()),
    }
}

pub fn decode_hello(payload: &[u8]) -> Result<Hello, String> {
    serde_json::from_slice(payload)
        .map_err(|error| format!("the editor's greeting could not be read: {error}"))
}

/// `value` as a frame of `kind`. Serialising these types cannot fail; a frame
/// over the limit can, and says so.
pub fn encode(kind: u8, value: &impl Serialize) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    frame(kind, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> Token {
        Token::from_bytes([7; TOKEN_LEN])
    }

    #[test]
    fn a_peer_is_ours_only_by_the_magic_then_the_version_then_a_whole_token() {
        let good = preamble(&token());
        assert!(
            matches!(opening(&mut good.as_slice()), Opening::Ours(t) if t.matches(&token())),
            "the token comes through as sent"
        );

        let mut other = good.clone();
        other[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        assert!(matches!(
            opening(&mut other.as_slice()),
            Opening::Version(v) if v == VERSION + 1
        ));

        let http = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert!(matches!(opening(&mut http.as_slice()), Opening::Stranger));
        for cut in 0..good.len() {
            assert!(
                matches!(opening(&mut &good[..cut]), Opening::Stranger),
                "a preamble cut at {cut} is no one's"
            );
        }
    }

    /// The cap is checked on the header alone: a reader holding nothing past
    /// it is never asked for more, and nothing is allocated for the payload.
    #[test]
    fn a_length_over_the_cap_is_refused_before_anything_is_allocated() {
        let mut header = (MAX_FRAME + 1).to_le_bytes().to_vec();
        header.push(REQUEST);
        let mut payload = Vec::new();
        assert_eq!(
            read_frame(&mut header.as_slice(), &mut payload),
            Err(ReadError::TooLarge(MAX_FRAME + 1))
        );
        assert_eq!(payload.capacity(), 0);

        let mut cut = frame(REQUEST, b"{}").unwrap();
        cut.pop();
        assert_eq!(
            read_frame(&mut cut.as_slice(), &mut payload),
            Err(ReadError::Closed)
        );
        let whole = frame(42, &[1, 2, 3]).unwrap();
        assert_eq!(read_frame(&mut whole.as_slice(), &mut payload), Ok(42));
        assert_eq!(payload, [1, 2, 3]);

        assert!(frame(RESPONSE, &vec![0; MAX_FRAME as usize + 1]).is_err());
    }

    /// Each class of bad request, refused with a reason and the id when there
    /// is one to answer, never a panic.
    #[test]
    fn every_malformed_request_is_refused_with_its_id_when_it_has_one() {
        let good = br#"{"id":3,"method":"scene_read","params":{"limit":2}}"#;
        assert_eq!(
            decode_request(good).unwrap(),
            Request {
                id: 3,
                method: "scene_read".into(),
                params: serde_json::json!({"limit": 2}),
            }
        );
        assert_eq!(
            decode_request(br#"{"id":4,"method":"editor_state"}"#)
                .unwrap()
                .params,
            Value::Null
        );

        let ids: Vec<Option<u64>> = [
            &b""[..],
            b"not json",
            b"\xff\xfe",
            br#"[1,2]"#,
            br#"{"method":"editor_state"}"#,
            br#"{"id":-1,"method":"editor_state"}"#,
            br#"{"id":5}"#,
            br#"{"id":6,"method":7}"#,
            br#"{"id":18446744073709551616,"method":"x"}"#,
        ]
        .into_iter()
        .map(|bad| decode_request(bad).expect_err("refused").id)
        .collect();
        assert_eq!(
            ids,
            [None, None, None, None, None, None, Some(5), Some(6), None]
        );

        let deep = "[".repeat(100_000);
        assert!(
            decode_request(deep.as_bytes()).is_err(),
            "no stack overflow"
        );
    }

    /// Every prefix of a good frame, and the frame with each byte flipped,
    /// reads or is refused; none panics.
    #[test]
    fn no_damage_to_a_frame_panics_a_decoder() {
        let request = Request {
            id: 9,
            method: "console_read".into(),
            params: serde_json::json!({"after": 12, "level": "warn"}),
        };
        let good = encode(REQUEST, &request).unwrap();
        let mut payload = Vec::new();
        for len in 0..good.len() {
            assert!(read_frame(&mut &good[..len], &mut payload).is_err());
            let body = &good[HEADER.min(len)..len];
            let _ = decode_request(body);
            let _ = decode_response(body);
            let _ = decode_hello(body);
        }
        for at in 0..good.len() {
            for flip in [0x01, 0x80, 0xff] {
                let mut bytes = good.clone();
                bytes[at] ^= flip;
                let _ = read_frame(&mut bytes.as_slice(), &mut payload);
                let body = &bytes[HEADER..];
                let _ = decode_request(body);
                let _ = decode_response(body);
                let _ = decode_hello(body);
            }
        }
    }

    #[test]
    fn a_response_is_a_result_or_an_error_and_never_both() {
        let ok = encode(RESPONSE, &Response::answer(Some(1), Ok(Value::Bool(true)))).unwrap();
        assert_eq!(
            decode_response(&ok[HEADER..]).unwrap().result,
            Some(Value::Bool(true))
        );
        let refused = encode(RESPONSE, &Response::answer(None, Err("no".into()))).unwrap();
        assert_eq!(
            decode_response(&refused[HEADER..])
                .unwrap()
                .error
                .as_deref(),
            Some("no")
        );
        assert!(decode_response(br#"{"id":1}"#).is_err());
        assert!(decode_response(br#"{"id":1,"result":1,"error":"x"}"#).is_err());
    }
}
