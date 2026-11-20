//! A blocking client: one connection, one call at a time.

use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::Value;

use crate::discovery::Discovery;
use crate::wire::{self, Hello, ReadError, Request};

/// How long the editor has to accept and greet.
const CONNECT: Duration = Duration::from_secs(3);

/// How long a call may wait for its answer. A request is answered at the
/// editor's next frame, so this is a frame that never came: the editor is
/// stopped in a debugger, or its thread is blocked.
const ANSWER: Duration = Duration::from_secs(30);

/// Why a call has no answer.
#[derive(Debug)]
pub enum Error {
    /// There is no editor at the other end, or it went: connect again, from a
    /// fresh read of the discovery file.
    Gone(String),
    /// The editor answered, and the answer is a refusal.
    Refused(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gone(why) | Self::Refused(why) => f.write_str(why),
        }
    }
}

pub struct Client {
    stream: TcpStream,
    hello: Hello,
    next: u64,
    payload: Vec<u8>,
}

impl Client {
    /// Connect to the editor `discovery` names and wait for its greeting.
    pub fn connect(discovery: &Discovery) -> Result<Self, Error> {
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, discovery.port));
        let mut stream = TcpStream::connect_timeout(&addr, CONNECT).map_err(|error| {
            Error::Gone(format!(
                "no editor is listening on {addr} (process {} may have quit): {error}",
                discovery.pid
            ))
        })?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(CONNECT));
        let _ = stream.set_write_timeout(Some(CONNECT));
        stream
            .write_all(&wire::preamble(&discovery.token))
            .map_err(|error| Error::Gone(format!("the editor at {addr} went: {error}")))?;
        let mut payload = Vec::new();
        let hello = loop {
            match wire::read_frame(&mut stream, &mut payload) {
                Ok(wire::HELLO) => break wire::decode_hello(&payload).map_err(Error::Gone)?,
                Ok(wire::ERROR) => {
                    return Err(Error::Refused(
                        String::from_utf8_lossy(&payload).into_owned(),
                    ));
                }
                Ok(_) => continue,
                Err(_) => {
                    return Err(Error::Gone(format!(
                        "the editor at {addr} closed the connection without a word: the token \
                         in its remote file is not that editor's, so the file is stale"
                    )));
                }
            }
        };
        let _ = stream.set_read_timeout(Some(ANSWER));
        Ok(Self {
            stream,
            hello,
            next: 1,
            payload,
        })
    }

    pub fn hello(&self) -> &Hello {
        &self.hello
    }

    /// Call `method` with `params` and wait for its answer.
    pub fn call(&mut self, method: &str, params: Value) -> Result<Value, Error> {
        self.call_with_blob(method, params)
            .map(|(result, _)| result)
    }

    /// [`call`](Self::call), with the bytes the editor sent ahead of the
    /// answer, for a method that sends any.
    pub fn call_with_blob(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<(Value, Option<Vec<u8>>), Error> {
        let id = self.next;
        self.next += 1;
        let request = Request {
            id,
            method: method.to_owned(),
            params,
        };
        let frame = wire::encode(wire::REQUEST, &request).map_err(Error::Refused)?;
        self.stream
            .write_all(&frame)
            .map_err(|error| Error::Gone(format!("the editor went: {error}")))?;
        let mut blob = None;
        loop {
            let kind = match wire::read_frame(&mut self.stream, &mut self.payload) {
                Ok(kind) => kind,
                Err(ReadError::Closed) => {
                    return Err(Error::Gone(format!(
                        "the editor closed the connection, or did not answer `{method}` within \
                         {} s",
                        ANSWER.as_secs()
                    )));
                }
                Err(ReadError::TooLarge(len)) => {
                    return Err(Error::Gone(format!(
                        "the editor sent a frame of {len} bytes, over the limit of {}",
                        wire::MAX_FRAME
                    )));
                }
            };
            match kind {
                wire::RESPONSE => {
                    let response = wire::decode_response(&self.payload).map_err(Error::Gone)?;
                    // An answer to a call this client gave up waiting on.
                    if response.id != Some(id) && response.id.is_some() {
                        continue;
                    }
                    return match (response.result, response.error) {
                        (Some(result), None) => Ok((result, blob)),
                        (_, Some(error)) => Err(Error::Refused(error)),
                        (None, None) => Err(Error::Gone("an empty response".into())),
                    };
                }
                wire::ERROR => {
                    return Err(Error::Gone(
                        String::from_utf8_lossy(&self.payload).into_owned(),
                    ));
                }
                wire::BLOB => {
                    if let Some((of, bytes)) = wire::decode_blob(&self.payload)
                        && of == id
                    {
                        blob = Some(bytes.to_vec());
                    }
                }
                // A kind added since this client, skipped by its length.
                _ => continue,
            }
        }
    }
}
