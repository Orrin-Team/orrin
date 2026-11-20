//! A blocking client: one connection, one call at a time, and the one test of
//! whether a remote file is stale, which the client and the editor share.

use std::io::{ErrorKind, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::Path;
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
    /// The remote file names an editor that is certainly not there: its port
    /// refuses the connection, or what listens there closed it without a
    /// word, which an editor does only to a token that is not its own. A file
    /// a crash or a kill left behind.
    Stale(String),
    /// The editor answered, and the answer is a refusal.
    Refused(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gone(why) | Self::Refused(why) | Self::Stale(why) => f.write_str(why),
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
            let why = format!(
                "no editor is listening on {addr} (process {} may have quit): {error}",
                discovery.pid
            );
            match error.kind() {
                ErrorKind::ConnectionRefused => Error::Stale(why),
                _ => Error::Gone(why),
            }
        })?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(CONNECT));
        let _ = stream.set_write_timeout(Some(CONNECT));
        stream
            .write_all(&wire::preamble(&discovery.token))
            .map_err(|error| Error::Gone(format!("the editor at {addr} went: {error}")))?;
        // Whether anything answers at all, told apart from an answer that is
        // late: a closed connection is a listener that is not this file's
        // editor, and a silent one may be an editor stopped in a debugger.
        let stale = || {
            Error::Stale(format!(
                "what listens at {addr} closed the connection without a word: it is not the \
                 editor whose token the remote file holds, so the file is stale"
            ))
        };
        match stream.peek(&mut [0; 1]) {
            Ok(0) => return Err(stale()),
            Ok(_) => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Err(Error::Gone(format!(
                    "the editor at {addr} did not greet within {} s; it may be stopped in a \
                     debugger",
                    CONNECT.as_secs()
                )));
            }
            Err(_) => return Err(stale()),
        }
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

/// Remove the remote file in `editor_dir` if it is stale, and say what was
/// removed; `None` when there is no file, or its editor is there, or whether
/// it is cannot be told. Removed only if it still holds the token it was read
/// with, because an editor starting meanwhile may have just written its own.
pub fn sweep(editor_dir: &Path) -> Option<String> {
    let found = Discovery::read(editor_dir).ok()??;
    match Client::connect(&found) {
        Err(Error::Stale(why)) => {
            found.remove(editor_dir);
            Some(format!(
                "removed the remote file process {} left behind: {why}",
                found.pid
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::net::TcpListener;

    use super::*;
    use crate::token::Token;
    use crate::wire::Hello;

    struct Dir(std::path::PathBuf);

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dir(name: &str) -> Dir {
        let dir = std::env::temp_dir().join(format!("orrin-remote-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Dir(dir)
    }

    fn file(dir: &Dir, port: u16) -> Discovery {
        let discovery = Discovery {
            port,
            token: Token::generate(),
            pid: 1,
        };
        discovery.write(&dir.0).unwrap();
        discovery
    }

    /// A file whose port refuses, or whose listener hangs up unanswered, is
    /// removed; one whose editor greets is left alone. Removing a live
    /// editor's file would cut every client off from it with nothing to say so.
    #[test]
    fn only_a_file_no_editor_answers_for_is_swept() {
        let refusing = dir("refused");
        let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        file(&refusing, port);
        assert!(sweep(&refusing.0).is_some());
        assert!(Discovery::read(&refusing.0).unwrap().is_none());

        let stranger = dir("stranger");
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        file(&stranger, listener.local_addr().unwrap().port());
        let hangs_up = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.read(&mut [0; 44]);
        });
        assert!(sweep(&stranger.0).is_some());
        hangs_up.join().unwrap();
        assert!(Discovery::read(&stranger.0).unwrap().is_none());

        let live = dir("live");
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        file(&live, listener.local_addr().unwrap().port());
        let greets = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.read_exact(&mut [0; 44]).unwrap();
            let hello = Hello {
                version: wire::VERSION,
                project: "game".into(),
                pid: 1,
            };
            stream
                .write_all(&wire::encode(wire::HELLO, &hello).unwrap())
                .unwrap();
            let _ = stream.read(&mut [0; 1]);
        });
        assert!(sweep(&live.0).is_none());
        greets.join().unwrap();
        assert!(Discovery::read(&live.0).unwrap().is_some());
    }
}
