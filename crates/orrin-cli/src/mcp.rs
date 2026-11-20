//! `orrin mcp`: the Model Context Protocol on stdin and stdout, so an AI agent
//! — Claude Code, or any MCP client — can read the editor session the
//! developer has open and build and test the project without it.
//!
//! **MCP lives here and nowhere else.** The editor speaks its own protocol,
//! `orrin_remote`'s, which this translates into; MCP revises several times a
//! year, and a revision changes this file rather than the engine.
//!
//! **The transport is stdio, by hand.** Newline-delimited JSON-RPC 2.0 on
//! serde_json. A tools-only server needs `initialize`, `ping`, `tools/list`
//! and `tools/call`; the official SDK would bring an async runtime into a
//! workspace that has none, for a few hundred lines. stdout carries protocol
//! messages and nothing else — a stray line there is a corrupt stream — so
//! every child's output is captured or sent to a file, and the build's own
//! lines are collected through `build::Log` rather than printed.
//!
//! **Two kinds of tool.** The offline ones are this CLI answering for itself
//! and work with no editor open: `project_info`, `build_scripts` (the `orrin
//! build` path), `run_tests` (the `orrin test` path) and `editor_open` (the
//! `orrin run` path, detached, with `ORRIN_REMOTE=1`). The live ones are the
//! editor's remote methods, forwarded as they are: `editor_state`,
//! `scene_read`, `component_list`, `asset_find` and `console_read`. The
//! editor is found through the project's `.orrin/remote` at the first live
//! call and the connection kept; one that has gone — the editor quit or
//! restarted, the file is stale — is dropped and found again once.
//!
//! **One request at a time.** A build or a test run blocks the loop until it
//! finishes, and a `ping` sent meanwhile is answered after it. A notification
//! is never answered, and nothing in one runs.
//!
//! No tool runs code and no tool writes a file the agent names: what a
//! connection can do is answerable from `tools/list`.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use orrin_project::Project;
use orrin_remote::{Client, Discovery, permissions};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::build::{self, Captured, Log};
use crate::{Fallible, ProjectArgs, engine, run, test};

/// The revisions this server speaks, newest first. Answered with the
/// client's own when it is one of these, else the newest.
const PROTOCOLS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "Orrin is a game engine. The live tools read and change the editor \
     session the developer has open, and need it attached: call editor_open, or ask them to turn \
     on Settings > Agents > Remote. The others build and test the project with no editor. \
     Entities are named by EntityId (a UUID), or #slot for one never saved. Scene text is \
     Orrin's scene grammar, the one in the project's .orrin files. Change the open scene only \
     with scene_edit, each call one undo step the developer can take back; edit scripts, shaders \
     and materials as files and the editor picks the save up. screenshot shows the game as a \
     player sees it, or the scene alone; quote frame_stats' banner with any number taken from \
     it. What agents may do is the developer's to set in Settings > Permissions: a refusal names \
     the switch or folder rule, and project_info and editor_state report what is allowed.";

/// The editor tools, which are the remote's methods by the same names.
const LIVE: [&str; 9] = [
    "editor_state",
    "scene_read",
    "component_list",
    "asset_find",
    "console_read",
    "frame_stats",
    "screenshot",
    "scene_edit",
    "editor_command",
];

/// What turns the editor's remote on for one run, whatever its settings say:
/// `editor::settings::REMOTE_ENV` in the engine, which this does not link.
const REMOTE_ENV: &str = "ORRIN_REMOTE";

/// How long `editor_open` waits by default: enough for cargo to build the
/// engine from a checkout the first time.
const OPEN_TIMEOUT: u64 = 300;

pub fn serve(project: Option<PathBuf>) -> Fallible {
    let mut server = Server {
        project,
        editor: None,
        client: Value::Null,
    };
    let stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lines() {
        let line = line.map_err(|err| format!("could not read stdin: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = server.handle(&line) {
            writeln!(stdout, "{reply}")
                .and_then(|()| stdout.flush())
                .map_err(|err| format!("could not write stdout: {err}"))?;
        }
    }
    Ok(())
}

struct Server {
    /// `--project`: where to look for `orrin.toml`, the current directory
    /// when absent. Located afresh each call, so a manifest edited mid-session
    /// is read as it is.
    project: Option<PathBuf>,
    /// The editor the live tools reach, once one has been found.
    editor: Option<Client>,
    /// Who the MCP client said it is at `initialize`, passed on to each editor
    /// this attaches to, for its Settings ▸ Agents.
    client: Value,
}

/// What a tool says: text, and whether it failed.
struct Answer {
    text: Vec<String>,
    failed: bool,
    /// A PNG, sent to the client as an image ahead of the text.
    png: Option<Vec<u8>>,
}

impl Answer {
    fn ok(text: impl Into<String>) -> Self {
        Self {
            text: vec![text.into()],
            failed: false,
            png: None,
        }
    }

    fn failed(text: impl Into<String>) -> Self {
        Self {
            text: vec![text.into()],
            failed: true,
            png: None,
        }
    }
}

fn reply(id: Value, result: Result<Value, (i64, String)>) -> String {
    let message = match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }),
    };
    message.to_string()
}

impl Server {
    /// The reply to one line, or `None` for a notification or a response.
    fn handle(&mut self, line: &str) -> Option<String> {
        let message: Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(err) => {
                return Some(reply(
                    Value::Null,
                    Err((-32700, format!("Parse error: {err}"))),
                ));
            }
        };
        let Some(object) = message.as_object() else {
            return Some(reply(
                Value::Null,
                Err((
                    -32600,
                    "Invalid Request: a message is one JSON object".to_owned(),
                )),
            ));
        };
        let id = object.get("id").filter(|id| !id.is_null()).cloned();
        let Some(method) = object.get("method").and_then(Value::as_str) else {
            // The answer to a request of this server's, which sends none.
            if object.contains_key("result") || object.contains_key("error") {
                return None;
            }
            return Some(reply(
                id.unwrap_or(Value::Null),
                Err((-32600, "Invalid Request: it names no method".to_owned())),
            ));
        };
        let id = id?;
        let params = object.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => {
                self.client = params.get("clientInfo").cloned().unwrap_or(Value::Null);
                Ok(initialize(&params))
            }
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tools()})),
            "tools/call" => self.call(&params),
            other => Err((-32601, format!("Method not found: {other}"))),
        };
        Some(reply(id, result))
    }

    fn call(&mut self, params: &Value) -> Result<Value, (i64, String)> {
        let invalid = |why: &str| (-32602, format!("Invalid params: {why}"));
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tools/call needs the tool's `name`"))?;
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(arguments @ Value::Object(_)) => arguments.clone(),
            Some(_) => return Err(invalid("`arguments` is an object")),
        };
        let offline = matches!(name, "build_scripts" | "run_tests" | "editor_open");
        let answer = match name {
            _ if offline && let Err(refused) = self.permitted(name, &arguments) => Err(refused),
            "project_info" => self.project_info(&arguments),
            "build_scripts" => self.build_scripts(&arguments),
            "run_tests" => self.run_tests(&arguments),
            "editor_open" => self.editor_open(&arguments),
            live if LIVE.contains(&live) => self.forward(live, arguments),
            other => return Err((-32602, format!("Unknown tool: {other}"))),
        };
        let answer = answer.unwrap_or_else(Answer::failed);
        let image = answer.png.map(|png| {
            use base64::Engine;
            let data = base64::engine::general_purpose::STANDARD.encode(png);
            json!({"type": "image", "data": data, "mimeType": "image/png"})
        });
        let content: Vec<Value> = image
            .into_iter()
            .chain(
                answer
                    .text
                    .into_iter()
                    .map(|text| json!({"type": "text", "text": text})),
            )
            .collect();
        Ok(json!({"content": content, "isError": answer.failed}))
    }

    /// Whether an offline tool may run, by the permissions the editor keeps:
    /// the user's capabilities, read now so a change on the page governs this
    /// call, and the project's folder rules for the scripts folder a build
    /// writes into. The live tools are the editor's to check.
    fn permitted(&self, tool: &str, arguments: &Value) -> Result<(), String> {
        capabilities().check(tool, arguments)?;
        let builds = tool != "editor_open"
            || !arguments
                .get("no_build")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        if !builds {
            return Ok(());
        }
        let project = self.project()?;
        let scripts = project.scripts_dir();
        let scripts = scripts.strip_prefix(project.root()).unwrap_or(&scripts);
        permissions::Folders::load(&project.editor_dir())?
            .permit(&scripts.to_string_lossy(), permissions::Access::ReadWrite)
            .map_err(|why| format!("`{tool}` builds the scripts, but {why}"))
    }

    fn project(&self) -> Result<Project, String> {
        self.args(false).locate()
    }

    fn args(&self, release: bool) -> ProjectArgs {
        ProjectArgs {
            project: self.project.clone(),
            release,
        }
    }

    /// The editor open on `project`, connecting to it when there is no
    /// connection yet.
    fn attach(&mut self, project: &Project) -> Result<&mut Client, String> {
        if self.editor.is_none() {
            let editor_dir = project.editor_dir();
            let found = Discovery::read(&editor_dir)?.ok_or_else(|| no_editor(project, None))?;
            let client = Client::connect(&found).map_err(|err| match err {
                // Left by an editor that was killed or crashed: whoever notices
                // first clears it, if it still holds the token read.
                orrin_remote::Error::Stale(why) => {
                    found.remove(&editor_dir);
                    no_editor(project, Some(&format!("{why}; the file was removed")))
                }
                err => no_editor(project, Some(&err.to_string())),
            })?;
            // An editor older than `client_info` refuses it, and an editor
            // that has gone is found out by the call that follows; neither
            // is this call's to report.
            let mut client = client;
            if let Some(name) = self.client.get("name").and_then(Value::as_str) {
                let version = self.client.get("version").and_then(Value::as_str);
                let _ = client.call("client_info", json!({"name": name, "version": version}));
            }
            self.editor = Some(client);
        }
        Ok(self.editor.as_mut().expect("just connected"))
    }

    /// `method` on the editor's remote, reconnecting once to an editor that
    /// went away since the last call.
    fn forward(&mut self, method: &str, arguments: Value) -> Result<Answer, String> {
        let project = self.project()?;
        let mut retried = false;
        loop {
            let client = self.attach(&project)?;
            match client.call_with_blob(method, arguments.clone()) {
                Ok((result, png)) => {
                    return Ok(Answer {
                        text: rendered(result),
                        failed: false,
                        png,
                    });
                }
                Err(orrin_remote::Error::Refused(why)) => return Ok(Answer::failed(why)),
                Err(orrin_remote::Error::Gone(why) | orrin_remote::Error::Stale(why)) => {
                    self.editor = None;
                    if retried {
                        return Err(no_editor(&project, Some(&why)));
                    }
                    retried = true;
                }
            }
        }
    }

    fn project_info(&mut self, arguments: &Value) -> Result<Answer, String> {
        arguments_of::<Nothing>(arguments)?;
        let project = self.project()?;
        let root = project.root();
        let relative = |path: PathBuf| {
            path.strip_prefix(root)
                .map_or_else(|_| path.clone(), Path::to_path_buf)
                .to_string_lossy()
                .replace('\\', "/")
        };
        let assets = project.assets_dir();
        let mut scenes = Vec::new();
        scene_files(&assets, &mut scenes, 500);
        scenes.sort();
        let scenes: Vec<String> = scenes.into_iter().map(relative).collect();
        let editor = match self.attach(&project).map(|client| client.hello().clone()) {
            Ok(hello) => json!({"attached": true, "pid": hello.pid, "project": hello.project}),
            Err(why) => json!({"attached": false, "why": why}),
        };
        let info = json!({
            "name": project.name(),
            "version": project.version(),
            "root": root.display().to_string(),
            "scripts": relative(project.scripts_dir()),
            "entry": project.entry_type(),
            "assets": relative(assets),
            "start_scene": project.start_scene_relative().map(|scene| scene.display().to_string()),
            "persistent_scene": project
                .persistent_scene_relative()
                .map(|scene| scene.display().to_string()),
            "scenes": scenes,
            "orrin": env!("CARGO_PKG_VERSION"),
            "permissions": permissions::summary(
                &capabilities(),
                &permissions::Folders::load(&project.editor_dir()),
            ),
            "editor": editor,
        });
        Ok(Answer::ok(info.to_string()))
    }

    fn build_scripts(&mut self, arguments: &Value) -> Result<Answer, String> {
        let build: Build = arguments_of(arguments)?;
        let mut log = Log::Captured(Captured::default());
        let built = build::build(&self.args(build.release), &mut log);
        let Log::Captured(captured) = log else {
            unreachable!("captured above")
        };
        Ok(built_answer(built.map(|_| ()), captured, "Build succeeded"))
    }

    fn run_tests(&mut self, arguments: &Value) -> Result<Answer, String> {
        let tests: Tests = arguments_of(arguments)?;
        let mut log = Log::Captured(Captured::default());
        let prepared = run::prepare(&self.args(tests.release), false, &mut log);
        let Log::Captured(captured) = log else {
            unreachable!("captured above")
        };
        let (project, bindings) = match prepared {
            Ok(prepared) => prepared,
            Err(why) => return Ok(built_answer(Err(why), captured, "")),
        };
        let engine = engine::find(project.root(), true)?;
        let output = test::command(
            &project,
            bindings.as_deref(),
            &engine,
            tests.filter.as_deref(),
            test::Format::Json,
        )
        .stdin(Stdio::null())
        .output()
        .map_err(|err| format!("could not launch the engine ({}): {err}", engine.origin))?;
        let report = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let counts = serde_json::from_str::<Value>(&report).ok().map(|report| {
            let count = |key: &str| report.get(key).and_then(Value::as_u64).unwrap_or(0);
            (count("passed"), count("failed"))
        });
        let filter = tests.filter.as_deref().unwrap_or_default();
        let summary = match (output.status.code(), counts) {
            (Some(0 | 1), Some((0, 0))) if !filter.is_empty() => {
                format!("No test's name contains `{filter}`, so none ran")
            }
            (Some(0 | 1), Some((0, 0))) => "The project has no tests, so none ran".to_owned(),
            (Some(0), Some((passed, _))) => format!("All {passed} test(s) passed"),
            (Some(1), Some((passed, failed))) => {
                format!("{failed} test(s) failed and {passed} passed")
            }
            (Some(0), None) => "Every test passed".to_owned(),
            (Some(1), None) => "A test failed, or none matched; the report did not read".to_owned(),
            (Some(code), _) => {
                format!("The tests could not be run (the engine exited with {code})")
            }
            (None, _) => "The engine was stopped by a signal while testing".to_owned(),
        };
        let mut text = vec![summary];
        if !report.is_empty() {
            text.push(report);
        }
        let ran = matches!(output.status.code(), Some(0 | 1));
        if !ran {
            text.push(tail(&String::from_utf8_lossy(&output.stderr), 40));
        }
        let none = counts == Some((0, 0));
        Ok(Answer {
            text,
            failed: !ran || none,
            png: None,
        })
    }

    fn editor_open(&mut self, arguments: &Value) -> Result<Answer, String> {
        let open: Open = arguments_of(arguments)?;
        let project = self.project()?;
        if let Ok(client) = self.attach(&project) {
            let hello = client.hello();
            return Ok(Answer::ok(format!(
                "An editor is already open on `{}` (process {}) and attached",
                hello.project, hello.pid
            )));
        }
        let mut log = Log::Captured(Captured::default());
        let prepared = run::prepare(&self.args(false), open.no_build, &mut log);
        let Log::Captured(captured) = log else {
            unreachable!("captured above")
        };
        let (project, bindings) = match prepared {
            Ok(prepared) => prepared,
            Err(why) => {
                let mut answer = built_answer(Err(why), captured, "");
                answer.text.push(
                    "The editor was not opened. Fix the build, or pass no_build to open what was \
                     last compiled"
                        .to_owned(),
                );
                return Ok(answer);
            }
        };
        let engine = engine::find(project.root(), true)?;
        let editor_dir = project.editor_dir();
        std::fs::create_dir_all(&editor_dir)
            .map_err(|err| format!("could not create {}: {err}", editor_dir.display()))?;
        let log_path = editor_dir.join("editor.log");
        let log_file = std::fs::File::create(&log_path)
            .map_err(|err| format!("could not create {}: {err}", log_path.display()))?;
        let err_file = log_file
            .try_clone()
            .map_err(|err| format!("could not share {}: {err}", log_path.display()))?;
        let mut command = run::command(&project, bindings.as_deref(), &engine, &[]);
        command
            .env(REMOTE_ENV, "1")
            .stdin(Stdio::null())
            .stdout(log_file)
            .stderr(err_file);
        // A process group of its own, so the client stopping this server
        // with a signal does not take the editor with it.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = command
            .spawn()
            .map_err(|err| format!("could not launch the engine ({}): {err}", engine.origin))?;

        let timeout = Duration::from_secs(open.timeout_secs.unwrap_or(OPEN_TIMEOUT));
        let started = Instant::now();
        let outcome = loop {
            if let Ok(Some(status)) = child.try_wait() {
                let log = std::fs::read_to_string(&log_path).unwrap_or_default();
                break Answer {
                    text: vec![
                        format!("The editor exited before it was reachable ({status})"),
                        tail(&log, 40),
                    ],
                    failed: true,
                    png: None,
                };
            }
            if let Ok(client) = self.attach(&project) {
                let hello = client.hello();
                break Answer::ok(format!(
                    "Opened the editor on `{}` (process {}) with {}, and attached; it logs to {}",
                    hello.project,
                    hello.pid,
                    engine.origin,
                    log_path.display()
                ));
            }
            if started.elapsed() > timeout {
                break Answer::failed(format!(
                    "The editor did not start listening within {} s. It may still be compiling: \
                     from a checkout, cargo builds the engine the first time. Its output is in \
                     {}; call editor_open again to wait longer",
                    timeout.as_secs(),
                    log_path.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        };
        // Reaped once it exits, whenever that is; this server never waits on it.
        std::thread::spawn(move || child.wait());
        Ok(outcome)
    }
}

/// The user's capabilities for agents, as the editor last saved them.
fn capabilities() -> permissions::Capabilities {
    orrin_project::user::dir()
        .map(|dir| permissions::Capabilities::of_user(&dir))
        .unwrap_or_default()
}

fn no_editor(project: &Project, why: Option<&str>) -> String {
    let why = why.map_or_else(String::new, |why| format!(" ({why})"));
    format!(
        "No editor is attached to `{}`{why}. Call editor_open to open one with its remote on, \
         or turn on Settings > Agents > Remote in the editor already open",
        project.name()
    )
}

/// A remote result as tool text: the JSON, compact, with every top-level
/// string that runs over several lines — scene text above all — lifted out
/// after it as text of its own, so it reads as the grammar rather than as an
/// escaped string.
fn rendered(mut result: Value) -> Vec<String> {
    let mut lifted = Vec::new();
    if let Some(object) = result.as_object_mut() {
        let keys: Vec<String> = object
            .iter()
            .filter(|(_, value)| value.as_str().is_some_and(|text| text.contains('\n')))
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            if let Some(Value::String(text)) = object.remove(&key) {
                lifted.push(text);
            }
        }
    }
    let mut text = vec![result.to_string()];
    text.extend(lifted);
    text
}

/// A build's outcome as tool text: the diagnostics by file and line, and the
/// build's own lines when it failed with none to show.
fn built_answer(built: Result<(), String>, captured: Captured, success: &str) -> Answer {
    let diagnostics: Vec<Value> = captured
        .diagnostics
        .iter()
        .map(|line| diagnostic(line))
        .collect();
    match built {
        Ok(()) => {
            let warnings = diagnostics.len();
            let mut text = vec![format!("{success}, with {warnings} warning(s)")];
            if warnings > 0 {
                text.push(Value::Array(diagnostics).to_string());
            }
            Answer {
                text,
                failed: false,
                png: None,
            }
        }
        Err(why) => {
            let mut text = vec![why];
            if diagnostics.is_empty() {
                text.push(captured.lines.join("\n"));
            } else {
                text.push(Value::Array(diagnostics).to_string());
            }
            Answer {
                text,
                failed: true,
                png: None,
            }
        }
    }
}

/// One MSBuild diagnostic, `File.cs(12,9): error CS0103: message`, by its
/// parts, or the line as it is when it is not in that form.
fn diagnostic(line: &str) -> Value {
    let parsed = (|| {
        let (location, rest) = line.split_once("): ")?;
        let (file, position) = location.rsplit_once('(')?;
        let (row, column) = position.split_once(',')?;
        let (severity, rest) = rest.split_once(' ')?;
        let (code, message) = rest.split_once(": ")?;
        Some(json!({
            "file": file,
            "line": row.trim().parse::<u32>().ok()?,
            "column": column.trim().parse::<u32>().ok()?,
            "severity": severity,
            "code": code,
            "message": message,
        }))
    })();
    parsed.unwrap_or_else(|| json!({"text": line}))
}

/// The last `count` non-blank lines of `text`.
fn tail(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

/// Every `.orrin` scene under `dir`, at most `limit`, skipping dot
/// directories and symlinks as the asset index does.
fn scene_files(dir: &Path, out: &mut Vec<PathBuf>, limit: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= limit {
            return;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_symlink() || entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if kind.is_dir() {
            scene_files(&path, out, limit);
        } else if path.extension().is_some_and(|ext| ext == "orrin") {
            out.push(path);
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Nothing {}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Build {
    release: bool,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Tests {
    filter: Option<String>,
    release: bool,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Open {
    no_build: bool,
    timeout_secs: Option<u64>,
}

fn arguments_of<T: DeserializeOwned>(arguments: &Value) -> Result<T, String> {
    T::deserialize(arguments).map_err(|err| format!("the arguments do not read: {err}"))
}

fn initialize(params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str);
    let version = asked
        .filter(|asked| PROTOCOLS.contains(asked))
        .unwrap_or(PROTOCOLS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "orrin", "title": "Orrin", "version": env!("CARGO_PKG_VERSION")},
        "instructions": INSTRUCTIONS,
    })
}

/// What a tool may do, as its annotations tell a client.
enum Hint {
    Reads,
    /// Writes only what a build or a launch writes: outputs, never sources.
    Builds,
    /// Changes the open scene or the editor's state.
    Changes,
}

fn tool(name: &str, description: &str, properties: Value, hint: Hint) -> Value {
    let annotations = match hint {
        Hint::Reads => json!({"readOnlyHint": true, "openWorldHint": false}),
        Hint::Builds => {
            json!({"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false})
        }
        Hint::Changes => {
            json!({"readOnlyHint": false, "destructiveHint": true, "openWorldHint": false})
        }
    };
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "additionalProperties": false,
        },
        "annotations": annotations,
    })
}

/// `tool` with `required` arguments.
fn requiring(mut tool: Value, required: &[&str]) -> Value {
    tool["inputSchema"]["required"] = json!(required);
    tool
}

/// One op of `scene_edit`'s, by its tag and its own arguments.
fn op(name: &str, properties: Value, required: &[&str]) -> Value {
    let mut properties = properties;
    properties["op"] = json!({"const": name});
    let mut required = required.to_vec();
    required.insert(0, "op");
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn tools() -> Vec<Value> {
    let limit = |what: &str| json!({"type": "integer", "minimum": 1, "description": what});
    let offset = json!({"type": "integer", "minimum": 0, "description": "How many to skip; an answer's `next` is the offset of the page after it"});
    vec![
        tool(
            "project_info",
            "The project's manifest (name, version, entry behaviour, scripts and assets \
             folders), its scenes, and whether an editor is attached. Works with no editor open.",
            json!({}),
            Hint::Reads,
        ),
        tool(
            "build_scripts",
            "Compile the project's C# scripts as `orrin build` does, and index its assets. \
             Returns the compiler's errors and warnings by file, line and column.",
            json!({"release": {"type": "boolean", "description": "Build in Release rather than Debug"}}),
            Hint::Builds,
        ),
        tool(
            "run_tests",
            "Build the scripts, then run the project's [Test] methods in the engine with no \
             window, as `orrin test` does. Returns the engine's JSON report.",
            json!({
                "filter": {"type": "string", "description": "Only tests whose full name (Namespace.Type.Method) contains this"},
                "release": {"type": "boolean", "description": "Build the scripts in Release"},
            }),
            Hint::Builds,
        ),
        tool(
            "editor_open",
            "Build the scripts, then open the editor on the project with its remote on, as \
             `orrin run` does, and wait until the live tools can reach it. Does nothing if one \
             is already attached.",
            json!({
                "no_build": {"type": "boolean", "description": "Open what was last compiled, without building"},
                "timeout_secs": {"type": "integer", "minimum": 1, "description": "How long to wait for the editor to listen (default 300: a checkout compiles the engine first)"},
            }),
            Hint::Builds,
        ),
        tool(
            "editor_state",
            "The open editor: project, open scene and whether it has unsaved changes, play \
             state, which world reads see (edit, or the playing fork), the selection by entity \
             id, and the camera.",
            json!({}),
            Hint::Reads,
        ),
        tool(
            "scene_read",
            "The world the editor shows. By default an outline: each entity's id, name, parent \
             and component ids. With blocks, whole entities in Orrin's scene grammar, exactly \
             as a save writes them. Filters combine; results are paged.",
            json!({
                "ids": {"type": "array", "items": {"type": "string"}, "description": "Only these entities, by EntityId or #slot"},
                "root": {"type": "string", "description": "Only this entity and everything under it"},
                "name": {"type": "string", "description": "Only entities whose name contains this, ignoring case"},
                "blocks": {"type": "boolean", "description": "Whole entity blocks in the scene grammar instead of an outline"},
                "limit": limit("At most this many entities (default 100 outlined, 20 as blocks)"),
                "offset": offset,
            }),
            Hint::Reads,
        ),
        tool(
            "component_list",
            "Every component the editor can attach, the game's own scripts included: its id, \
             display name, and its default value in the scene grammar.",
            json!({
                "query": {"type": "string", "description": "Only components whose id or name contains this, ignoring case"},
                "defaults": {"type": "boolean", "description": "Include each default value (default true)"},
                "limit": limit("At most this many (default 200)"),
                "offset": offset,
            }),
            Hint::Reads,
        ),
        tool(
            "asset_find",
            "The project's asset index: each asset's id, path, kind, load state and import \
             error, filtered by kind and path.",
            json!({
                "kind": {"type": "string", "enum": ["texture", "mesh", "prefab", "material", "scene", "font", "audio", "shader"]},
                "path": {"type": "string", "description": "Only assets whose path contains this, ignoring case"},
                "limit": limit("At most this many (default 50)"),
                "offset": offset,
            }),
            Hint::Reads,
        ),
        tool(
            "console_read",
            "The editor console's entries from a cursor on, with the script or module that \
             logged each and its line. Pass the returned `next` as `cursor` to read only what \
             is new; compiler errors from a script save land here.",
            json!({
                "cursor": {"type": "integer", "minimum": 0, "description": "The number of the first entry wanted (default 0, the oldest)"},
                "level": {"type": "string", "enum": ["trace", "debug", "info", "warn", "error"], "description": "The quietest level returned (default info)"},
                "limit": limit("At most this many entries (default 100)"),
            }),
            Hint::Reads,
        ),
        tool(
            "frame_stats",
            "What the editor's Profiler shows: frame time, GPU time, memory, and the CPU phases \
             and GPU passes slowest first, with the run banner that says how the run was \
             started. A number means little without the banner beside it.",
            json!({"limit": limit("Rows a lane (default 20)")}),
            Hint::Reads,
        ),
        tool(
            "screenshot",
            "What the editor's viewport renders, as a PNG no larger than max_size on its \
             longest side: by default the game as a player sees it, its UI included; or the \
             scene alone. Never the editor's own panels or marks.",
            json!({
                "view": {"type": "string", "enum": ["game", "scene"], "description": "game: the scene and the game's UI (default); scene: the scene alone"},
                "max_size": {"type": "integer", "minimum": 16, "maximum": 4096, "description": "Longest side in pixels (default 1024)"},
            }),
            Hint::Reads,
        ),
        scene_edit(),
        requiring(
            tool(
                "editor_command",
                "Drive the editor as its buttons do, answered once the effect is seen: play, \
                 pause, step, stop, save (path for a never-saved scene), open_scene (path; save \
                 or discard over unsaved changes), undo, redo, reload_scripts, select (ids), \
                 frame (the camera on the selection).",
                json!({
                    "command": {"type": "string", "enum": ["play", "pause", "step", "stop", "save", "open_scene", "undo", "redo", "reload_scripts", "select", "frame"]},
                    "path": {"type": "string", "description": "save, open_scene: a scene file relative to the project, ending .orrin"},
                    "save": {"type": "boolean", "description": "open_scene: save unsaved changes first"},
                    "discard": {"type": "boolean", "description": "open_scene: lose unsaved changes"},
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "select: entity ids or #slots; empty clears"},
                }),
                Hint::Changes,
            ),
            &["command"],
        ),
    ]
}

fn scene_edit() -> Value {
    let id = json!({"type": "string", "description": "An EntityId, or #slot"});
    let text = json!({"type": "string", "description": "Components in the scene grammar, each as component_list prints it"});
    let component = json!({"type": "string", "description": "A component id"});
    let ops = json!({
        "type": "array",
        "items": {"oneOf": [
            op("set", json!({"entity": id, "text": text}), &["entity", "text"]),
            op("add_component", json!({"entity": id, "component": component, "text": text}), &["entity", "component"]),
            op("remove_component", json!({"entity": id, "component": component}), &["entity", "component"]),
            op("spawn", json!({
                "id": {"type": "string", "description": "A new UUID of your choosing, for later ops to name"},
                "name": {"type": "string"},
                "parent": id,
                "text": text,
                "prefab": {"type": "string", "description": "A prefab or mesh asset, by id or path, placed instead"},
            }), &[]),
            op("despawn", json!({"entities": {"type": "array", "items": id}}), &["entities"]),
            op("reparent", json!({
                "entity": id,
                "parent": {"type": ["string", "null"], "description": "null makes it a root"},
                "keep_world": {"type": "boolean", "description": "Stay where it is in the world (default true)"},
            }), &["entity"]),
        ]},
    });
    requiring(
        tool(
            "scene_edit",
            "Change the open scene: ops applied in order as one undo step named after `label`, \
             all or none. set replaces only the fields its text names; added and spawned \
             components fill the rest from their defaults, and a spawn holds only what it is \
             given (add orrin.transform to place it). Edit mode only.",
            json!({
                "label": {"type": "string", "description": "What the step is called in the Edit menu"},
                "ops": ops,
            }),
            Hint::Changes,
        ),
        &["label", "ops"],
    )
}
