# movement-demo

The engine's first-person movement and target range as a project of its own,
so it can be run and exported like any game. The gameplay is
`scripting/DemoGame`'s — `FpsPlayer`, `BallLauncher`, `Target` and
`MovementDemo` — compiled in by link, and `crates/core/tests/movement.rs` is
what holds its feel to numbers.

```sh
cargo run -p orrin-cli -- run --project examples/movement-demo
cargo run -p orrin-cli -- export --project examples/movement-demo
```

Click to lock the mouse. WASD moves, Space jumps (hold to hop on landing),
Ctrl or C crouches into a slide, click fires, Esc lets go of the mouse.

`assets/scenes/main.orrin` is a scene as the editor's New Scene makes one: the
built-in sun and sky. The course itself is built by `MovementDemo` when it
starts, a hundred metres down -Z.
