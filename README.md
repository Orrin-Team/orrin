<img width="680" height="340" alt="OrrinBanner" src="https://github.com/user-attachments/assets/a0973d92-878c-4c1e-a892-0be2dc7921c9" />

**A fast, fit to purpose game engine.**

Orrin is a new, open source game engine for 3D games(2D planned for the future) for all kinds of fidelity. Orrin is built to allow you to purpose the game engine to your specific needs, be super fast when editing, coding or publishing.
Join the [discord](https://discord.gg/sY8YBGFmRy) to discuss the development or see progress of the engine!

---

## Why another engine?

* **Fit to purpose** - Select the options and features you need for your game and not be held back by the unnecessary. Many game engines are generalized this game engine is makes it your own.

* **Iteration** - Test and change code on the run. Orrin allows you to edit and test code while running the project unlike many mainstream engines.

* **Real Time Collaboration** - (Feature not released) Orrin allows you to edit projects with teams of people in real time without conflict. 

* **Peak Tooling** - (Feature not released) Want plugins? want code packages? want assets? It's all part of orrin. Many game engines don't let you have all 3, orrin has it baked in.

## Quick start

```bash
# Run the engine on its built-in demo
cargo run -p orrin-core

# With C# scripting (requires the .NET SDK)
dotnet build scripting/Orrin -c Debug
cargo run -p orrin-core --features scripting
```

### Starting a game

The `orrin` CLI creates and runs projects, so you don't need to know any of the
above:

```bash
cargo build -p orrin-cli

./target/debug/orrin new my-game
cd my-game
../target/debug/orrin run
```

`orrin new` scaffolds the manifest, a `.csproj`, and an entry `Behaviour`;
`orrin run` builds the scripts, indexes the assets, and launches the engine on
the project. See [`crates/orrin-cli/README.md`](crates/orrin-cli/README.md).

## Contributing

TBD - This project is to early to be open for contribution. I am afraid people will go of course or misunderstand the project. After phase 5 everyone is open to contribution
