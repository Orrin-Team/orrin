# Orrin

**A game engine you actually own, built in Rust, scripted in C#, designed around everything other engines neglect.**

Orrin is an early-stage, open game engine for 3D games of every style, from low-poly to (eventually) path-traced realism. The whole stack core, renderer, and editor is yours to read, change, and ship. No black boxes, no royalties, no licensing traps, and no cloud service you can't run yourself.

> **Status:** early and moving fast. APIs will change. The honest version of every claim below is tagged: things that work **today**, things the current milestones deliver **this year**, and things the architecture is being laid for **on the roadmap**. An engine that tells you what's aspirational is an engine you can trust about what isn't.

---

## Why another engine?

The mainstream engines are enormously capable and they all neglect the same things: the daily loop of *iterating, collaborating, and owning your work*. Scenes that can't merge. References that break when files move. Renderers you can't see inside. Script changes that cost a restart. Package managers that don't resolve anything. Terms that change under you.

Orrin is architected around six commitments instead:

**Iteration is sacred.** Change code, see the result, engine still running. Your game is an external C# assembly that hot-reloads on save script state survives because state lives in components, by construction. Shaders and assets reload the same way. Editor startup and reload latency are guarded by CI benchmarks, because speed decays one dependency at a time. *(Hot reload: this year Phase 2.)*

**Your scene is a document, not a minefield.** Scenes serialize to a deterministic text format: save an unchanged scene, get a byte-identical file. Git diffs show only real changes; level edits can be code-reviewed; merges work. Every entity and asset has a permanent UUID rename anything, move any file, nothing breaks, and a stale runtime reference is a clean `IsAlive == false`, never a crash. *(This year Phases 2–3.)*

**Real-time collaboration, self-hosted.** Every editor mutation is an operation in a single stream the same stream that powers undo/redo. Live co-editing (CRDT-merged scenes, presence, asset locks) is a subscriber to that stream, against a server that is one small open-source binary on your own machine. Solo and offline work is the identical code path with zero peers. Hosted collaboration will exist as a paid convenience it will never gate a feature the self-hosted server lacks. *(Roadmap foundations being laid now.)*

**A renderer you can see inside.** The frame is a render graph data, not hidden code. The editor shows the pass DAG, per-pass GPU timings, and any intermediate texture on click. Barriers and sync are derived from declared usage, so adding a render pass can't introduce a synchronization bug. Path tracing arrives as an alternative subgraph over the same scene starting with an in-editor reference path tracer: **press a key, see ground truth**, validate every raster feature against an unbiased render on any Vulkan GPU. *(Graph: this year Phase 4. Reference tracer: roadmap.)*

**Packages that resolve Cargo for game content.** Versioned packages carrying assets *and* C# code, with manifests, semver, transitive resolution, and lockfiles. Content-addressed assets mean no path collisions and no duplicated data. Registries are simple, static, and self-hostable a team-private registry on your own server is a first-class case. *(Roadmap.)*

**Errors that explain themselves.** Every failure names the entity, asset, or pass involved and suggests a fix. A bad asset is a pink placeholder plus a named error. A throwing script disables itself, logs script/method/stack, and the engine keeps rendering even if every script in the scene throws. *(Error isolation: today. Structured logging: this year Phase 2.)*

---

## What works today

- **Rust core** with a lightweight ECS entities, components, queries, resources.
- **Vulkan renderer** (`vulkano`): forward+ shading, MSAA, SSAO, HDR tonemapping, point & directional lights, textured materials.
- **In-window editor** (`egui`): scene hierarchy, inspector, environment controls, and a live performance HUD (FPS, CPU/GPU frame time, VRAM).
- **C# scripting** (.NET / CoreCLR): Unity-style `Behaviour` scripts with `OnStart`/`OnUpdate`, driven from the engine with script exceptions isolated so they can never crash the engine.

## The road to a shippable engine (2026)

Five public milestones take Orrin from tech demo to "a newcomer ships a small game with only the public C# API":

1. **Scripting Foundation** — the permanent C# API surface: collision, entity querying, error isolation, core math. *(~75% complete)*
2. **External Projects & Hot Reload** — your game as an external `game.dll`, hot reload, `orrin` CLI, project structure, prefabs, input mapping.
3. **Editor Usability & Asset Pipeline** — the editor becomes the authoring tool: undo/redo, play/stop, deterministic scene persistence, UUID + content-addressed asset pipeline.
4. **Visual & Gameplay Foundations** — render graph, shadow mapping, transform hierarchy, many-light forward+, in-game UI, audio.
5. **Shippability** — PBR material system (bindless, path-tracing-ready), scene management, `orrin build --export`, a complete demo game, and docs.

Beyond the showcase: skeletal animation, physics (Rapier), the render-graph inspector and reference path tracer, the package ecosystem, live collaboration, and as the long-term north star real-time path tracing. Architecture decisions for all of these are recorded as ADRs in `docs/adr/` as they're made, so contributors can see not just what was decided but why.

## Quick start

```bash
# Run the engine
cargo run -p orrin-core

# With C# scripting (requires the .NET SDK)
dotnet build scripting/Orrin -c Debug
cargo run -p orrin-core --features scripting
```

## Tech

- **Core & renderer:** Rust + Vulkan (`vulkano`)
- **Scripting:** C# on .NET, hosted in-process via CoreCLR
- **Editor:** `egui`

## Contributing

TBD - This project is to early to be open for contribution. I am afraid people will go of course or misunderstand the project. After phase 5 everyone is open to contribution