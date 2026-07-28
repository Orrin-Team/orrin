# Orrin Architecture Plan

This document describes the long-term architecture for Orrin. It assumes the current state of the codebase (Rust core with a lightweight ECS, Vulkan renderer via vulkano, egui editor, C# scripting hosted in-process through CoreCLR) and the Phase 1 to 5 showcase roadmap. The point of writing it down now is to make sure the big features planned for later, meaning real-time collaboration, the asset and package ecosystem, and path tracing, can be built on top of what exists without a rewrite. The project is planned on a horizon of roughly a decade, so decisions here favour designs that stay correct over designs that ship fastest.

## 1. What mainstream engines get wrong

Unity and Unreal are very capable, and this plan doesn't pretend otherwise. But they share a consistent blind spot: they optimise the moment of creation and neglect the daily grind of iterating, collaborating, and owning your work. Orrin targets six specific failures.

Iteration speed. Unreal's compile-and-restart loop and Unity's domain reload stalls are the most complained-about aspects of both engines. Orrin treats "change code, see the result in under two seconds" as a hard constraint that every subsystem has to respect, covering game code, shaders, and assets.

Version control and collaboration. Unity scenes produce constant merge conflicts. Unreal's answer is binary files plus Perforce plus exclusive locking. Neither engine can put two people in the same scene at the same time. Orrin treats the scene as a mergeable, diffable, concurrently editable document from the start (section 4).

Renderer opacity. Unity's SRP is hard to see into and Unreal's renderer is a monolith in practice. Debugging a slow or wrong frame means external tools and guesswork. Orrin's render graph is data: inspectable in the editor, profiled per pass, and extensible by registering nodes instead of forking the engine (section 3).

Dependency management. Unity's package manager is shallow and its asset store has no dependency resolution. Unreal's plugin ecosystem has none either. There is currently no good answer to "I want to share a gameplay system, with its assets, versioned and self-hostable." Orrin ships a real package manager over content-addressed assets (section 5). The short description is Cargo, for game content.

Ownership. Licensing changes, runtime fees, closed editors. Orrin's position is full source including the editor, and no subsystem may ever depend on a service that only Orrin-the-organisation can run. Anything networked (collaboration server, registry, build cache) must be self-hostable, with paid hosting as a convenience on top.

Diagnostics. GPU device losses with no explanation, shaders that silently render black, assets that fail to import without saying why. Orrin borrows the Rust compiler's attitude: a failure names the entity, asset, or pass involved and suggests a likely fix. Cheap to build early, nearly impossible to retrofit.

Everything below is justified against these six commitments.

## 2. Core data architecture

Whether collaboration, the asset ecosystem, and path tracing turn out to be feasible depends heavily on one early decision: is the engine's state plain data addressed by stable IDs, or objects connected by pointers? Orrin commits to plain data.

### 2.1 The ECS is the source of truth

The existing ECS becomes the canonical representation of everything: game entities, editor state, and eventually render passes and in-scene asset references. Components are plain serializable data. Systems own all behaviour. No component may hold a raw pointer or a direct reference to another entity or asset, only an EntityId or AssetId handle.

This one rule is what makes scenes serializable, diffable, syncable over a network, and safe to hot-reload. It's worth enforcing mechanically (a lint, or a sealed set of allowed field types) rather than by convention, because a single violation quietly breaks all three flagship features and probably won't be noticed for years.

### 2.2 Stable identity

Two ID spaces, both 128-bit UUIDs assigned at creation and never reused.

EntityId identifies an entity across sessions, machines, and collaborators. It is not the ECS's internal index. The ECS keeps using dense generational indices internally for iteration speed, with a map between the two. Serialization, networking, and the editor speak UUIDs; hot loops speak indices. In the mature version of this, the serializer is the only place that translates between them.

AssetId identifies an asset independent of its file path. Paths are a human convenience resolved through the asset database. Nothing in a scene, component, or package references a path, which is what makes rename, move, deduplication, and remote fetch all trivial.

### 2.3 The component registry

Every component type registers once into a central registry: how to serialize it, how to diff two instances, how to apply a diff, how to draw its inspector UI, and its default value. From that single registration the engine derives scene save and load, undo and redo, the inspector, prefab overrides, and later the collaboration protocol, with no per-feature code.

This registry is the highest-leverage piece of infrastructure in the whole engine. It should be built during Phases 2 and 3 and treated as load-bearing forever after. C# components participate through a small interop shim that exposes their fields as typed property bags, so scripted components get saving, inspection, and sync support automatically rather than being second-class.

### 2.4 Scene format

Scenes serialize to a line-oriented, deterministic text format: one entity per block, components sorted by type, fields sorted by name, floats printed canonically. Determinism matters more than the syntax. Identical scenes must produce byte-identical files, so that git diffs are meaningful and CRDT snapshots are stable.

Binary payloads (baked lightmaps, navmesh data) never live in the scene file; they're assets referenced by AssetId. Every file starts with a version number, and migrations between versions are explicit functions kept in the repo. A project on this timescale will change its formats, and ad-hoc migration is how engines rot.

## 3. Renderer

### 3.1 Render graph

The current pipeline (forward+, MSAA, SSAO, HDR tonemap) should be restructured into an explicit render graph before Phase 4 adds shadows. Passes become nodes that declare their inputs, outputs, and resource usage; the graph compiles that into execution order, barriers, and layout transitions.

This isn't mainly a performance decision. A graph is what makes the renderer inspectable (the editor can show the pass DAG, per-pass GPU timings, and any intermediate texture on click), extensible (a user pass is a node registration), and replaceable (path tracing becomes an alternative subgraph rather than a second renderer). It also removes the worst Vulkan bug class: since barriers are derived from declarations, a pass author can't write an incorrect one.

The timing matters because shadow cascades are the first feature that multiplies pass count. Migrating five passes into a graph is a contained refactor. Migrating fifteen, once the editor depends on frame structure, is surgery.

### 3.2 Bindless, scheduled deliberately

Path tracing has one hard prerequisite: when a ray can hit anything, the shader has to be able to read any material, texture, and mesh without the CPU rebinding descriptors. That means descriptor-indexed (bindless) resources and a GPU-resident scene: a global vertex and index arena, a per-instance buffer, and a material table indexing into a global texture array.

This should land alongside the Phase 5 material system, because a material system built on per-draw descriptor sets would have to be rebuilt for path tracing, whereas a bindless material table serves the rasterizer today and rays later without change. vulkano supports descriptor indexing; this is the point to find out whether its ergonomics hold up. If they don't, dropping to ash for the RHI layer only is the natural seam, and everything above it stays put.

### 3.3 One material model, two integrators

The material system defines a single energy-conserving PBR parameterization (base colour, metallic, roughness, emission, normals, transmission later) and treats the forward+ rasterizer and the future path tracer as two integrators over the same materials and lights. Lights are defined in physical units (lumens, lux) from the first release, because retrofitting units later means re-tuning every scene anyone has ever authored.

The payoff is that content made during the showcase years never gets re-authored, and the two integrators can be compared pixel for pixel. Which leads to the most useful early step:

### 3.4 A reference path tracer, early and slow

Well before real-time ray tracing is attempted, Orrin should ship an offline reference path tracer as an editor mode: brute force, unbiased, seconds per frame, running as a compute shader against the same bindless scene. Press a key, see ground truth.

It earns its keep three ways. It's a correctness oracle for every raster feature (SSAO, shadows, tonemapping all get validated against an unbiased render). It builds the actual infrastructure that real-time path tracing needs later: acceleration structures, ray payloads, BSDF sampling. And it's a showcase feature in its own right, since neither Unity nor Unreal offers a one-keystroke in-editor ground-truth comparison. Real-time path tracing then becomes an optimisation programme over proven pieces (hardware RT, ReSTIR-family sampling, denoising) that can be scheduled over years instead of being a research project.

### 3.5 Diagnostics

Validation layers on in every dev build. Device loss dumps the render graph state with the offending pass named. Shader errors surface in the editor with file and line. One button captures a frame's graph, resources, and timings to a file that can be attached to a bug report. None of this is expensive now, and it compounds into reputation.

## 4. Real-time collaboration

Collaboration is Orrin's most defensible differentiator, and it's also the feature that depends most on the groundwork in section 2. If components are ID-addressed plain data with registry-derived diffs, collaboration is an engineering project. If they aren't, it's impossible, which is roughly where Unity and Unreal find themselves.

### 4.1 CRDT scenes, locked binaries

Two different problems hide inside the word collaboration and they need different mechanisms.

Structured scene data (entities, components, hierarchy) becomes a CRDT document. Each editing session is a replica. The operations are the same property diffs the registry already produces, plus create, delete, and reparent. CRDTs merge automatically and deterministically: two people move different objects and both edits land; two people move the same object and the last write wins per field. Rust has mature CRDT libraries (Automerge, Loro, yrs), and Orrin should adopt one behind an engine-owned SceneDocument API rather than building this from scratch.

Binary assets don't merge, and pretending they might is a research trap. For textures, meshes, and audio the server provides checkout locks: open a texture for editing and others see who holds it. This is the one place the Perforce model is actually right.

A useful consequence of the CRDT choice: offline editing costs nothing. A solo developer's session is a replica with no peers. Collaboration infrastructure never taxes the person not using it, which keeps the design honest.

### 4.2 The server

The collaboration server is a single small Rust binary. It relays and persists operations per project, manages asset locks, brokers presence (who's in the scene, their selection, their camera, drawn as ghosts in other viewports), and serves asset blobs. Transport is WebSocket or QUIC over TLS, auth is token-based, and the whole thing is open source and self-hostable with one command.

The paid offering is hosting, not features: zero-ops servers, backups, access management, bandwidth. The hosted version must never do anything the self-hosted one can't, because the first time it does, the ownership pitch is dead. This is the Gitea and GitLab model, and it aligns the business incentive with keeping the open server good.

### 4.3 Play mode and scripts

Collaborative editing applies to edit mode. Entering play mode forks the current document snapshot locally; peer edits queue and apply on stop. Live shared play sessions are a multiplayer-game problem, not an editor problem, and conflating the two has sunk other attempts. C# source files are collaborated on through git like any code. Script component values in the scene are CRDT data like everything else.

### 4.4 What this costs now

Nothing in Phases 1 to 5 implements any of this. The only requirement is that Phase 3's scene persistence routes all mutation through a single command and diff API, which undo and redo need anyway. Undo, single-user persistence, and multi-user sync then become three consumers of one mutation stream. That's the entire price of keeping collaboration cheap: one discipline, imposed now.

## 5. Assets and packages

Mainstream engines treat "assets in my project" and "packages from elsewhere" as unrelated systems. Orrin unifies them. A package is a versioned, signed collection of assets plus optional C# code, and the local asset database is the union of the project's own assets and its resolved dependencies.

### 5.1 The asset database

Source files (glTF, PNG, WAV) live in the project tree and belong to the user and to git. The engine watches them and imports each into compiled artifacts (GPU-ready meshes, transcoded textures) in a local cache, keyed by a hash of the source bytes, the importer version, and the import settings. Settings live in a small text sidecar next to the source, so they're diffable and collaborative. The cache is disposable and never committed.

Hot reload falls out of this naturally: the watcher fires, the asset reimports, subscribers to that AssetId are notified, GPU resources swap between frames. Runtime references go through typed handles with explicit load states, so a missing asset is a pink placeholder and a named error rather than a crash.

The cache being content-addressed quietly powers a lot downstream: deduplication across packages, a shared team cache on the same server as the collaboration service (nobody reimports what a teammate's machine already imported), and reproducible export builds.

### 5.2 The package manager

Cargo's model applied to game content. An orrin.toml manifest with name, semver version, dependency ranges, and declared contents; a lockfile committed to the project; a resolver for transitive dependencies. Registries are deliberately boring, self-hostable static services (an index plus content-addressed blobs), and a project can pin several, including a private one on a team's own server. A public community registry hosted by Orrin is the ecosystem play, but as with collaboration, hosting is convenience rather than privilege.

Because assets carry UUIDs and content hashes, a package's assets drop into a project with no path conflicts, and two packages sharing a texture store it once. Package code loads through CoreCLR the same way project code does, and runs under the same error isolation being built in Phase 1: a misbehaving package logs and disables itself instead of taking down the editor.

What this unlocks that Unity and Unreal don't have: a team-private registry of studio assets with real versioning, a single command that pulls a character controller along with its animations and config as one resolvable unit, reproducible builds from a lockfile, and community infrastructure any user can replicate.

## 6. Invariants

These aren't features, just constraints every phase has to preserve, listed so they can be enforced in review.

Hot reload of C# game code survives every later system; anything that requires an editor restart on a script change is a regression. Shader edits swap within a frame or two. Asset edits propagate without a restart. Editor cold start stays under a few seconds, guarded by a CI benchmark, because startup time decays one dependency at a time. Every user-facing failure names the thing involved and suggests a fix. The performance HUD grows into the render graph profiler rather than staying a separate widget.

## 7. Sequencing

The showcase phases keep their scope. This plan changes how three of their issues get built and defines what comes after.

Phase 2 (external projects and hot reload): route all scene mutation through the single command pipeline. Build the component registry here or at the start of Phase 3, since prefabs and hot reload both want it.

Phase 3 (editor usability and asset pipeline): the asset pipeline issue implements section 5.1 in its future-proof shape, which is the same scope the milestone already describes. Scene persistence serializes through the mutation pipeline into the deterministic format with a version header.

Phases 4 and 5 (visuals, materials, shippability): restructure existing passes into the render graph before adding shadows, not after. The material system implements the bindless table and physical light units. Export packaging consumes the content-addressed cache.

Phase 6, first half of 2027: render graph inspector and frame capture; the reference path tracer, which is the visible payoff for the Phase 4 and 5 groundwork; skeletal animation components designed into the ECS before more systems assume static meshes; and a physics decision, which realistically means integrating Rapier rather than growing the Phase 1 collision system into a physics engine.

Phase 7, through 2027: package manager, registry, shared import cache. This comes before collaboration because it exercises the same server infrastructure with lower protocol risk.

Phase 8, 2027 into 2028: the CRDT scene document behind the existing mutation pipeline, the self-hosted server, presence, asset locks, then the hosted offering once self-hosting is stable. This lands around the point the project realistically expects an audience that can use multi-user editing.

Phase 9 onward: real-time path tracing as an alternative render graph over the same scene and materials, benchmarked against the reference tracer that has been validating the renderer for two years by then.

## 8. Summary

The bet is that the next engine people love wins on daily developer experience rather than renderer feature count: fast iteration, scenes that merge and sync like text, a renderer you can see inside, dependencies that resolve like Cargo, errors that explain themselves, infrastructure you can host yourself, and path-traced ground truth as proof the engineering is serious.

Architecturally it all reduces to a few early disciplines. Plain-data components addressed by stable UUIDs. One registry that derives serialization, diffing, and UI from each component. One mutation pipeline feeding undo, persistence, and eventually sync. A data-driven render graph over a bindless GPU scene. Content-addressed assets behind handles. None of these delays the 2026 showcase, and each one is the difference between the 2028 features being an implementation project and being a rewrite.