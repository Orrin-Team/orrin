# Orrin.Math

The engine's public math API for C# scripts: `Vector2`, `Vector3`, `Vector4`,
`Quaternion`, `Mathf`, `Color`. The shape of the API deliberately follows
Unity's (`Mathf.PingPong`, `Quaternion.LookRotation`, lowercase `x`/`y`/`z`
fields) so it feels familiar, but the *conventions* are Orrin's own — read
them before assuming Unity behavior.

## Conventions (permanent — do not change casually)

| Topic | Convention |
| --- | --- |
| Handedness | **Right-handed**, matching glam on the Rust side. |
| Forward | **`Vector3.forward = (0, 0, -1)`** — *not* Unity's +Z. `back` is +Z. |
| Angles | Degrees at the API surface (`AngleAxis`, `Euler`, `Angle`, …). Convert with `Mathf.Deg2Rad` / `Mathf.Rad2Deg`. |
| Euler order | Z (roll), then X (pitch), then Y (yaw) — `Euler(x, y, z) = Ry * Rx * Rz`, numerically identical to Unity and `System.Numerics.CreateFromYawPitchRoll`. |
| Quaternions | Hamilton product; `a * b` applies `b` first, then `a`. `q * v` rotates a vector. Fields stored x, y, z, w. |
| Lerp/Slerp | Clamped by default; `*Unclamped` variants exist. |
| Equality | `==` is exact component equality. Note `q` and `-q` are the same rotation but compare unequal — use `Quaternion.Angle` for "same rotation". |
| Mutability | Every type is a `readonly struct`: operations return new values, `v.x = 1` does not compile. |
| Formatting | `ToString()` uses invariant culture (safe to parse from logs). |

## ABI lock-step

`Vector3` and `Quaternion` sit inside `Orrin.Transform`, which crosses the
native boundary as the Rust `CTransform` (`[f32; 3]`, `[f32; 4]` xyzw,
`[f32; 3]`). All types are `[StructLayout(LayoutKind.Sequential)]` and must
stay plain sequential floats. `Orrin.MathTests` asserts the sizes
(`Transform` = 40 bytes); if you add a field to any of these types, you are
changing the ABI — don't.

## Verification

```
dotnet run --project scripting/Orrin.MathTests
```

Checks the implementations against `System.Numerics` as an oracle (same
quaternion conventions) plus property tests for `LookRotation` /
`FromToRotation`, and the struct-layout asserts. Run it after touching
anything in this folder. It's a dependency-free console app on purpose;
convert to xunit when the scripting side grows a real test suite.

## Growing the API

- Add, don't rename: scripts in the wild will depend on these names.
- New angle-taking methods take degrees; new field layouts must stay blittable.
- If a method exists in Unity, match Unity's name and semantics unless the
  handedness/forward convention forces a difference — then document it here.
