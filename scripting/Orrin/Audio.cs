using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

/// <summary>A WAV, Ogg Vorbis or MP3 file under the project's assets directory.</summary>
public sealed class AudioClip : Asset
{
    AudioClip() { }
}

/// <summary>
/// How a spatial sound quietens between its minimum and maximum distance.
/// Numbering is lock-step with the engine's.
/// </summary>
public enum AudioRolloff : uint
{
    /// <summary>
    /// Halves with every doubling of distance, as a sound in open air does;
    /// stops falling at the maximum distance.
    /// </summary>
    Inverse = 0,

    /// <summary>Falls in a straight line to silence at the maximum distance.</summary>
    Linear = 1,
}

/// <summary>
/// A sound that plays from an entity: from where it stands when
/// <see cref="Spatial"/>, or into both ears alike when not. Read and write it
/// with <see cref="World.GetComponent{T}"/>, <see cref="World.SetComponent{T}"/>
/// and <see cref="World.AddComponent{T}"/>; start and stop it with
/// <see cref="Audio.Play(Entity)"/> and <see cref="Audio.Stop(Entity)"/>.
/// Distances are metres.
/// </summary>
// Layout must match Rust CAudioSource: a clip's 16 bytes and seven words.
[StructLayout(LayoutKind.Sequential)]
public struct AudioSource
{
    AssetBytes _clip;

    /// <summary>A linear gain, not decibels: 0.5 is half the amplitude.</summary>
    public float Volume;

    uint _loop;
    uint _playOnStart;
    uint _spatial;

    /// <summary>Inside this distance the sound is at its full volume.</summary>
    public float MinDistance;

    /// <summary>Past this distance the rolloff stops changing anything.</summary>
    public float MaxDistance;

    public AudioRolloff Rolloff;

    public AudioSource()
    {
        Volume = 1f;
        _playOnStart = 1;
        _spatial = 1;
        MinDistance = 1f;
        MaxDistance = 50f;
        Rolloff = AudioRolloff.Inverse;
    }

    /// <summary>The null reference plays nothing.</summary>
    public AssetRef<AudioClip> Clip
    {
        readonly get => new(_clip.ToGuid());
        set => _clip = AssetBytes.Of(value.Id);
    }

    public bool Loop
    {
        readonly get => _loop != 0;
        set => _loop = value ? 1u : 0u;
    }

    /// <summary>
    /// Starts when the game starts, or when the entity first appears in a
    /// running game.
    /// </summary>
    public bool PlayOnStart
    {
        readonly get => _playOnStart != 0;
        set => _playOnStart = value ? 1u : 0u;
    }

    /// <summary>Heard from where the entity stands, rather than in both ears alike.</summary>
    public bool Spatial
    {
        readonly get => _spatial != 0;
        set => _spatial = value ? 1u : 0u;
    }
}

// Layout must match Rust CSound: a clip's 16 bytes, an Entity, a Vector3 and
// six words.
[StructLayout(LayoutKind.Sequential)]
internal struct SoundRequest
{
    public AssetBytes Clip;
    public Entity Entity;
    public Vector3 Position;
    public float Volume;
    public uint Loop;
    /// 0 flat, 1 at Position, 2 on Entity.
    public uint Emitter;
    public float MinDistance;
    public float MaxDistance;
    public AudioRolloff Rolloff;
}

/// <summary>A sound that was started; the default handle is one that was refused.</summary>
public readonly struct SoundHandle : IEquatable<SoundHandle>
{
    /// <summary>The engine's id for the sound; zero for one that was refused.</summary>
    public readonly ulong Id;

    internal SoundHandle(ulong id) => Id = id;

    /// <summary>False once the sound has ended, been stopped, or was refused.</summary>
    public bool IsPlaying => Id != 0 && Native.AudioPlaying(Id);

    public bool Equals(SoundHandle other) => Id == other.Id;

    public override bool Equals(object? obj) => obj is SoundHandle other && Equals(other);

    public override int GetHashCode() => Id.GetHashCode();

    public static bool operator ==(SoundHandle a, SoundHandle b) => a.Id == b.Id;

    public static bool operator !=(SoundHandle a, SoundHandle b) => a.Id != b.Id;
}

/// <summary>
/// Sound effects, sounds placed in the world, and one music channel. Only a
/// running game makes a sound: leaving play stops every one.
/// </summary>
/// <remarks>
/// A sound starts once its clip has decoded, which a clip on an
/// <see cref="AudioSource"/> in the scene already has by the time the game
/// runs; any other clip decodes the first time it is played. Spatial sounds are
/// heard from the first entity with an AudioListener, or from the camera.
/// Volumes are linear gains. A refused call logs why and returns the default
/// handle.
/// </remarks>
public static class Audio
{
    /// <summary>Play <paramref name="clip"/> in both ears alike.</summary>
    public static SoundHandle Play(AssetRef<AudioClip> clip, float volume = 1f, bool loop = false) =>
        Start(Request(clip, volume, loop, 0, default, Vector3.zero, 1f, 50f, AudioRolloff.Inverse));

    /// <summary>Play <paramref name="clip"/> from a fixed point in the world.</summary>
    public static SoundHandle PlayAt(
        AssetRef<AudioClip> clip,
        Vector3 position,
        float volume = 1f,
        bool loop = false,
        float minDistance = 1f,
        float maxDistance = 50f,
        AudioRolloff rolloff = AudioRolloff.Inverse) =>
        Start(Request(clip, volume, loop, 1, default, position, minDistance, maxDistance, rolloff));

    /// <summary>
    /// Play <paramref name="clip"/> from wherever <paramref name="entity"/> is,
    /// for as long as it exists.
    /// </summary>
    public static SoundHandle PlayOn(
        AssetRef<AudioClip> clip,
        Entity entity,
        float volume = 1f,
        bool loop = false,
        float minDistance = 1f,
        float maxDistance = 50f,
        AudioRolloff rolloff = AudioRolloff.Inverse) =>
        Start(Request(clip, volume, loop, 2, entity, Vector3.zero, minDistance, maxDistance, rolloff));

    /// <summary>Play <paramref name="entity"/>'s <see cref="AudioSource"/> from the start.</summary>
    public static SoundHandle Play(Entity entity) => new(Native.AudioPlaySource(entity));

    /// <summary>Stop what <paramref name="entity"/>'s source is playing. False if it was not.</summary>
    public static bool Stop(Entity entity) => Native.AudioStopSource(entity);

    /// <summary>Stop a sound, fading it over a few milliseconds. False if it had already ended.</summary>
    public static bool Stop(SoundHandle sound) => sound.Id != 0 && Native.AudioStop(sound.Id);

    /// <summary>
    /// A linear gain on top of the sound's own, and of an AudioSource's volume.
    /// False if the sound has ended.
    /// </summary>
    public static bool SetVolume(SoundHandle sound, float volume) =>
        sound.Id != 0 && Native.AudioSetVolume(sound.Id, volume);

    /// <summary>
    /// Play <paramref name="clip"/> on the music channel, replacing what it was
    /// playing. Music is decoded as it plays rather than up front.
    /// </summary>
    public static SoundHandle PlayMusic(AssetRef<AudioClip> clip, bool loop = true) =>
        new(Native.AudioPlayMusic(clip.Id, loop));

    public static void StopMusic() => Native.AudioStopMusic();

    /// <summary>The music channel's gain, under <see cref="MasterVolume"/>.</summary>
    public static float MusicVolume
    {
        set => Native.AudioSetMusicVolume(value);
    }

    /// <summary>The gain on everything the game plays.</summary>
    public static float MasterVolume
    {
        set => Native.AudioSetMasterVolume(value);
    }

    static SoundRequest Request(
        AssetRef<AudioClip> clip,
        float volume,
        bool loop,
        uint emitter,
        Entity entity,
        Vector3 position,
        float minDistance,
        float maxDistance,
        AudioRolloff rolloff) => new()
    {
        Clip = AssetBytes.Of(clip.Id),
        Entity = entity,
        Position = position,
        Volume = volume,
        Loop = loop ? 1u : 0u,
        Emitter = emitter,
        MinDistance = minDistance,
        MaxDistance = maxDistance,
        Rolloff = rolloff,
    };

    static SoundHandle Start(in SoundRequest request) => new(Native.AudioPlay(request));
}
