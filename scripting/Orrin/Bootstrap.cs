using System.Runtime.InteropServices;

namespace Orrin;

public static unsafe class Bootstrap
{
    const int Ok = 0;
    const int AbiMismatch = 1;

    [UnmanagedCallersOnly]
    public static int Init(OrrinApi* api, int apiSize)
    {
        // ABI handshake. The engine passes the byte size of *its* OrrinApi; if
        // it differs from ours, the two were built against different definitions
        // of the struct. `Native.Initialize` would copy `sizeof(our OrrinApi)`
        // bytes out of the engine's table and later call through those offsets —
        // reading past a smaller table or landing on the wrong function pointer,
        // i.e. undefined behaviour on the first script call. Refuse first.
        //
        // sizeof grows automatically as fields are appended, so this needs no
        // manual version bump. Reported via stderr, not `Native.Log`, because the
        // log callback lives in the very table that may be malformed.
        int expected = sizeof(OrrinApi);
        if (apiSize != expected)
        {
            Console.Error.WriteLine(
                $"[Orrin] ABI mismatch: engine OrrinApi is {apiSize} bytes, this "
                + $"assembly expects {expected}. Rebuild the Orrin assembly against "
                + "the current engine (dotnet build scripting/Orrin).");
            return AbiMismatch;
        }

        Native.Initialize(api);
        Native.Log("hello from C#");
        return Ok;
    }
}
