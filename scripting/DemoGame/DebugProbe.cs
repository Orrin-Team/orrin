using Orrin;
using Orrin.Math;

namespace DemoGame;

public class DebugProbe : Behaviour
{
    private float _t;

    public override void OnStart()
    {
        Debug.Log("DebugProbe started");
        Debug.LogWarning("warning test");
        Debug.LogError("error test");
    }

    public override void OnUpdate(float deltaTime)
    {
        _t += deltaTime;
        
        Debug.DrawLine(new Vector3(-3f, 0f, 0f),new Vector3(-3f, 0f, 0f), Color.green);
        
        var dir = new Vector3(Mathf.Cos(_t), 0f , Mathf.Sin(_t)) * 2f;
        Debug.DrawLine(LocalTransform.Position, dir, Color.cyan);
        
        if (Mathf.Repeat(_t, 1f) < deltaTime)
            Debug.DrawLine(LocalTransform.Position, LocalTransform.Position + Vector3.up * 2f, Color.magenta, 2f);
    }
}