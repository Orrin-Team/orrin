using Orrin;
using Orrin.Math;

namespace DemoGame;

// Storybook stories: each builds one piece of UI onto the canvas it is handed,
// in a world of its own. Open them from the editor's Storybook tab.
public static class DemoStories
{
    static readonly Color Ink = new(0.95f, 0.96f, 0.98f);
    static readonly Color Accent = new(0.27f, 0.52f, 0.96f);
    static readonly Color Surface = new(0.17f, 0.18f, 0.21f);

    static Entity Element(Entity parent, RectTransform rect)
    {
        var entity = World.Spawn();
        World.AddComponent(entity, rect);
        entity.SetParent(parent, keepWorld: false);
        return entity;
    }

    static RectTransform Centered(Vector2 size) => new() { SizeDelta = size };

    static RectTransform Stretched() => new()
    {
        AnchorMin = Vector2.zero,
        AnchorMax = Vector2.one,
        SizeDelta = Vector2.zero,
    };

    static Entity Label(Entity parent, string text, float size)
    {
        var label = Element(parent, Stretched());
        World.AddComponent(label, new Text { Size = size, Color = Ink, Anchor = TextAnchor.MiddleCenter });
        World.SetText(label, text);
        return label;
    }

    // A pointer-driven button: hover and press scale it through tweens, and
    // the Disabled control shows its resting state with the tint it gets.
    [Story("Buttons/Primary")]
    static void PrimaryButton(StoryContext ctx)
    {
        var width = ctx.Float("Width", 220f, 80f, 480f);
        var disabled = ctx.Bool("Disabled");

        var button = Element(ctx.Canvas, Centered(new Vector2(width, 56f)));
        World.AddComponent(button, new Image { Color = Color.white });
        World.AddComponent(button, new Button
        {
            Normal = Accent,
            Hover = new Color(0.36f, 0.6f, 1f),
            Pressed = new Color(0.2f, 0.42f, 0.84f),
            Disabled = new Color(0.32f, 0.34f, 0.38f),
            Interactable = !disabled,
        });
        World.AddComponent(button, new Corner { Radius = 14f });
        World.AddComponent(button, new Shadow { Color = new Color(0f, 0f, 0f, 0.35f), Offset = new Vector2(0f, 6f), Blur = 12f });
        Label(button, disabled ? "Unavailable" : "Play", 24f);
        if (!disabled)
            World.AddScript<Pop>(button);
    }

    // A toast that slides in from the top, waits, and then says it is done —
    // the completion callback is what changes the label.
    [Story("Feedback/Toast")]
    static void Toast(StoryContext ctx)
    {
        var delay = ctx.Float("Delay", 0.2f, 0f, 2f);

        var toast = Element(ctx.Canvas, new RectTransform
        {
            AnchorMin = new Vector2(0.5f, 0f),
            AnchorMax = new Vector2(0.5f, 0f),
            Pivot = new Vector2(0.5f, 0f),
            AnchoredPosition = new Vector2(0f, -80f),
            SizeDelta = new Vector2(360f, 64f),
        });
        World.AddComponent(toast, new Image { Color = Surface });
        World.AddComponent(toast, new Corner { Radius = 12f });
        World.AddComponent(toast, new Shadow { Color = new Color(0f, 0f, 0f, 0.4f), Offset = new Vector2(0f, 8f), Blur = 16f });
        var label = Label(toast, "Saving…", 20f);

        Tween.To(toast, TweenProperty.Position, new Vector2(0f, 32f),
                new TweenInfo(0.3f, EaseStyle.Quint, EaseDirection.Out, delay))
            .OnComplete(() => World.SetText(label, "Saved"));
    }

    // Every modifier at once, with the gradient turning forever.
    [Story("Cards/Gradient")]
    static void GradientCard(StoryContext ctx)
    {
        var seconds = ctx.Float("Seconds per turn", 4f, 0.5f, 12f);

        var card = Element(ctx.Canvas, Centered(new Vector2(320f, 200f)));
        World.AddComponent(card, new Image { Color = Color.white });
        World.AddComponent(card, new Corner { Radius = 24f });
        World.AddComponent(card, new Stroke { Color = new Color(1f, 1f, 1f, 0.25f), Thickness = 2f });
        World.AddComponent(card, new Gradient { Start = Accent, End = new Color(0.62f, 0.33f, 0.92f) });
        Label(card, "Orrin", 40f);

        Tween.To(card, TweenProperty.GradientRotation, 360f,
            new TweenInfo(seconds, EaseStyle.Linear, repeat: -1));
    }
}

// Scales its element up a touch under the pointer and down while pressed.
public class Pop : Behaviour, IPointerEnterHandler, IPointerExitHandler, IPointerDownHandler, IPointerUpHandler
{
    static readonly TweenInfo Quick = new(0.12f, EaseStyle.Quad, EaseDirection.Out);

    void ScaleTo(float scale) => Tween.To(Entity, TweenProperty.Scale, new Vector2(scale, scale), Quick);

    public void OnPointerEnter(PointerEvent pointer) => ScaleTo(1.04f);
    public void OnPointerExit(PointerEvent pointer) => ScaleTo(1f);
    public void OnPointerDown(PointerEvent pointer) => ScaleTo(0.97f);
    public void OnPointerUp(PointerEvent pointer) => ScaleTo(1.04f);
}
