# Fonts

- `JetBrainsMono-Regular.ttf` — [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono),
  SIL Open Font License 1.1 (`OFL.txt`). Every figure, path and log line.
- `Phosphor.ttf` — the regular weight of [Phosphor Icons](https://phosphoricons.com)
  2.1.1, MIT (`PHOSPHOR-LICENSE`), from `@phosphor-icons/web@2.1.1/src/regular/`.
  The editor's icon set: each icon is a glyph, named against its codepoint in
  `editor/icons.rs`. To add one, look its codepoint up in that release's
  `style.css` and add a line there; `every_icon_is_in_the_font` checks it.

Both are embedded at compile time, so a missing file is a build error rather
than a blank glyph in a shipped binary.
