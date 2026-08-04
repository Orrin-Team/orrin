# Icons

[Lucide](https://lucide.dev) 0.446.0, ISC licensed — see `LICENSE`. Vendored
rather than fetched: `egui::include_image!` embeds each file at compile time, so
a missing icon is a build error instead of a blank square in a shipped binary.

Only the glyphs the editor actually draws are here. To add one, take it from
`lucide-static@0.446.0` and apply the same two edits every file in this
directory has:

- `stroke="currentColor"` → `stroke="#ffffff"`. resvg resolves `currentColor` to
  black, and `Image::tint` multiplies — so a black icon stays black in every
  colour. White is the identity for that multiply.
- `stroke-width="2"` → `stroke-width="1.5"`. The design system's weight; 2 fights
  the editor's 1px chrome.
