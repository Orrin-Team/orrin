#version 460

// Depth-only: the render pass has no color attachment and this stage writes
// nothing. A vertex-only pipeline is legal Vulkan for this, but MoltenVK's
// handling of a nil Metal fragment function has been inconsistent, and an empty
// shader costs nothing measurable.
void main() {}
