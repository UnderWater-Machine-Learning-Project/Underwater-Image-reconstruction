# Asset Sources

This folder is the editable source-of-truth for campaign media.

Use it for:
- raw still compositions
- Photoshop or Krita files
- Figma boards
- After Effects comps
- Blender files
- Spline scene files

Do not load files from this folder directly in the app.
Only optimized exports from `public/campaign/` should be referenced at runtime.

If source files become very large, prefer Git LFS or keep the masters outside the repo and track only exports plus notes.
