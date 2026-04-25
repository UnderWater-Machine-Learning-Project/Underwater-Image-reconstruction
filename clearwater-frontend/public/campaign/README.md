# Campaign Runtime Assets

This folder contains optimized assets that the app can load directly.

Rules:
- keep paths stable
- use compressed web formats
- keep scene ownership obvious
- update `lib/assets/manifest.ts` when adding or replacing anything important

Recommended formats:
- stills: `avif`, then `webp` if needed
- alpha images: `webp` or `png`
- loops: `mp4` or `webm`
- vectors: `svg`
- 3D: `glb`
