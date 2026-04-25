# Clearwater Asset Pipeline

This guide is the working pipeline for the underwater campaign site.

The goal is not "collect some nice pictures".
The goal is to build a consistent campaign system where every asset has:
- a scene owner
- a source file
- a compressed runtime export
- a clear reason to exist

## 1. Folder Strategy

Use these directories consistently:

```text
assets-src/
  stills/
    surface/
    intro/
    hero/
    cards/
    lab/
  motion/
    loops/
    transitions/
    grain/
  3d/
    blender/
    spline/
    glb/
  comps/
    figma/
    psd/
    ae/

public/campaign/
  images/
    surface/
    intro/
    hero/
    cards/
    lab/
  video/
  sprites/
  glb/
  lottie/

lib/assets/manifest.ts
```

Rules:
- `assets-src/` is editable source.
- `public/campaign/` is web-ready export only.
- The app should reference only `public/campaign/` paths.
- Add important files to `lib/assets/manifest.ts` so scenes stay clean.

## 2. What To Build First

Do these in order:

1. Surface opening plate
2. Surface caustics loop
3. Intro light beam plate
4. Hero depth plate
5. Hero ambient loop
6. Cards sheen or texture overlay
7. Lab before image
8. Lab after image

Do not start with characters, creatures, or full cinematic 3D shots.
First make the environment believable.

## 3. Asset Types By Scene

### Surface
- `AVIF` wide matte plate for the horizon and water mass
- short seamless `MP4` caustics or shimmer loop
- optional subtle grain or spray overlay

### Intro
- still beam plate
- still particulate haze plate
- SVG or CSS grid overlays

### Hero
- one premium still plate with strong depth separation
- one ambient motion loop only if the static version feels dead
- future: one `GLB` hero object if the still version is already excellent

### Cards
- mostly CSS-driven
- optional texture overlays and reflective sheens
- no heavy video here

### Lab
- before/after demo images
- maybe one tiny loop later for processing state

## 4. Export Specs

### Stills
- Format: `AVIF`
- Fallback: `WebP`
- Use `PNG` only if you need transparency
- Keep final runtime stills under `700 KB` whenever possible

Target sizes:
- hero plate: `2200 to 2800 px` wide
- section stills: `1400 to 2000 px` wide
- mobile-specific crops later if needed

### Loops
- Format: `MP4` first, `WebM` optional later
- Duration: `6 to 8 seconds`
- Frame rate: `24 fps`
- Resolution: `1280x720` to start
- Always export muted
- Use `preload="none"` unless above the fold and essential

### 3D
- Format: `GLB`
- First object budget: under `3 MB`
- One object only in phase one
- No huge textures

## 5. Naming Convention

Use this pattern:

```text
scene-subject-purpose-v001.ext
hero-depth-plate-v001.avif
surface-caustics-loop-v001.mp4
lab-demo-before-v001.webp
hero-submersible-v001.glb
```

When replacing an asset:
- make a new version
- update the manifest
- remove older versions only when the replacement is approved

## 6. AI Image Generation Workflow

If you use image generation, do not ask for "beautiful underwater scene".
That produces generic wallpaper.

Use prompt blocks with:
- subject
- camera
- lighting
- mood
- material detail
- composition
- exclusions

### Prompt Template

```text
cinematic underwater campaign still, premium editorial art direction,
[subject],
wide lens composition, strong foreground/midground/background separation,
sun shafts filtered through ocean surface, suspended particulate matter,
cool blue-green palette, off-white highlights, soft atmospheric depth,
minimal but luxurious, realistic water attenuation, subtle caustics,
high-end brand campaign image, detailed but restrained, no clutter,
no cartoon look, no neon fantasy, no diver mask close-up, no text
```

### Scene Prompts

#### Surface opening

```text
cinematic underwater campaign still, premium editorial art direction,
ocean surface seen from just below the waterline, distant light bloom above,
wide lens composition, strong depth separation, calm but mysterious,
sun shafts filtered through ocean surface, suspended particulate matter,
cool blue-green palette, off-white highlights, subtle caustics,
minimal but luxurious, realistic water attenuation, no boats, no people, no text
```

#### Hero environment

```text
cinematic underwater campaign still, premium editorial art direction,
deep ocean restoration environment, layered reef silhouettes and subtle engineered forms,
wide lens composition, strong foreground coral shadow, mid-depth haze, distant glow,
soft light absorption, premium minimalism, realistic underwater atmosphere,
high-end brand campaign image, no fish swarm, no fantasy palace, no text
```

#### Lab before image

```text
realistic underwater photograph, degraded visibility, green-blue cast, backscatter,
soft loss of contrast, slight haze, believable distortion, documentary framing,
no fake blur, no overdone noise, no fantasy lighting
```

#### Lab after image

```text
realistic restored underwater photograph, improved clarity, corrected color balance,
retained realism, preserved natural haze depth, premium technical demonstration,
not oversharpened, not oversaturated, not fake HDR
```

## 7. Blender For Beginners

You do not need to master Blender to help this site.
Use Blender for simple premium assets first.

Start with these three outputs:
- one depth plate render
- one caustics light render
- one simple `GLB` object

### First 60-minute Blender path

Learn only this sequence:

1. Move around the viewport
2. Add a plane and a cube
3. Scale, rotate, move
4. Add a camera
5. Add one area light
6. Add fog using volume in the world or a cube with volume
7. Render one still
8. Export one `GLB`

Ignore advanced modeling at first.

### Your first useful Blender exercise

Build a hero "depth plate":

1. Create three large planes
2. Put one close, one mid, one far
3. Give each a different blue-green material
4. Add a bright area light from above
5. Add light fog
6. Put the camera at a slight upward angle
7. Render at `2560x1440`

This already teaches you:
- composition
- depth
- silhouette
- underwater light falloff

### Your second useful Blender exercise

Build a simple caustics loop:

1. Make a plane below the camera
2. Add a light from above
3. Animate noise or a moving texture
4. Render `6 seconds` at `24 fps`
5. Export a soft loop with no hard contrast

This can become:
- a hero loop
- a surface shimmer layer
- a CTA ambient layer

### Your third useful Blender exercise

Build one simple `GLB`:

Use:
- a submersible silhouette
- a scanner ring
- a floating buoy

Keep it simple:
- low geometry
- one or two materials
- no heavy particles
- no giant texture maps

## 8. What Not To Do

Avoid these common mistakes:

- Do not build the whole site around 3D on day one
- Do not use giant 4K videos everywhere
- Do not make every section a separate style
- Do not use oversaturated "fantasy aquarium" colors
- Do not over-sharpen restored demo images
- Do not commit random file names like `final-final-3.png`

## 9. Approval Flow For Each Asset

Before wiring an asset into the UI, check:

1. Does it belong to a specific scene?
2. Is there a source file in `assets-src/`?
3. Is there a runtime export in `public/campaign/`?
4. Is it small enough?
5. Is it listed in the manifest if it matters?
6. Does it match the blue-green premium palette?
7. Does it improve the story, not just add noise?

## 10. Immediate Next Step

Create these first actual files:

- `assets-src/stills/surface/surface-moodboard-v001.png`
- `assets-src/stills/hero/hero-depth-plate-v001.png`
- `assets-src/motion/loops/surface-caustics-loop-v001.mp4`
- `assets-src/stills/lab/lab-demo-before-v001.png`
- `assets-src/stills/lab/lab-demo-after-v001.png`

Then export web versions into:

- `public/campaign/images/surface/`
- `public/campaign/images/hero/`
- `public/campaign/video/`
- `public/campaign/images/lab/`

After that, wire the real files into `lib/assets/manifest.ts`.
