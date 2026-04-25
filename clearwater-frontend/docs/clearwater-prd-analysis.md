# Clearwater Campaign Site - PRD and Implementation Analysis

Last updated: 2026-04-08

## 1. Executive Summary

Clearwater is now a short, cinematic underwater campaign website for an underwater image restoration project. The experience has been moved away from a long static landing page and toward a compact brand-campaign flow with a loader, smooth scroll, a master GSAP timeline, animated wave transitions, an underwater depth system, an interactive pipeline section, and a simulated restoration lab.

The current product direction is:

- Theme: premium underwater image restoration and ocean vision.
- Reference quality target: campaign-grade motion and presentation inspired by Shopify Editions-level execution, but not visually copied.
- Core story: surface entry -> system reveal -> restoration pipeline -> lab action.
- Primary technical stack: Next.js App Router, React, GSAP ScrollTrigger, Lenis, Zustand, CSS/Tailwind global styling.
- Current status: production build passes, but the product remains an art-directed prototype, not a finished restoration application.

The largest progress so far is the interaction system. The largest remaining gap is asset and product realism: generated PNG images are still heavy, the lab workflow is simulated, and the hero focal object is currently procedural rather than a final art asset or 3D object.

## 2. Product Objective

The site should feel like a living underwater brand campaign rather than a generic project page. The user should experience the project as a cinematic dive into a restoration system, with motion used to support story and depth rather than decoration.

Primary objectives:

- Present "Clearwater" as a premium underwater image restoration campaign.
- Communicate the system clearly: Dark Channel Prior + U-Net + 6-channel fusion.
- Use short-scroll storytelling instead of a long information-heavy page.
- Make transitions feel physical through waves, depth haze, light absorption, particles, and parallax.
- Provide a lab interaction that demonstrates before/after restoration intent.
- Preserve performance and architecture by avoiding new animation libraries for now.

Non-objectives for the current phase:

- No full Three.js or Spline integration yet.
- No real backend restoration inference yet.
- No heavy video-based hero or loader yet.
- No full production asset pipeline conversion yet.

## 3. Current User Journey

The current page flow is:

1. Loader
   A procedural ocean loader starts the experience with a Clearwater wordmark, rising wave, flash, blur/brightness impact, particles, light absorption, and a wave-to-hero handoff.

2. Surface / Descent
   The page opens on an underwater surface image with the Clearwater title treatment, team details, project summary, and a short campaign-style context block.

3. System / Hero
   The hero section introduces the restoration system and now includes a central procedural "restoration lens" focal object with a raw/restored split, scan pass, aperture pulse, and data-ring overlays.

4. Pipeline
   Three interactive cards explain the technical pipeline: Dark Channel Prior, U-Net Encoder, and 6-channel Hybrid fusion. Cards include tilt, hover glow, image plates, and breathing panel light.

5. CTA
   A visual portal section invites the user to enter the lab. The CTA button has idle scale, mouse magnet behavior, click ripple, and smooth scroll to the lab.

6. Lab
   The lab simulates the restoration pipeline with an upload/demo area, processing messages, a progress bar, and a before/after slider using static demo images.

## 4. Application Architecture

The app is still intentionally simple.

Entry points:

- `app/layout.tsx`
  Root layout. Loads fonts, global CSS, smooth scroll, loader, navigation, and page children.

- `app/page.tsx`
  Main campaign composition. Renders `CampaignRail`, wraps the story scenes in `CampaignTimeline`, and defines the current short-scroll scene order.

Core orchestration:

- `components/SmoothScroll.tsx`
  Initializes Lenis smooth scrolling and keeps GSAP ScrollTrigger in sync by calling `ScrollTrigger.update()` on Lenis scroll events.

- `components/CampaignTimeline.tsx`
  The page-level animation controller. It owns the main scroll timeline, depth parallax, scene reveals, wave transitions, light pass, ambient particles, panel breathing, and wave material motion.

- `lib/gsap.ts`
  Centralized GSAP setup. Registers ScrollTrigger only in the browser.

Navigation:

- `components/Nav.tsx`
  Fixed pill navigation with animated entrance and anchors for Story, System, Pipeline, and Lab.

- `components/CampaignRail.tsx`
  Fixed bottom depth index. Uses ScrollTrigger to mark active chapters.

State:

- `store/useStore.ts`
  Zustand store for lab state: `upload`, `processing`, and `result`.

Assets:

- `lib/assets/manifest.ts`
  Typed manifest for campaign image/video/3D paths. Scenes use this instead of hardcoding asset paths directly.

- `public/campaign/`
  Runtime assets used by the app.

- `assets-src/`
  Source pipeline area for editable stills, motion, 3D, and composition files.

## 5. Key Code Roles

| File | Role | Notes |
| --- | --- | --- |
| `app/layout.tsx` | App shell | Loads fonts, global CSS, `SmoothScroll`, `OceanLoader`, and `Nav`. |
| `app/page.tsx` | Page composition | Defines the current short campaign sequence. |
| `components/OceanLoader.tsx` | Opening cinematic loader | Procedural wave impact, flash, particles, absorption, and handoff to hero. |
| `components/CampaignTimeline.tsx` | Master animation system | Main ScrollTrigger timeline, parallax, wave transitions, light pass, ambient particles, panel breathing. |
| `components/SmoothScroll.tsx` | Lenis integration | Smooth scroll loop and ScrollTrigger syncing. |
| `components/CampaignRail.tsx` | Depth navigation | Active section tracking for the bottom chapter index. |
| `components/Nav.tsx` | Header navigation | Animated fixed nav and anchors. |
| `components/scenes/SurfaceScene.tsx` | Opening content scene | Combined surface/descent entry, title, project/team/flow metadata. |
| `components/scenes/HeroScene.tsx` | System hero | Main message, technical credits, prompt chips, procedural focal object. |
| `components/scenes/CardsScene.tsx` | Pipeline explanation | Three hover/tilt cards explaining the restoration method. |
| `components/scenes/CTAScene.tsx` | Action transition | Portal art, prompt chips, magnetic Enter Lab CTA, ripple feedback. |
| `components/scenes/LabScene.tsx` | Lab container | Switches between upload, processing, and result states. |
| `components/lab/UploadZone.tsx` | Lab entry UI | Prompt chips, pointer-follow glow, file input, demo trigger. |
| `components/lab/ProcessingState.tsx` | Simulated processing | Timed messages and fake progress loop. |
| `components/lab/BeforeAfterSlider.tsx` | Result demo | Static before/after comparison with draggable divider. |
| `app/globals.css` | Visual system | Global tokens, typography, layout, loader, wave material, hero object, cards, lab styling. |
| `lib/assets/manifest.ts` | Asset source of truth | Maps named campaign assets to runtime paths. |
| `docs/asset-pipeline.md` | Asset workflow | Existing guide for generating, exporting, naming, and approving assets. |

## 6. Implemented Feature Inventory

### 6.1 Smooth Scrolling

Implemented with Lenis in `SmoothScroll.tsx`.

Behavior:

- Smooth wheel scrolling.
- Touch sync enabled.
- ScrollTrigger updated during Lenis scroll.
- ScrollTrigger refreshed after initialization.

Current settings:

- `lerp: 0.065`
- `wheelMultiplier: 0.92`
- `touchMultiplier: 1.08`

Product impact:

- Makes the campaign feel less like a normal document scroll.
- Enables GSAP scrubbed animation to feel more cinematic.

### 6.2 Central GSAP Setup

Implemented in `lib/gsap.ts`.

Behavior:

- Imports GSAP and ScrollTrigger.
- Registers ScrollTrigger only when `window` exists.
- Prevents server-side registration errors in Next.js.

Product impact:

- Keeps animation setup consistent.
- Avoids registering plugins in every component.

### 6.3 Ocean Loader

Implemented in `OceanLoader.tsx`.

Loader sequence:

1. Clearwater title fades in with blur removal.
2. Foam particles build.
3. Wave rises.
4. Page underneath receives blur and brightness reduction.
5. Tiny camera shake creates a soundless impact.
6. Splash flash hits.
7. Light absorption shifts the screen darker.
8. Micro particles reveal.
9. Wave flattens into a thin foam/waterline.
10. Loader fades while hero is visible underneath.

Key implementation details:

- Body overflow is temporarily locked.
- The page underneath is targeted through `.campaign-timeline` and `#nav`.
- Loader restores page filters and body overflow on completion and cleanup.
- Uses procedural CSS/GSAP rather than video.

Why this matters:

- It avoids a hard cut between loading and content.
- It simulates the physical feeling of entering water.
- It keeps payload light compared to a high-resolution video loader.

### 6.4 Master Campaign Timeline

Implemented in `CampaignTimeline.tsx`.

Responsibilities:

- Sets initial hidden states for reveal targets.
- Controls the main scrubbed scroll timeline.
- Runs the global depth system.
- Coordinates reveal timing for hero, cards, CTA, and lab.
- Runs ambient animations that should always feel alive.
- Owns wave transitions between sections.

Current main timeline:

- Background depth: `yPercent: -5`, `scale: 1.055`.
- Mid depth: `yPercent: -10`.
- Foreground depth: `yPercent: -14`.
- Surface plate fades/dims slightly during the dive.
- Hero content reveals around `0.18` to `0.27`.
- Cards reveal around `0.34` to `0.38`.
- CTA reveals around `0.66` to `0.73`.
- Lab reveals around `0.84` to `0.88`.

Why this matters:

- The site now has one main pacing source.
- Scene sections feel connected instead of disconnected.
- Local component animations are mostly idle or micro-interactions rather than competing scroll timelines.

### 6.5 Depth System

Implemented with `data-depth` attributes.

Depth layers:

- `data-depth="background"` for plates and large environment.
- `data-depth="mid"` for visual objects and prompts.
- `data-depth="foreground"` for key text and CTA elements.

Current behavior:

- Background moves slowly.
- Mid layers move more.
- Foreground moves the most, but capped at `-14` to avoid abrupt text jumps.

Product impact:

- Creates fake 3D depth without WebGL.
- Keeps the underwater camera-rig feel.

Known tradeoff:

- Foreground depth cannot be pushed too hard because earlier tests made the Clearwater title and section text feel abrupt.

### 6.6 Wave Transitions

Implemented in `CampaignTimeline.tsx` and `globals.css`.

The transition system now uses:

- Inline SVG wave paths.
- GSAP morphing of `d` attributes between two wave shapes.
- SVG turbulence and displacement filters.
- Horizontal drift.
- ScaleY breathing.
- Base water layer.
- Foam edge layer.
- Light/caustic overlay layer.
- Haze and color absorption styling.

Transition beats:

- `0.14`
- `0.33`
- `0.60`
- `0.80`

Product impact:

- Replaces static cut-paper dividers with animated material transitions.
- Helps scenes feel physically connected.

### 6.7 Ambient Light Pass

Implemented in `CampaignTimeline.tsx` and `globals.css`.

Behavior:

- A fixed light pass moves across the campaign repeatedly.
- Uses blurred gradients and `mix-blend-mode: screen`.
- Runs independently from scroll.

Product impact:

- Makes scenes feel less static.
- Simulates underwater light shafts and refracted movement.

### 6.8 Ambient Particles

Implemented in `CampaignTimeline.tsx` and `globals.css`.

Behavior:

- 56 scoped campaign particles.
- Randomized positions, sizes, opacity, x/y drift, duration, and stagger.
- Uses scoped `.campaign-particle`, not global `.particle`.

Product impact:

- Creates constant depth emergence.
- Adds underwater micro-motion without requiring sprite assets yet.

### 6.9 Panel Breathing

Implemented through GSAP-updated CSS variable `--panel-breath`.

Targets:

- `.card`
- `.hero-chip`
- `.cta-chip`
- `.upload-z`

Behavior:

- GSAP animates `--panel-breath` between low values.
- CSS uses that variable inside radial gradients.

Product impact:

- Cards and chips feel alive without transform conflicts.
- Avoids fighting hover tilt or scroll parallax.

### 6.10 Surface / Descent Scene

Implemented in `SurfaceScene.tsx`.

Features:

- Combines the original surface and descent direction into one short opening scene.
- Uses generated surface plate and intro light-beam plate.
- Includes a shimmer layer and a flash reveal.
- Shows project title, copy, project metadata, team, and flow.

Current team text:

- Atharv Jandial
- Advitya Sharma
- Darsh Dhawan
- Divij Singh Soodan

Product impact:

- Provides the campaign landing moment.
- Avoids a long separate descent scene.

### 6.11 Hero Scene and Focal Object

Implemented in `HeroScene.tsx`.

Features:

- Core sentence: "The ocean holds secrets. We restore what was lost."
- Supporting technical copy.
- Prompt chips that scroll to the lab.
- Technical credits: Computer Vision, FastAPI - Python, DCP + U-Net.
- A procedural "Clearwater restoration lens" hero object.

Hero focal object behavior:

- Raw/restored split.
- Glass-like shell.
- Scan pass.
- Split divider.
- Aperture pulse.
- Light beams.
- Data rings and dots.
- Subtle floating motion.

Product impact:

- Addresses the previous missing hero anchor problem.
- Gives the campaign a recognizable visual motif until a real 3D/object asset is produced.

Known limitation:

- The focal object is procedural CSS, not final art.
- It is currently hidden on mobile through the existing responsive rule for `.hero-right`.

### 6.12 Pipeline Cards

Implemented in `CardsScene.tsx`.

Cards:

- Dark Channel Prior
- U-Net Encoder
- 6-ch Hybrid

Interactions:

- Mouse tilt through CSS variables `--rx` and `--ry`.
- Pointer-follow glow through `--mx` and `--my`.
- Image plate zoom and brightening on hover.
- Top line animation on hover.
- Panel breathing via `--panel-breath`.

Product impact:

- Turns technical explanation into an interactive exhibit.
- Keeps the content concise: visual -> message -> proof.

### 6.13 CTA Scene

Implemented in `CTAScene.tsx`.

Features:

- Background portal plate.
- Prompt chips.
- Magnetic "Enter the Lab" button.
- Click ripple effect.
- Smooth scroll to lab after click.

Product impact:

- Converts story into action.
- Keeps CTA aligned with the conversational lab idea.

### 6.14 Lab Prototype

Implemented across `LabScene.tsx`, `UploadZone.tsx`, `ProcessingState.tsx`, and `BeforeAfterSlider.tsx`.

Current states:

- `upload`
- `processing`
- `result`

Upload behavior:

- File input accepts images.
- Selecting a file starts the simulated processing state.
- "Try demo" also starts the simulated processing state.
- Prompt chips update local prompt text.

Processing behavior:

- Cycles through technical messages.
- Progress bar moves with randomized increments.
- Automatically finishes after about 5.5 seconds.

Result behavior:

- Displays static before/after images.
- Supports mouse and touch dragging.
- Shows demo metrics: PSNR, SSIM, and resolution.
- "New image" resets the lab state.

Product impact:

- Demonstrates the intended restoration product loop.
- Provides a working interactive demo shell.

## 7. Visual System

Current art direction:

- Palette: deep blue, cyan, teal/green signal accents, off-white text.
- Mood: premium underwater campaign, technical but cinematic.
- Typography:
  - `Cormorant Garamond` for editorial hero/display text.
  - `JetBrains Mono` for technical labels and campaign UI.
  - `DM Sans` for body/interface text.
- Visual motifs:
  - Light absorption.
  - Underwater haze.
  - Caustics.
  - Bubbles/particles.
  - Wave handoffs.
  - Restoration lens.
  - Depth index.

Important visual tokens in `globals.css`:

- `--bg`
- `--bg2`
- `--surface`
- `--surface2`
- `--border`
- `--border2`
- `--accent`
- `--accent2`
- `--signal`
- `--abyss`
- `--glass`
- `--text`
- `--text2`
- `--text3`

The earlier global overlay issue was addressed by keeping the `body::after` overlay subtle. It now supports atmosphere without heavy screen dimming.

## 8. Asset Pipeline State

Current runtime asset strategy:

- Assets are referenced through `lib/assets/manifest.ts`.
- Runtime files live in `public/campaign/`.
- Source pipeline is documented in `docs/asset-pipeline.md`.

Current image assets:

- `surface-threshold-wide-v001.png`
- `intro-descent-cathedral-v001.png`
- `hero-explorer-depth-plate-v001.png`
- `cards-reef-memory-v001.png`
- `cards-diver-lens-v001.png`
- `cards-sonar-abyss-v001.png`
- `cta-luminous-portal-v001.png`
- `lab-restored-reef-before-v001.png`
- `lab-restored-reef-after-v001.png`

Current asset concern:

- All runtime stills are PNG and large.
- Approximate PNG file sizes observed:
  - Cards: 2.3 MB to 2.7 MB each.
  - CTA: 3.0 MB.
  - Hero: 3.2 MB.
  - Intro: 2.7 MB.
  - Lab: 2.8 MB to 3.3 MB each.
  - Surface: 3.0 MB.
- Total campaign image payload is therefore too heavy for production if all images are loaded early.

Recommended next asset step:

- Convert final stills to AVIF/WebP.
- Add dimensions to the manifest.
- Keep hero/above-the-fold priority only where needed.
- Add mobile-specific crops for heavy plates.
- Keep a future GLB or Spline focal object to one controlled hero moment only.

## 9. Current Verification Status

The following checks have passed after the latest implementation passes:

- `npx.cmd tsc --noEmit`
- `npm.cmd run lint`
- `npm.cmd run build`

Build warning still present:

- Next/Turbopack infers the workspace root from `C:\Users\athar\package-lock.json` because it detects multiple lockfiles.
- The project also has `C:\Users\athar\OneDrive\Desktop\Projects\legendary-octo-waffle\clearwater-frontend\package-lock.json`.
- This does not currently fail the build, but it should be fixed before final handoff by setting `turbopack.root` in `next.config.ts` or removing the unintended higher-level lockfile if safe.

## 10. Known Bugs, Gaps, and Risks

### 10.1 Build Warning: Multiple Lockfiles

Severity: Low to medium.

Status:

- Build passes, but Next.js warns that it inferred the workspace root from a parent lockfile.

Risk:

- Turbopack may use the wrong root in some environments.
- Future build behavior could become confusing.

Recommended fix:

- Set `turbopack.root` in `next.config.ts`, or remove the parent `C:\Users\athar\package-lock.json` if it is not needed.

### 10.2 Image Payload Is Too Heavy

Severity: Medium to high for production.

Status:

- Runtime images are PNG files around 2.3 MB to 3.3 MB each.

Risk:

- Slow first load.
- Poor mobile performance.
- Higher memory use during scroll and transitions.

Recommended fix:

- Convert runtime stills to AVIF or WebP.
- Use smaller responsive exports.
- Keep PNG only where alpha is required.

### 10.3 Lab Upload Is Simulated

Severity: High if this is expected to be a real product demo.

Status:

- Uploading an image starts the fake processing state.
- The uploaded file is not displayed, sent to a backend, or restored.
- Result always shows static demo before/after images.

Risk:

- User may believe the model is running when it is only a prototype.

Recommended fix:

- Decide whether the lab is a demo-only story device or real app feature.
- If real, add backend/API integration and show the uploaded image in the result flow.
- If demo-only, label it clearly as "demo preview" or "sample run".

### 10.4 Download PNG Button Does Nothing

Severity: Medium.

Status:

- `BeforeAfterSlider.tsx` renders a `Download PNG` button without an `onClick` handler.

Risk:

- User-facing dead control.

Recommended fix:

- Either implement download behavior or hide the button until export is available.

### 10.5 Reduced Motion Is Not Implemented

Severity: Medium.

Status:

- There is no `prefers-reduced-motion` CSS or JS branch for the continuous GSAP loops, loader, particles, wave motion, or ScrollTrigger animation.

Risk:

- Accessibility issue for motion-sensitive users.
- Some devices may struggle with multiple continuous animations.

Recommended fix:

- Add reduced-motion detection.
- Skip or simplify loader, particles, wave morphing, and infinite loops when reduced motion is requested.

### 10.6 Hero Focal Object Is Hidden On Mobile

Severity: Medium for visual identity.

Status:

- `.hero-right` is hidden at `max-width: 900px`.
- This also hides the new restoration lens focal object.

Risk:

- Mobile users lose the key visual anchor.

Recommended fix:

- Add a simplified mobile version of the focal object.
- Keep it smaller and non-parallax to avoid layout and performance issues.

### 10.7 Slider Is Not Keyboard Accessible

Severity: Medium.

Status:

- The before/after slider supports mouse and touch drag.
- It does not expose keyboard controls or a semantic range input.

Risk:

- Accessibility gap.

Recommended fix:

- Add keyboard support or replace the divider interaction with an accessible range input styled as the divider.

### 10.8 Future Manifest Paths Are Placeholders

Severity: Low.

Status:

- The manifest references future video/GLB paths that are not currently present.
- They are safe while unused, but will fail if rendered before the files exist.

Recommended fix:

- Keep placeholders documented, but do not wire them into rendered components until the files exist.

### 10.9 No Real 3D Asset Yet

Severity: Low for prototype, medium for final campaign target.

Status:

- The hero focal object is procedural CSS/HTML, not a final Blender/Spline/GLB object.

Risk:

- The page has motion richness, but not yet a bespoke campaign art object at the same level as the target reference.

Recommended fix:

- Produce one custom hero object in Blender/Spline.
- Replace the procedural `restoration-core` while preserving the surrounding hero structure.

### 10.10 Some Scene Animations Still Live Locally

Severity: Low.

Status:

- The main scroll choreography is centralized in `CampaignTimeline.tsx`.
- Some local idle interactions remain in `HeroScene.tsx`, `CTAScene.tsx`, `OceanLoader.tsx`, and card hover handlers.

Risk:

- This is acceptable now, but can become hard to reason about if every scene starts adding its own scroll timelines again.

Recommended rule:

- Keep scroll-triggered story pacing in `CampaignTimeline.tsx`.
- Allow local components to own only idle loops, hover, pointer, and click feedback.

## 11. Product Decisions Made So Far

Important decisions:

- Keep the site short, not a long-scroll landing page.
- Use GSAP + CSS for animation rather than adding Framer Motion.
- Use one master scroll timeline for campaign pacing.
- Use local GSAP only for idle/micro interactions.
- Use procedural loader and wave material instead of heavy video for now.
- Use a typed asset manifest instead of hardcoded paths across scenes.
- Use a procedural hero focal object now, with a future path to real 3D.

## 12. Recommended Next Roadmap

### Phase 1: Clean Production Readiness

- Fix Turbopack root warning.
- Convert PNG assets to AVIF/WebP.
- Add reduced-motion support.
- Fix or remove the dead Download PNG button.
- Add keyboard accessibility for the before/after slider.

### Phase 2: Visual Identity Upgrade

- Create one final hero focal object in Blender or Spline.
- Replace procedural `restoration-core` with the final object.
- Add a mobile version of the focal object.
- Reduce generic AI-image feel by applying a consistent art direction pass to every plate.

### Phase 3: Real Lab Integration

- Decide if the lab is real or demo-only.
- If real, connect to the restoration backend.
- Display the uploaded image.
- Stream or simulate real processing status from the backend.
- Generate downloadable output.

### Phase 4: Motion Refinement

- Fine-tune the wave transition intensity after asset compression.
- Add a reduced-motion timeline branch.
- Profile animation performance on a mid-range mobile device.
- Consider one video loop only if procedural CSS cannot deliver the desired water material.

## 13. Acceptance Criteria For Current Prototype

Current prototype can be considered successful if:

- The site loads into a cinematic underwater experience.
- The loader hands off into the hero without a hard cut.
- Scroll feels smooth and short.
- Sections feel connected through waves, light, and depth.
- Hero has a clear focal object.
- Pipeline cards explain the system.
- Lab demonstrates the intended before/after story.
- Build, lint, and TypeScript checks pass.

Current prototype should not be considered production-complete until:

- Image payloads are optimized.
- Accessibility motion and slider issues are handled.
- The lab either becomes a real backend-backed feature or is clearly labeled as a demo.
- The mobile hero keeps a visual anchor.
- The Turbopack root warning is resolved.

