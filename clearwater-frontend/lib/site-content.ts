export type SiteStat = {
  value: string;
  label: string;
  detail: string;
};

export type DetailSection = {
  title: string;
  body: string;
  bullets: string[];
};

export type DetailPage = {
  slug: string;
  navLabel: string;
  eyebrow: string;
  title: string;
  intro: string;
  summary: string;
  stats: SiteStat[];
  sections: DetailSection[];
};

export type FeatureCallout = {
  label: string;
  detail: string;
};

export const heroSignals: FeatureCallout[] = [
  {
    label: 'Absorption',
    detail: 'Water strips red wavelengths first, pushing underwater scenes toward blue and green casts.',
  },
  {
    label: 'Scattering',
    detail: 'Suspended particles diffuse incoming light into haze, flattening contrast and erasing fine edges.',
  },
  {
    label: 'Low light',
    detail: 'Exponential attenuation at depth darkens the frame long before human or model analysis can begin.',
  },
];

export const homeStats: SiteStat[] = [
  {
    value: '7-Stage Pipeline',
    label: 'Physics-guided restoration',
    detail: 'From LAB chrominance correction to Guided Filter dehazing, every pixel is physically modeled.',
  },
  {
    value: '15-Class Detection',
    label: 'Marine species recovery',
    detail: 'Two-pass tiled YOLO inference recovers small marine species that standard models miss.',
  },
  {
    value: 'Neural Fusion',
    label: 'Adaptive arbitration',
    detail: 'A dynamic SS-UIE network arbitrates between classical physics and deep learning recovery.',
  },
];

export const pipelineSteps = [
  {
    step: '01',
    title: 'Raw Intake',
    body: 'Each run begins with the untouched underwater frame so the original signal remains scientifically faithful.',
  },
  {
    step: '02',
    title: 'Restoration',
    body: 'A seven-stage enhancement engine rebuilds color, contrast, and detail lost to the water column.',
  },
  {
    step: '03',
    title: 'Detection',
    body: 'Marine species are localized on the raw image through full-frame and tiled inference passes.',
  },
  {
    step: '04',
    title: 'Review',
    body: 'The system returns a side-by-side comparison image and per-object metrics for confident interpretation.',
  },
];

export const labFlow = [
  {
    title: 'Frame Intake',
    body: 'Raw underwater imagery enters the lab untouched so detection can preserve the original optical signal.',
  },
  {
    title: 'Enhancement Pass',
    body: 'LAB correction, CLAHE, neural fusion, guided dehazing, sharpening, denoising, and saturation recovery rebuild the scene.',
  },
  {
    title: 'Detection Report',
    body: 'Two-pass tiled YOLO inference, box validity filtering, and cross-class NMS produce the final marine species report.',
  },
];

export const labOutputs = [
  {
    title: 'Comparison Image',
    body: 'Every run is designed to resolve into a side-by-side raw and enhanced panel with detection overlays.',
  },
  {
    title: 'Species Metrics',
    body: 'Each object carries confidence, area, diameter, and circularity-style geometry so review remains quantitative.',
  },
  {
    title: 'Model Path',
    body: 'The restoration path can surface whether the frame was handled by SS-UIE, WaterNet, or the classical fallback stack.',
  },
];

export const detailPages: DetailPage[] = [
  {
    slug: 'architecture',
    navLabel: 'Architecture',
    eyebrow: 'System architecture',
    title: 'A two-stage computer vision system built for degraded underwater imagery.',
    intro:
      'Clearwater separates visual restoration from species detection so the image can be rebuilt for human review while the detector stays anchored to the untouched raw frame.',
    summary:
      'Stage 1 reconstructs clarity and color for display. Stage 2 detects marine life on the original signal and returns a comparison image plus structured metrics.',
    stats: [
      {
        value: '2 stages',
        label: 'Restoration + detection',
        detail: 'Enhancement is display-first, detection is raw-signal-first.',
      },
      {
        value: '7 passes',
        label: 'Enhancement depth',
        detail: 'The restoration engine chains seven classical and neural-aware recovery passes.',
      },
      {
        value: '15 classes',
        label: 'Species model',
        detail: 'The detector is trained on fifteen underwater species classes.',
      },
      {
        value: 'Compare output',
        label: 'Review surface',
        detail: 'Each run is framed as a side-by-side output image with object annotations.',
      },
    ],
    sections: [
      {
        title: 'Stage 1: Restoration for visual recovery',
        body:
          'The first stage rebuilds underwater frames for human interpretation, compensating for color cast, haze, and depth-driven darkness before the image is presented for review.',
        bullets: [
          'LAB chrominance correction neutralizes green and blue cast drift.',
          'CLAHE, adaptive gamma, guided dehazing, sharpening, and denoising restore usable local contrast.',
          'The final enhanced frame is written to the comparison surface rather than used for detection.',
        ],
      },
      {
        title: 'Stage 2: Detection on the untouched signal',
        body:
          'Clearwater keeps detection on the raw image to avoid enhancement artifacts biasing the location or class of the predicted boxes.',
        bullets: [
          'Full-frame YOLO inference captures large and medium objects across the scene.',
          'A second tiled pass restores recall for small marine species lost by standard resizing.',
          'Per-class and cross-class NMS resolve duplicate and generic-versus-specific detections.',
        ],
      },
      {
        title: 'The review artifact',
        body:
          'The final product surface is not just an image. It is a structured interpretive layer for both visual inspection and numeric analysis.',
        bullets: [
          'The output image presents raw and enhanced panels side by side.',
          'Detection overlays can be compared visually against both views.',
          'Each object is paired with confidence and geometry metrics for downstream review.',
        ],
      },
    ],
  },
  {
    slug: 'enhancement',
    navLabel: 'Enhancement',
    eyebrow: 'Enhancement engine',
    title: 'Seven restoration stages counter the three physics failures of underwater imaging.',
    intro:
      'Underwater scenes degrade through wavelength absorption, suspended-particle scattering, and severe illumination loss. Clearwater treats each of those failures as an explicit restoration target.',
    summary:
      'The enhancement engine combines cast-aware classical processing with optional neural fusion so the final frame feels recovered rather than cosmetically filtered.',
    stats: [
      {
        value: '3 degradations',
        label: 'Physics targets',
        detail: 'Absorption, scattering, and low illumination define the restoration problem.',
      },
      {
        value: '7 stages',
        label: 'Recovery sequence',
        detail: 'Each stage restores a distinct failure mode before the final frame is produced.',
      },
      {
        value: '60/40',
        label: 'Neural blend',
        detail: 'When available, neural output is fused back into the LAB-corrected base.',
      },
      {
        value: '25% guard',
        label: 'Safety threshold',
        detail: 'If the restored frame drops too far below the source brightness, the original is preserved.',
      },
    ],
    sections: [
      {
        title: 'Absorption and color cast',
        body:
          'Water attenuates red wavelengths first, then progressively collapses the warm end of the spectrum. Clearwater measures cast relative to the red channel and uses that signal to drive correction strength.',
        bullets: [
          'LAB chrominance correction shifts the a* and b* channels back toward neutral.',
          'Cast strength scales the amount of correction so green-heavy and blue-heavy scenes are treated differently.',
          'Final saturation restore returns chroma after the frame regains structural clarity.',
        ],
      },
      {
        title: 'Scattering and haze',
        body:
          'Suspended particles scatter light into a veil that erases local edges and compresses contrast. Clearwater addresses that by restoring local separation before polishing detail.',
        bullets: [
          'CLAHE on the luminance channel reopens local contrast without blowing out highlights.',
          'Guided Filter dehazing removes haze while respecting image boundaries.',
          'Unsharp masking and bilateral denoising recover edge presence while suppressing artifact growth.',
        ],
      },
      {
        title: 'Low illumination and tonal recovery',
        body:
          'Depth-driven darkness is treated as more than exposure loss. The pipeline uses adaptive gamma and recovery blending so dim scenes regain readability without looking synthetic.',
        bullets: [
          'Adaptive gamma brightens dark frames more aggressively than bright ones.',
          'Multi-Scale Retinex provides a classical fallback when neural enhancement is unavailable.',
          'The final safety guard prevents over-processing from returning a frame darker than the source.',
        ],
      },
      {
        title: 'Neural fusion in the recovery stack',
        body:
          'When SS-UIE or WaterNet is available, Clearwater does not simply replace the classical frame. It fuses neural recovery with the corrected base so the output remains grounded in physically plausible structure.',
        bullets: [
          'SS-UIE and WaterNet occupy the neural slot in the restoration sequence.',
          'The neural result is blended into the LAB-corrected base rather than trusted blindly.',
          'If no neural model is present, the classical stack continues through the same review surface.',
        ],
      },
    ],
  },
  {
    slug: 'detection',
    navLabel: 'Detection',
    eyebrow: 'Detection engine',
    title: 'Two-pass tiled YOLO inference preserves small-species recall in hostile underwater scenes.',
    intro:
      'A single resized inference pass is not enough for dense underwater imagery. Clearwater pairs full-frame context with overlapping tile scans so smaller marine species survive the trip through the detector.',
    summary:
      'The detector is built around tiled recovery, geometric box validation, and cross-class suppression that resolves the classic double-box artifact.',
    stats: [
      {
        value: '640 px',
        label: 'Inference size',
        detail: 'The full frame and every tile are normalized to a 640 pixel YOLO input.',
      },
      {
        value: '20% overlap',
        label: 'Tile stride',
        detail: 'Overlapping tiles prevent edge loss and recover objects split across tile boundaries.',
      },
      {
        value: '0.15 conf',
        label: 'Recall bias',
        detail: 'The detector intentionally accepts low-confidence candidates before geometric cleanup.',
      },
      {
        value: '2x NMS',
        label: 'Artifact control',
        detail: 'Per-class and cross-class suppression collapse duplicates and class conflicts.',
      },
    ],
    sections: [
      {
        title: 'The two-pass tiled strategy',
        body:
          'Clearwater first reads the entire image for scene context, then sweeps overlapping tiles across the frame to recover smaller targets that disappear after standard resizing.',
        bullets: [
          'Full-frame inference is responsible for large and medium marine objects.',
          'Tiled inference recovers fish that would otherwise compress below the detector threshold.',
          'Tile coordinates are remapped to the original image so the final boxes stay frame-accurate.',
        ],
      },
      {
        title: 'Validity filtering before suppression',
        body:
          'Not every low-confidence box deserves a second life. Clearwater rejects geometrically implausible candidates before suppression begins.',
        bullets: [
          'Boxes smaller than the minimum area threshold are discarded as noise.',
          'Detections that occupy too much of the frame are rejected as scene-level false positives.',
          'Extreme aspect ratios are removed before they can contaminate final species output.',
        ],
      },
      {
        title: 'Cross-class NMS and the double-box artifact',
        body:
          'Underwater detectors often fire both a generic fish label and a species-specific label on the same animal. Clearwater resolves that conflict explicitly instead of trusting score order alone.',
        bullets: [
          'Per-class NMS collapses duplicates generated by overlapping tiles.',
          'Cross-class NMS compares generic and specific labels over the same object footprint.',
          'When a specific species label overlaps a generic fish box, the specific label wins and the double-box artifact disappears.',
        ],
      },
      {
        title: 'Structured review metrics',
        body:
          'Detection output is designed to be inspected, not just rendered. Each object ships with interpretable numbers that support review and downstream filtering.',
        bullets: [
          'Confidence scores quantify detector certainty for every predicted object.',
          'Area and diameter provide an immediate read on object scale inside the frame.',
          'Circularity-style geometry helps distinguish compact objects from elongated false positives.',
        ],
      },
    ],
  },
];

export function getDetailPage(slug: string) {
  return detailPages.find((page) => page.slug === slug);
}
