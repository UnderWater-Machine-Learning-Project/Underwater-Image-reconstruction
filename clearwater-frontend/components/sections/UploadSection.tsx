'use client';

import Image from 'next/image';
import { useCallback, useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { CustomEase } from 'gsap/CustomEase';
import { CustomWiggle } from 'gsap/CustomWiggle';
import { Physics2DPlugin } from 'gsap/Physics2DPlugin';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger, CustomEase, CustomWiggle, Physics2DPlugin);
  CustomWiggle.create('rejectWiggle', { wiggles: 7, type: 'anticipate' });
  CustomWiggle.create('successWiggle', { wiggles: 3, type: 'easeOut' });
}

const PARTICLE_COUNT = 24;
const PRIMARY_ACCENT = '#8edfd2';
const SECONDARY_ACCENT = '#d5bb8b';

type UploadState = 'idle' | 'dragging' | 'loaded' | 'processing' | 'done';

export default function UploadSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const dropzoneRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const checkRef = useRef<SVGSVGElement>(null);
  const particlePoolRef = useRef<HTMLDivElement>(null);

  const [state, setState] = useState<UploadState>('idle');
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<'enhance' | 'detect' | null>(null);
  const dragCounter = useRef(0);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        '.upload-reveal',
        { y: 48, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.9,
          stagger: 0.12,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top 75%',
            once: true,
          },
        },
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const fireCompletionBurst = useCallback(() => {
    const dropzone = dropzoneRef.current;
    const pool = particlePoolRef.current;

    if (!dropzone || !pool) {
      return;
    }

    const { left, top, width, height } = dropzone.getBoundingClientRect();
    const originX = left + width / 2;
    const originY = top + height / 2;

    for (let index = 0; index < PARTICLE_COUNT; index += 1) {
      const size = gsap.utils.random(3, 9);
      const isRing = Math.random() > 0.5;
      const particle = document.createElement('div');
      const accent =
        Math.random() > 0.3
          ? `rgba(142,223,210,${gsap.utils.random(0.6, 0.9).toFixed(2)})`
          : `rgba(213,187,139,${gsap.utils.random(0.45, 0.72).toFixed(2)})`;

      Object.assign(particle.style, {
        position: 'fixed',
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        background: isRing ? 'transparent' : accent,
        border: isRing ? `1px solid ${accent}` : 'none',
        left: `${originX}px`,
        top: `${originY}px`,
        transform: 'translate(-50%,-50%)',
        opacity: '0',
        pointerEvents: 'none',
        zIndex: '200',
      });

      pool.appendChild(particle);

      gsap
        .timeline({ onComplete: () => particle.remove() })
        .set(particle, { opacity: 1 })
        .to(
          particle,
          {
            physics2D: {
              angle: Math.random() * 360,
              velocity: gsap.utils.random(200, 600),
              gravity: 400,
            },
            rotation: gsap.utils.random(-180, 180),
            duration: gsap.utils.random(1.2, 2.2),
          },
          0,
        )
        .to(
          particle,
          {
            opacity: 0,
            scale: gsap.utils.random(0.2, 0.6),
            duration: 0.35,
            ease: 'power1.out',
          },
          gsap.utils.random(0.6, 1.1),
        );
    }
  }, []);

  const shakeReject = useCallback(() => {
    const dropzone = dropzoneRef.current;

    if (!dropzone) {
      return;
    }

    gsap.fromTo(
      dropzone,
      { x: 0 },
      { x: 10, duration: 0.7, ease: 'rejectWiggle', clearProps: 'x' },
    );

    gsap.to(dropzone, {
      outlineColor: 'rgba(239,68,68,0.5)',
      outlineWidth: '2px',
      outlineStyle: 'solid',
      duration: 0.1,
      yoyo: true,
      repeat: 3,
      clearProps: 'outline',
    });
  }, []);

  const bounceCheck = useCallback(() => {
    const check = checkRef.current;

    if (!check) {
      return;
    }

    gsap.fromTo(
      check,
      { scale: 0, rotation: -15, opacity: 1 },
      {
        scale: 1,
        rotation: 0,
        duration: 0.55,
        ease: 'successWiggle',
        transformOrigin: '50% 50%',
      },
    );
  }, []);

  const runProcessing = useCallback(
    (selectedMode: 'enhance' | 'detect') => {
      setMode(selectedMode);
      setState('processing');

      const pulseTween = gsap.to(dropzoneRef.current, {
        boxShadow: '0 0 0 3px rgba(142,223,210,0.28), 0 0 50px rgba(142,223,210,0.1)',
        duration: 0.9,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });

      window.setTimeout(() => {
        pulseTween.kill();
        gsap.to(dropzoneRef.current, { boxShadow: 'none', duration: 0.3 });
        setState('done');
        fireCompletionBurst();

        gsap.fromTo(
          resultsRef.current,
          { opacity: 0, y: 28, filter: 'blur(6px)' },
          {
            opacity: 1,
            y: 0,
            filter: 'blur(0px)',
            duration: 0.85,
            ease: 'power3.out',
            delay: 0.15,
          },
        );

        window.setTimeout(bounceCheck, 250);
      }, 2600);
    },
    [bounceCheck, fireCompletionBurst],
  );

  const onDragEnter = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    dragCounter.current += 1;

    if (dragCounter.current === 1) {
      setState('dragging');
    }
  }, []);

  const onDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    dragCounter.current -= 1;

    if (dragCounter.current === 0) {
      setState((previous) => (previous === 'dragging' ? 'idle' : previous));
    }
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      dragCounter.current = 0;

      const file = event.dataTransfer.files[0];

      if (!file || !file.type.startsWith('image/')) {
        shakeReject();
        return;
      }

      if (preview) {
        URL.revokeObjectURL(preview);
      }

      setPreview(URL.createObjectURL(file));
      setState('loaded');

      gsap.fromTo(
        dropzoneRef.current,
        { scale: 0.97 },
        { scale: 1, duration: 0.5, ease: 'back.out(1.6)' },
      );
    },
    [preview, shakeReject],
  );

  const onFileInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];

      if (!file) {
        return;
      }

      if (preview) {
        URL.revokeObjectURL(preview);
      }

      setPreview(URL.createObjectURL(file));
      setState('loaded');
    },
    [preview],
  );

  const reset = useCallback(() => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setPreview(null);
    setState('idle');
    setMode(null);
    gsap.to(resultsRef.current, { opacity: 0, y: 16, duration: 0.3, ease: 'power2.in' });
  }, [preview]);

  const isDragging = state === 'dragging';
  const isLoaded = ['loaded', 'processing', 'done'].includes(state);
  const isProcessing = state === 'processing';
  const isDone = state === 'done';

  return (
    <section ref={sectionRef} id="upload" className="relative w-full bg-transparent py-32">
      <div
        ref={particlePoolRef}
        className="pointer-events-none fixed inset-0 z-[200] overflow-hidden"
        aria-hidden="true"
      />

      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute left-1/2 top-1/2 h-[500px] w-[700px] -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl"
          style={{ background: 'rgba(124,214,204,0.04)' }}
        />
      </div>

      <div className="relative z-10 mx-auto max-w-4xl px-6">
        <div className="upload-reveal mb-16 opacity-0">
          <p className="mb-3 font-mono text-[10px] uppercase tracking-[0.35em] text-[rgba(213,187,139,0.72)]">
            Processing Engine
          </p>
          <h2
            className="text-4xl font-light text-white md:text-5xl"
            style={{ fontFamily: 'var(--font-display, "Playfair Display", Georgia, serif)' }}
          >
            Drop a frame.
            <br />
            <em className="text-[#8edfd2]">Watch it transform.</em>
          </h2>
          <p className="mt-4 font-mono text-xs text-slate-600">
            PNG · JPG · WEBP - invalid types trigger reject animation
          </p>
        </div>

        <div
          ref={dropzoneRef}
          className={`upload-reveal relative overflow-hidden rounded-3xl border opacity-0 transition-all duration-300 ${
            isDragging
              ? 'border-[rgba(142,223,210,0.6)] bg-[rgba(142,223,210,0.07)] shadow-[0_0_40px_rgba(142,223,210,0.12)]'
              : isLoaded
                ? 'border-[rgba(142,223,210,0.22)] bg-[rgba(9,22,28,0.8)]'
                : 'border-dashed border-white/10 bg-[rgba(9,22,28,0.62)] hover:border-white/20'
          }`}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          style={{ minHeight: 280 }}
        >
          {!isLoaded ? (
            <label className="flex h-full min-h-[280px] cursor-pointer flex-col items-center justify-center gap-5 p-10">
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                className="sr-only"
                onChange={onFileInput}
              />
              <div
                className={`flex h-14 w-14 items-center justify-center rounded-full border transition-all duration-300 ${
                  isDragging ? 'border-[rgba(142,223,210,0.6)] bg-[rgba(142,223,210,0.14)]' : 'border-white/15 bg-white/5'
                }`}
              >
                <svg
                  width="22"
                  height="22"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={isDragging ? PRIMARY_ACCENT : '#64748b'}
                  strokeWidth="1.5"
                  aria-hidden="true"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <div className="text-center">
                <p
                  className={`font-mono text-sm uppercase tracking-widest transition-colors ${
                    isDragging ? 'text-[#8edfd2]' : 'text-slate-400'
                  }`}
                >
                  {isDragging ? 'Release to upload' : 'Drop degraded frame here'}
                </p>
                <p className="mt-1.5 font-mono text-[10px] uppercase tracking-widest text-slate-600">
                  PNG · JPG · WEBP
                </p>
              </div>
            </label>
          ) : (
            <div className="flex min-h-[280px] items-center gap-6 p-8">
              {preview && (
                <div className="relative h-48 w-64 flex-shrink-0 overflow-hidden rounded-xl border border-white/10">
                  <Image
                    src={preview}
                    alt="Uploaded frame preview"
                    fill
                    unoptimized
                    sizes="256px"
                    className="object-cover"
                  />
                  {isProcessing && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60 backdrop-blur-sm">
                      <div className="h-8 w-8 animate-spin rounded-full border-2 border-transparent border-t-[#8edfd2]" />
                      <span className="font-mono text-[9px] uppercase tracking-widest text-[rgba(142,223,210,0.7)]">
                        {mode === 'enhance' ? 'Enhancing...' : 'Detecting...'}
                      </span>
                    </div>
                  )}
                  {isDone && (
                    <div className="absolute inset-0 flex items-center justify-center bg-[rgba(142,223,210,0.08)]">
                      <svg
                        ref={checkRef}
                        width="36"
                        height="36"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke={PRIMARY_ACCENT}
                        strokeWidth="2"
                        style={{ opacity: 0 }}
                        aria-label="Complete"
                      >
                        <path d="M20 6 9 17l-5-5" />
                      </svg>
                    </div>
                  )}
                </div>
              )}
              <div className="flex-1">
                <p className="font-mono text-xs text-slate-400">
                  {isProcessing
                    ? `Running ${mode === 'enhance' ? 'neural enhancement' : 'object detection'}…`
                    : isDone
                      ? `${mode === 'enhance' ? 'Enhancement' : 'Detection'} complete`
                      : 'Frame loaded - select an operation below'}
                </p>
                {isDone && (
                  <button
                    onClick={reset}
                    className="mt-4 font-mono text-[10px] uppercase tracking-widest text-slate-600 underline hover:text-slate-400"
                  >
                    Upload another
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {(state === 'loaded' || isProcessing) && (
          <div className="upload-reveal mt-8 flex items-center justify-center gap-4 opacity-100">
            <MagBtn
              accent={PRIMARY_ACCENT}
              label="Run Enhancement"
              loading={isProcessing && mode === 'enhance'}
              disabled={isProcessing}
              onClick={() => runProcessing('enhance')}
            />
            <MagBtn
              accent={SECONDARY_ACCENT}
              label="Run Detection"
              border
              loading={isProcessing && mode === 'detect'}
              disabled={isProcessing}
              onClick={() => runProcessing('detect')}
            />
          </div>
        )}

        <div ref={resultsRef} className="mt-12 opacity-0">
          {isDone && (
            <div className="overflow-hidden rounded-3xl border border-white/8 bg-[rgba(9,22,28,0.82)] p-1">
              <div className="flex min-h-[300px] items-center justify-center rounded-2xl bg-[rgba(3,10,14,0.9)]">
                <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-slate-600">
                  Comparison slider placeholder
                </p>
              </div>
              {mode === 'detect' && (
                <div className="flex gap-2 p-4">
                  {['Marine Life', 'Coral', 'Debris'].map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-[rgba(142,223,210,0.2)] bg-[rgba(142,223,210,0.06)] px-3 py-1 font-mono text-[9px] uppercase tracking-wider text-[rgba(142,223,210,0.72)]"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function MagBtn({
  label,
  accent,
  border = false,
  loading = false,
  disabled = false,
  onClick,
}: {
  label: string;
  accent: string;
  border?: boolean;
  loading?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  const zoneRef = useRef<HTMLSpanElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const zone = zoneRef.current;

    if (!zone) {
      return;
    }

    const onMove = (event: PointerEvent) => {
      const bounds = zone.getBoundingClientRect();
      const x = gsap.utils.mapRange(
        bounds.left,
        bounds.right,
        -bounds.width / 2,
        bounds.width / 2,
        event.clientX,
      );
      const y = gsap.utils.mapRange(
        bounds.top,
        bounds.bottom,
        -bounds.height / 2,
        bounds.height / 2,
        event.clientY,
      );

      gsap.to(buttonRef.current, {
        x: x * 0.38,
        y: y * 0.38,
        duration: 0.28,
        ease: 'power3.out',
        overwrite: 'auto',
      });

      gsap.to(labelRef.current, {
        x: x * 0.15,
        y: y * 0.15,
        duration: 0.28,
        ease: 'power3.out',
        overwrite: 'auto',
      });
    };

    const onLeave = () => {
      gsap.to([buttonRef.current, labelRef.current], {
        x: 0,
        y: 0,
        duration: 0.6,
        ease: 'elastic.out(1,0.45)',
        overwrite: 'auto',
      });
    };

    zone.addEventListener('pointermove', onMove);
    zone.addEventListener('pointerleave', onLeave);

    return () => {
      zone.removeEventListener('pointermove', onMove);
      zone.removeEventListener('pointerleave', onLeave);
    };
  }, []);

  return (
    <span ref={zoneRef} className="inline-block cursor-pointer p-4">
      <button
        ref={buttonRef}
        disabled={disabled}
        onClick={onClick}
        className={`relative overflow-hidden rounded-full px-8 py-3.5 font-mono text-xs uppercase tracking-[0.2em] transition-opacity ${
          disabled ? 'opacity-50' : ''
        }`}
        style={
          border
            ? {
                background: 'transparent',
                border: `1px solid ${accent}40`,
                color: accent,
              }
            : {
                background: accent,
                color: '#071219',
                fontWeight: 600,
              }
        }
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <span className="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent" />
            Processing...
          </span>
        ) : (
          <span ref={labelRef}>{label}</span>
        )}
      </button>
    </span>
  );
}
