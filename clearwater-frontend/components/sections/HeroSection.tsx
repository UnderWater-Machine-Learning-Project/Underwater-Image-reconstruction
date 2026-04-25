'use client';

import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { CustomEase } from 'gsap/CustomEase';
import { CustomWiggle } from 'gsap/CustomWiggle';
import { Physics2DPlugin } from 'gsap/Physics2DPlugin';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(ScrollTrigger, CustomEase, CustomWiggle, Physics2DPlugin);
  CustomWiggle.create('hero-heading-wiggle', { wiggles: 8, type: 'anticipate' });
}

const PARTICLE_COUNT = 55;
const TITLE_FLARE_COUNT = 12;
const titleText = 'CLEARWATER';
const titleText2 = 'VISION';
const flairParticles = Array.from({ length: TITLE_FLARE_COUNT }, (_, index) => ({
  size: 6 + (index % 3) * 4,
  color: index % 4 === 0 ? 'rgba(245,241,233,0.95)' : 'rgba(124,214,204,0.9)',
}));

export default function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const primaryLineRef = useRef<HTMLSpanElement>(null);
  const taglineRef = useRef<HTMLParagraphElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const onResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', onResize);

    const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: 0.4 + Math.random() * 1.6,
      vy: -(0.08 + Math.random() * 0.18),
      vx: (Math.random() - 0.5) * 0.06,
      a: Math.random(),
      da: 0.002 + Math.random() * 0.004,
      fade: Math.random() > 0.5 ? 1 : -1,
    }));

    let raf = 0;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      particles.forEach((particle) => {
        particle.a += particle.da * particle.fade;

        if (particle.a >= 1 || particle.a <= 0) {
          particle.fade *= -1;
        }

        particle.x += particle.vx;
        particle.y += particle.vy;

        if (particle.y < -4) particle.y = height + 4;
        if (particle.x < -4) particle.x = width + 4;
        if (particle.x > width + 4) particle.x = -4;

        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(124, 214, 204, ${particle.a * 0.52})`;
        ctx.fill();
      });

      raf = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
      const primaryLine = primaryLineRef.current;
      const flairItems = gsap.utils.toArray<HTMLElement>('.hero-flair', sectionRef.current);

      tl.to(
        overlayRef.current,
        {
          scaleY: 0,
          transformOrigin: 'top center',
          duration: 1.4,
          ease: 'expo.inOut',
        },
        0,
      );

      tl.fromTo(
        '.hero-ray',
        { scaleY: 0, opacity: 0, transformOrigin: 'top center' },
        { scaleY: 1, opacity: 1, duration: 2.2, stagger: 0.12, ease: 'expo.out' },
        0.3,
      );

      tl.fromTo(
        '.hero-char',
        { y: 80, opacity: 0, filter: 'blur(8px)' },
        {
          y: 0,
          opacity: 1,
          filter: 'blur(0px)',
          duration: 1.1,
          stagger: 0.04,
          ease: 'back.out(1.4)',
        },
        0.8,
      );

      if (primaryLine) {
        tl.to(
          primaryLine,
          {
            duration: 1.15,
            transformOrigin: 'center center',
            rotation: 6,
            x: 8,
            y: 8,
            scale: 0.92,
            ease: 'hero-heading-wiggle',
          },
          1.02,
        ).to(
          primaryLine,
          {
            duration: 0.8,
            rotation: 0,
            x: 0,
            y: 0,
            scale: 1,
            ease: 'power3.out',
          },
          1.94,
        );
      }

      if (flairItems.length > 0) {
        gsap.set(flairItems, {
          opacity: 1,
          scale: 0,
          x: 0,
          y: 0,
        });

        tl.to(
          flairItems,
          {
            scale: 1,
            duration: 0.16,
            stagger: 0.016,
            ease: 'back.out(2)',
          },
          1.08,
        );

        flairItems.forEach((item, index) => {
          const angle = (index / flairItems.length) * 360 + gsap.utils.random(-18, 18);
          const velocity = gsap.utils.random(360, 700);

          tl.to(
            item,
            {
              physics2D: {
                angle,
                velocity,
                gravity: 760,
              },
              rotation: gsap.utils.random(-180, 180),
              duration: 2.8,
            },
            1.08,
          ).to(
            item,
            {
              opacity: 0,
              scale: gsap.utils.random(0.55, 0.95),
              duration: 0.25,
              ease: 'power1.out',
            },
            2.35,
          );
        });
      }

      tl.fromTo(
        taglineRef.current,
        { y: 24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.9 },
        1.6,
      );

      tl.fromTo(
        '.hero-cta',
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.7, stagger: 0.1 },
        1.9,
      );

      gsap.to('.hero-depth-layer', {
        y: (index) => (index + 1) * 120,
        ease: 'none',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top top',
          end: 'bottom top',
          scrub: 1.2,
        },
      });

      gsap.to(sectionRef.current, {
        opacity: 0,
        scale: 0.96,
        ease: 'none',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: '60% top',
          end: 'bottom top',
          scrub: true,
        },
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="hero"
      className="relative h-screen w-full overflow-hidden bg-transparent"
    >
      <div ref={overlayRef} className="absolute inset-0 z-30 bg-[#041016]" />

      <canvas
        ref={canvasRef}
        className="hero-depth-layer pointer-events-none absolute inset-0 z-[1]"
      />

      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[rgba(4,13,18,0.92)] via-[rgba(8,20,24,0.72)] to-[rgba(3,10,14,0.28)]" />

      <div
        className="pointer-events-none absolute inset-0 z-[2] opacity-30"
        style={{
          background:
            'radial-gradient(ellipse 82% 62% at 50% 0%, rgba(124,214,204,0.16) 0%, transparent 70%), ' +
            'radial-gradient(ellipse 36% 28% at 32% 22%, rgba(213,187,139,0.1) 0%, transparent 58%)',
        }}
      />

      <div className="hero-depth-layer pointer-events-none absolute inset-0 z-[2] overflow-hidden">
        {Array.from({ length: 8 }).map((_, index) => (
          <div
            key={index}
            className="hero-ray absolute top-0 h-[70vh]"
            style={{
              left: `${10 + index * 11}%`,
              width: `${1.5 + (index % 3) * 0.8}%`,
              transform: `rotate(${-12 + index * 3.5}deg) skewX(${index % 2 === 0 ? -3 : 2}deg)`,
              background: `linear-gradient(to bottom, rgba(124,214,204,${0.065 - index * 0.004}) 0%, transparent 100%)`,
              opacity: 0,
            }}
          />
        ))}
      </div>

      <div className="absolute bottom-0 left-0 right-0 z-[3] h-px bg-gradient-to-r from-transparent via-[rgba(124,214,204,0.22)] to-transparent" />

      <div className="relative z-10 flex h-full flex-col items-center justify-center px-6 text-center">
        <p className="hero-cta mb-8 font-mono text-[10px] uppercase tracking-[0.4em] text-[rgba(213,187,139,0.72)] opacity-0">
          Clearwater Vision System · v2.4
        </p>

        <h1
          className="flex flex-col items-center gap-1 leading-none"
          aria-label="Clearwater Vision"
        >
          <span className="relative block">
            <span className="pointer-events-none absolute inset-0 z-20">
              {flairParticles.map((particle, index) => (
                <span
                  key={index}
                  className="hero-flair absolute left-1/2 top-1/2 block -translate-x-1/2 -translate-y-1/2 rounded-full opacity-0"
                  style={{
                    width: `${particle.size}px`,
                    height: `${particle.size}px`,
                    background: particle.color,
                    boxShadow:
                      particle.color === 'rgba(245,241,233,0.95)'
                        ? '0 0 18px rgba(245,241,233,0.75)'
                        : '0 0 24px rgba(124,214,204,0.68)',
                  }}
                />
              ))}
            </span>
            <span ref={primaryLineRef} className="block overflow-hidden">
              <span className="flex">
                {titleText.split('').map((character, index) => (
                  <span
                    key={index}
                    className="hero-char inline-block opacity-0"
                    style={{
                      fontFamily: "'Playfair Display', Georgia, serif",
                      fontSize: 'clamp(4rem, 12vw, 9rem)',
                      fontWeight: 700,
                      letterSpacing: '-0.02em',
                      color: '#fff',
                    }}
                  >
                    {character}
                  </span>
                ))}
              </span>
            </span>
          </span>

          <span className="block overflow-hidden">
            <span className="flex">
              {titleText2.split('').map((character, index) => (
                <span
                  key={index}
                  className="hero-char inline-block opacity-0"
                  style={{
                    fontFamily: "'Playfair Display', Georgia, serif",
                    fontSize: 'clamp(4rem, 12vw, 9rem)',
                    fontWeight: 400,
                    fontStyle: 'italic',
                    letterSpacing: '-0.02em',
                    color: '#8edfd2',
                    textShadow: '0 0 56px rgba(124,214,204,0.28)',
                  }}
                >
                  {character}
                </span>
              ))}
            </span>
          </span>
        </h1>

        <p
          ref={taglineRef}
          className="mt-8 max-w-md font-mono text-xs leading-relaxed tracking-widest text-slate-400 opacity-0"
        >
          Underwater image enhancement & object detection.
          <br />
          Deep-sea clarity through neural intelligence.
        </p>

        <div className="hero-cta mt-12 flex items-center gap-4 opacity-0">
          <a
            href="#upload"
            className="group relative overflow-hidden rounded-full border px-8 py-3 font-mono text-xs uppercase tracking-[0.2em] backdrop-blur-sm transition-all duration-300"
            style={{
              borderColor: 'rgba(124,214,204,0.36)',
              background: 'rgba(124,214,204,0.1)',
              color: '#b8efe2',
            }}
          >
            <span className="relative z-10">Dive In</span>
            <span
              className="absolute inset-0 -translate-x-full transition-transform duration-500 group-hover:translate-x-full"
              style={{
                background:
                  'linear-gradient(to right, transparent, rgba(213,187,139,0.12), transparent)',
              }}
            />
          </a>
          <a
            href="#team"
            className="font-mono text-xs uppercase tracking-[0.2em] text-slate-500 transition-colors hover:text-slate-300"
          >
            Meet the Team ↓
          </a>
        </div>
      </div>

      <div className="absolute bottom-10 left-1/2 z-10 flex -translate-x-1/2 flex-col items-center gap-3 opacity-60">
        <span className="font-mono text-[9px] uppercase tracking-[0.35em] text-slate-500">
          Scroll
        </span>
        <div className="h-10 w-px overflow-hidden bg-slate-800">
          <div className="h-full w-full animate-[scrollLine_1.8s_ease-in-out_infinite] bg-gradient-to-b from-transparent via-[#8edfd2] to-transparent" />
        </div>
      </div>
    </section>
  );
}
