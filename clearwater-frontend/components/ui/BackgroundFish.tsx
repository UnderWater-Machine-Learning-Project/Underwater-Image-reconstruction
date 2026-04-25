'use client';

import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { CustomEase } from 'gsap/CustomEase';
import { CustomWiggle } from 'gsap/CustomWiggle';
import { MotionPathPlugin } from 'gsap/MotionPathPlugin';
import { Physics2DPlugin } from 'gsap/Physics2DPlugin';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

if (typeof window !== 'undefined') {
  gsap.registerPlugin(
    ScrollTrigger,
    MotionPathPlugin,
    CustomEase,
    CustomWiggle,
    Physics2DPlugin,
  );
  CustomWiggle.create('tailWiggle', { wiggles: 7, type: 'easeOut' });
  CustomWiggle.create('finWiggle', { wiggles: 4, type: 'uniform' });
}

const BUBBLE_COUNT = 8;

export default function BackgroundFish() {
  const containerRef = useRef<HTMLDivElement>(null);
  const bubblePoolRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const bubblePool = bubblePoolRef.current;

    if (!container || !bubblePool) {
      return;
    }

    const ctx = gsap.context(() => {
      const select = gsap.utils.selector(container);
      const fish = select('.fish')[0] as HTMLElement | undefined;
      const fishHeadAndBody = gsap.utils.toArray<HTMLElement>(
        '.fish__head, .fish__body',
        container,
      );
      const fishSkeleton = select('.fish__skeleton')[0] as HTMLElement | undefined;
      const fishInner = select('.fish__inner')[0] as HTMLElement | undefined;
      const tailFork = select('.fish__tail-fork')[0] as HTMLElement | undefined;
      const finPrimary = select('.fish__fin:not(.fish__fin--2)')[0] as HTMLElement | undefined;
      const finSecondary = select('.fish__fin--2')[0] as HTMLElement | undefined;
      const lights = gsap.utils.toArray<HTMLElement>('.lights__light', container);

      if (!fish) {
        return;
      }

      const width = window.innerWidth;
      const height = window.innerHeight;

      const path = [
        { x: width * 0.74, y: height * 0.18 },
        { x: width * 0.66, y: height * 0.12 },
        { x: width * 0.56, y: height * 0.2 },
        { x: width * 0.44, y: height * 0.36 },
        { x: width * 0.3, y: height * 0.54 },
        { x: width * 0.16, y: height * 0.66 },
        { x: width * 0.12, y: height * 0.76 },
        { x: width * 0.26, y: height * 0.72 },
        { x: width * 0.44, y: height * 0.58 },
        { x: width * 0.62, y: height * 0.42 },
        { x: width * 0.74, y: height * 0.3 },
      ];

      const timeline = gsap.timeline({
        scrollTrigger: {
          trigger: document.documentElement,
          start: 'top top',
          end: 'bottom bottom',
          scrub: 1.5,
        },
      });

      gsap.set(fish, {
        opacity: 0.94,
        scale: 1.04,
        rotateZ: -8,
        rotateY: 0,
        rotateX: 0,
      });

      timeline.to(
        fish,
        {
          motionPath: {
            path,
            autoRotate: true,
          },
          duration: 10,
          ease: 'none',
          immediateRender: true,
        },
        0,
      );

      timeline
        .to(fish, { scale: 1.12, rotateZ: 4, duration: 2.1, ease: 'sine.inOut' }, 0.2)
        .to(fish, { scale: 0.98, rotateZ: -10, duration: 2.4, ease: 'sine.inOut' }, 2.8)
        .to(fish, { scale: 1.08, rotateZ: 6, duration: 2.1, ease: 'sine.inOut' }, 5.3)
        .to(fish, { scale: 1.02, rotateZ: -4, duration: 2.2, ease: 'sine.inOut' }, 7.5);

      if (fishSkeleton) {
        gsap.to(fishSkeleton, {
          opacity: 0.28,
          duration: 4.2,
          repeat: -1,
          yoyo: true,
          ease: 'sine.inOut',
        });
      }

      if (fishInner) {
        gsap.to(fishInner, {
          opacity: 0.92,
          duration: 5,
          repeat: -1,
          yoyo: true,
          ease: 'sine.inOut',
        });
      }

      gsap.to(fishHeadAndBody, {
        opacity: 0.96,
        duration: 4.4,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        stagger: 0.06,
      });

      gsap.to(fish, {
        yPercent: 4,
        duration: 5.8,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      });

      if (tailFork) {
        gsap.to(tailFork, {
          rotateX: 38,
          duration: 1.1,
          ease: 'tailWiggle',
          repeat: -1,
          yoyo: true,
          transformOrigin: 'top center',
        });
      }

      if (finPrimary) {
        gsap.to(finPrimary, {
          rotateY: 12,
          rotateX: 22,
          rotate: -12,
          duration: 1.8,
          ease: 'finWiggle',
          repeat: -1,
          yoyo: true,
          transformOrigin: 'top center',
        });
      }

      if (finSecondary) {
        gsap.to(finSecondary, {
          rotateY: -12,
          rotateX: -22,
          rotate: -12,
          duration: 1.8,
          ease: 'finWiggle',
          repeat: -1,
          yoyo: true,
          delay: 0.3,
          transformOrigin: 'top center',
        });
      }

      const sectionTriggers = document.querySelectorAll<HTMLElement>('#hero, #story, #upload, #team');

      sectionTriggers.forEach((section) => {
        ScrollTrigger.create({
          trigger: section,
          start: 'top 60%',
          onEnter: () => spawnBubbleBurst(fish, bubblePool),
          onEnterBack: () => spawnBubbleBurst(fish, bubblePool),
        });
      });

      gsap.to(lights, {
        x: () => gsap.utils.random(-100, 100),
        y: () => gsap.utils.random(-100, 100),
        duration: () => gsap.utils.random(5, 10),
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        repeatRefresh: true,
      });
    }, container);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={containerRef} className="pointer-events-none fixed inset-0 z-[2]">
      <div className="lights">
        <div className="lights__group">
          {Array.from({ length: 5 }).map((_, index) => (
            <div key={index} className="lights__light" />
          ))}
        </div>
      </div>

      <div className="fish-wrapper">
        <div className="fish">
          <div className="fish__skeleton" />
          <div className="fish__inner">
            <div className="fish__body" />
            <div className="fish__body" />
            <div className="fish__body" />
            <div className="fish__body" />

            <div className="fish__head" />
            <div className="fish__head fish__head--2" />
            <div className="fish__head fish__head--3" />
            <div className="fish__head fish__head--4" />

            <div className="fish__tail-main" />
            <div className="fish__tail-fork" />

            <div className="fish__fin" />
            <div className="fish__fin fish__fin--2" />
          </div>
        </div>
      </div>

      <div
        ref={bubblePoolRef}
        className="pointer-events-none fixed inset-0 overflow-hidden"
        aria-hidden="true"
      />
    </div>
  );
}

function spawnBubbleBurst(fish: HTMLElement, bubblePool: HTMLElement) {
  const { left, top, width, height } = fish.getBoundingClientRect();
  const originX = left + width / 2;
  const originY = top + height / 2;

  for (let index = 0; index < BUBBLE_COUNT; index += 1) {
    const size = gsap.utils.random(6, 18);
    const bubble = document.createElement('div');

    Object.assign(bubble.style, {
      position: 'fixed',
      width: `${size}px`,
      height: `${size}px`,
      borderRadius: '50%',
      border: '1px solid rgba(124,214,204,0.55)',
      background:
        'radial-gradient(circle at 35% 35%, rgba(124,214,204,0.2), transparent 70%)',
      left: `${originX}px`,
      top: `${originY}px`,
      transform: 'translate(-50%,-50%)',
      opacity: '0',
      pointerEvents: 'none',
    });

    bubblePool.appendChild(bubble);

    const angle = 270 + gsap.utils.random(-35, 35);
    const velocity = gsap.utils.random(80, 220);

    gsap
      .timeline({ onComplete: () => bubble.remove() })
      .set(bubble, { opacity: gsap.utils.random(0.5, 0.9) })
      .to(
        bubble,
        {
          physics2D: {
            angle,
            velocity,
            gravity: -180,
            friction: 0.15,
          },
          rotation: gsap.utils.random(-60, 60),
          scale: gsap.utils.random(0.8, 1.4),
          duration: gsap.utils.random(2.5, 4),
        },
        0,
      )
      .to(
        bubble,
        {
          opacity: 0,
          scale: 0.2,
          duration: 0.6,
          ease: 'power1.in',
        },
        '-=0.8',
      );
  }
}
