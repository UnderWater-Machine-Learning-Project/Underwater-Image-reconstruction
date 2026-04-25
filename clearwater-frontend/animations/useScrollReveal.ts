import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export type RevealVariant =
  | 'fade-up'      // y offset + opacity
  | 'clip-left'    // clip-path wipe from left
  | 'clip-bottom'  // clip-path wipe from bottom
  | 'scale-in'     // scale + opacity
  | 'blur-in';     // blur + opacity

type Options = {
  variant?:  RevealVariant;
  duration?: number;
  stagger?:  number;
  delay?:    number;
  start?:    string;
  ease?:     string;
  once?:     boolean;
};

/**
 * Attach to a container ref — targets every child with className "sr-item".
 * Usage:
 *   const ref = useScrollReveal({ variant: 'fade-up', stagger: 0.1 });
 *   <div ref={ref}>
 *     <p className="sr-item">...</p>
 *     <p className="sr-item">...</p>
 *   </div>
 */
export function useScrollReveal<T extends HTMLElement = HTMLDivElement>(opts: Options = {}) {
  const ref = useRef<T>(null);
  const {
    variant  = 'fade-up',
    duration = 0.9,
    stagger  = 0.1,
    delay    = 0,
    start    = 'top 80%',
    ease     = 'power3.out',
    once     = true,
  } = opts;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const items = gsap.utils.toArray<HTMLElement>('.sr-item', el);
    if (items.length === 0) return;

    const fromVars: gsap.TweenVars = {};
    const toVars:   gsap.TweenVars = { duration, stagger, delay, ease };

    switch (variant) {
      case 'fade-up':
        Object.assign(fromVars, { y: 48, opacity: 0 });
        Object.assign(toVars,   { y: 0,  opacity: 1 });
        break;
      case 'clip-left':
        Object.assign(fromVars, { clipPath: 'inset(0 100% 0 0)',  opacity: 1 });
        Object.assign(toVars,   { clipPath: 'inset(0 0% 0 0)',    opacity: 1 });
        break;
      case 'clip-bottom':
        Object.assign(fromVars, { clipPath: 'inset(100% 0 0 0)',  opacity: 1 });
        Object.assign(toVars,   { clipPath: 'inset(0% 0 0 0)',    opacity: 1 });
        break;
      case 'scale-in':
        Object.assign(fromVars, { scale: 0.88, opacity: 0 });
        Object.assign(toVars,   { scale: 1,    opacity: 1 });
        break;
      case 'blur-in':
        Object.assign(fromVars, { opacity: 0, filter: 'blur(12px)', y: 20 });
        Object.assign(toVars,   { opacity: 1, filter: 'blur(0px)',  y: 0 });
        break;
    }

    const ctx = gsap.context(() => {
      gsap.fromTo(items, fromVars, {
        ...toVars,
        scrollTrigger: {
          trigger: el,
          start,
          once,
        },
      });
    }, el);

    return () => ctx.revert();
  }, [variant, duration, stagger, delay, start, ease, once]);

  return ref;
}

/**
 * Morph transition between two sections.
 * Applies a clip-path expansion as the trigger enters the viewport.
 */
export function useMorphIn(options: { start?: string; ease?: string } = {}) {
  const ref = useRef<HTMLElement>(null);
  const { start = 'top 85%', ease = 'expo.out' } = options;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        el,
        { clipPath: 'inset(8% 4% 8% 4% round 32px)', opacity: 0.4 },
        {
          clipPath: 'inset(0% 0% 0% 0% round 0px)',
          opacity: 1,
          duration: 1.2,
          ease,
          scrollTrigger: {
            trigger: el,
            start,
            once: true,
          },
        },
      );
    }, el);

    return () => ctx.revert();
  }, [start, ease]);

  return ref;
}

/**
 * Parallax effect — moves element at a fractional scroll speed.
 * @param speed  - 0 = pinned, 1 = normal scroll, negative = counter-scroll
 */
export function useParallax<T extends HTMLElement = HTMLDivElement>(speed = 0.4) {
  const ref = useRef<T>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const ctx = gsap.context(() => {
      gsap.to(el, {
        yPercent: speed * -50,
        ease: 'none',
        scrollTrigger: {
          trigger: el,
          start: 'top bottom',
          end:   'bottom top',
          scrub: 1,
        },
      });
    }, el);

    return () => ctx.revert();
  }, [speed]);

  return ref;
}
