'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

type TeamNode = {
  detail: string;
  id: string;
  left: string;
  name: string;
  role: string;
  top: string;
};

type PhysicsState = {
  ox: number;
  oy: number;
  parallaxFactor: number;
  phaseShift: number;
  radiusX: number;
  radiusY: number;
  speed: number;
  timeOffset: number;
};

const nodes: TeamNode[] = [
  {
    name: 'Atharv Jandal',
    id: '25BEMNC15',
    role: 'Vision Engineer',
    detail: 'Core ML pipeline and model training.',
    top: '22%',
    left: '18%',
  },
  {
    name: 'Darsh Dhawan',
    id: '25BECCS22',
    role: 'Vision Engineer',
    detail: 'Dark Channel Prior physics modelling.',
    top: '34%',
    left: '72%',
  },
  {
    name: 'Advitya Sharma',
    id: '25BEMNC05',
    role: 'Vision Engineer',
    detail: 'SS-UIE neural architecture.',
    top: '74%',
    left: '22%',
  },
  {
    name: 'Divij Singh Soodan',
    id: '25BECSE26',
    role: 'Vision Engineer',
    detail: 'Cross-class NMS and detection.',
    top: '78%',
    left: '72%',
  },
];

const edges: Array<[number, number]> = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
];

const MAGNETIC_RADIUS = 220;
const MAX_SCALE = 1.34;
const SCALE_DURATION = 0.42;
const proximityScale = gsap.utils.mapRange(0, MAGNETIC_RADIUS, 1, 0);

export default function TeamConstellation() {
  const containerRef = useRef<HTMLElement>(null);
  const nodesRef = useRef<Array<HTMLDivElement | null>>([]);
  const linesRef = useRef<Array<SVGLineElement | null>>([]);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const targetMouseRef = useRef({ x: 0, y: 0 });
  const smoothMouseRef = useRef({ x: 0, y: 0 });
  const physicsRef = useRef<PhysicsState[]>([]);
  const rafRef = useRef<number>(0);
  const hoveredIndexRef = useRef<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  useEffect(() => {
    hoveredIndexRef.current = hoveredIndex;
  }, [hoveredIndex]);

  const updateLines = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const containerRect = container.getBoundingClientRect();

    edges.forEach(([startIndex, endIndex], index) => {
      const startNode = nodesRef.current[startIndex];
      const endNode = nodesRef.current[endIndex];
      const line = linesRef.current[index];

      if (!startNode || !endNode || !line) {
        return;
      }

      const startRect = startNode.getBoundingClientRect();
      const endRect = endNode.getBoundingClientRect();

      line.setAttribute(
        'x1',
        (startRect.left + startRect.width / 2 - containerRect.left).toString(),
      );
      line.setAttribute(
        'y1',
        (startRect.top + startRect.height / 2 - containerRect.top).toString(),
      );
      line.setAttribute(
        'x2',
        (endRect.left + endRect.width / 2 - containerRect.left).toString(),
      );
      line.setAttribute(
        'y2',
        (endRect.top + endRect.height / 2 - containerRect.top).toString(),
      );
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    if (physicsRef.current.length === 0) {
      physicsRef.current = nodes.map((_, index) => ({
        timeOffset: Math.random() * Math.PI * 2,
        phaseShift: (index / nodes.length) * Math.PI * 2,
        speed: 0.24 + Math.random() * 0.18,
        radiusX: 10 + Math.random() * 12,
        radiusY: 8 + Math.random() * 12,
        parallaxFactor: 10 + Math.random() * 16,
        ox: 0,
        oy: 0,
      }));
    }

    const handleMouseMove = (event: MouseEvent) => {
      mouseRef.current = { x: event.clientX, y: event.clientY };
      targetMouseRef.current = {
        x: (event.clientX / window.innerWidth) * 2 - 1,
        y: (event.clientY / window.innerHeight) * 2 - 1,
      };

      const target =
        event.target instanceof Element
          ? event.target.closest<HTMLElement>('[data-team-node="true"]')
          : null;
      const nextHoveredIndex = target?.dataset.teamNodeIndex
        ? Number(target.dataset.teamNodeIndex)
        : null;

      if (hoveredIndexRef.current !== nextHoveredIndex) {
        hoveredIndexRef.current = nextHoveredIndex;
        setHoveredIndex(nextHoveredIndex);
      }
    };

    const handleMouseLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 };
      targetMouseRef.current = { x: 0, y: 0 };
      hoveredIndexRef.current = null;
      setHoveredIndex(null);
    };

    const tick = () => {
      smoothMouseRef.current.x +=
        (targetMouseRef.current.x - smoothMouseRef.current.x) * 0.045;
      smoothMouseRef.current.y +=
        (targetMouseRef.current.y - smoothMouseRef.current.y) * 0.045;

      const time = Date.now() / 1000;
      const containerRect = container.getBoundingClientRect();

      nodes.forEach((node, index) => {
        const element = nodesRef.current[index];
        const state = physicsRef.current[index];
        if (!element) {
          return;
        }

        const driftX =
          Math.sin(time * state.speed + state.timeOffset) * state.radiusX * 0.72 +
          Math.sin(time * state.speed * 0.54 + state.phaseShift) * state.radiusX * 0.28;
        const driftY =
          Math.cos(time * state.speed + state.timeOffset) * state.radiusY * 0.72 +
          Math.cos(time * state.speed * 0.48 + state.phaseShift) * state.radiusY * 0.28;

        const parallaxX = -smoothMouseRef.current.x * state.parallaxFactor;
        const parallaxY = -smoothMouseRef.current.y * state.parallaxFactor;

        state.ox = driftX + parallaxX;
        state.oy = driftY + parallaxY;

        gsap.set(element, {
          x: state.ox,
          y: state.oy,
          transformOrigin: '50% 50%',
        });

        const baseX = (parseFloat(node.left) / 100) * containerRect.width + state.ox;
        const baseY = (parseFloat(node.top) / 100) * containerRect.height + state.oy;
        const distance = Math.hypot(
          mouseRef.current.x - containerRect.left - baseX,
          mouseRef.current.y - containerRect.top - baseY,
        );

        const proximity = gsap.utils.clamp(0, 1, proximityScale(distance));
        const eased = Math.pow(proximity, 1.55);
        let targetScale = 1 + (MAX_SCALE - 1) * eased;

        if (hoveredIndexRef.current !== null && hoveredIndexRef.current !== index) {
          targetScale = Math.min(targetScale, 0.86);
        }

        gsap.to(element, {
          scale: targetScale,
          duration: SCALE_DURATION,
          ease: 'power2.out',
          overwrite: 'auto',
        });
      });

      updateLines();
      rafRef.current = requestAnimationFrame(tick);
    };

    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mouseleave', handleMouseLeave);
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(rafRef.current);
    };
  }, [updateLines]);

  useEffect(() => {
    nodesRef.current.forEach((element, index) => {
      if (!element) {
        return;
      }

      gsap.fromTo(
        element,
        { opacity: 0, scale: 0.5, y: 16 },
        {
          opacity: 1,
          scale: 1,
          y: 0,
          duration: 0.9,
          delay: index * 0.12,
          ease: 'back.out(1.5)',
        },
      );
    });
  }, []);

  return (
    <section
      id="team"
      ref={containerRef}
      className="relative min-h-screen w-full overflow-visible border-t border-white/5 bg-transparent py-24"
      style={{ cursor: 'none' }}
    >
      <MagneticCursor />

      <p className="pointer-events-none absolute top-24 left-1/2 z-20 -translate-x-1/2 text-xs uppercase tracking-[0.32em] text-[#8edfd2]">
        The Architects
      </p>

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          background:
            'radial-gradient(ellipse 64% 52% at 50% 50%, rgba(124,214,204,0.05) 0%, transparent 72%)',
        }}
      />

      <svg className="pointer-events-none absolute inset-0 z-0 h-full w-full" aria-hidden="true">
        {edges.map((_, index) => (
          <line
            key={`line-${index}`}
            ref={(element) => {
              linesRef.current[index] = element;
            }}
            stroke="rgba(124,214,204,0.08)"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>

      {nodes.map((node, index) => (
        <div
          key={node.id}
          ref={(element) => {
            nodesRef.current[index] = element;
          }}
          className="group absolute z-10 w-max -translate-x-1/2 -translate-y-1/2"
          style={{
            top: node.top,
            left: node.left,
            opacity: 0,
            zIndex: hoveredIndex === index ? 40 : 10,
          }}
          data-team-node="true"
          data-team-node-index={index}
        >
          <TeamNodeCard node={node} isHovered={hoveredIndex === index} />
        </div>
      ))}
    </section>
  );
}

function TeamNodeCard({
  node,
  isHovered,
}: {
  node: TeamNode;
  isHovered: boolean;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const card = cardRef.current;
    if (!card) {
      return;
    }

    if (isHovered) {
      gsap.to(card, {
        opacity: 1,
        y: 0,
        scale: 1,
        duration: 0.28,
        ease: 'back.out(1.55)',
        overwrite: 'auto',
      });
      return;
    }

    gsap.to(card, {
      opacity: 0,
      y: 24,
      scale: 0.9,
      duration: 0.18,
      ease: 'power3.in',
      overwrite: 'auto',
    });
  }, [isHovered]);

  return (
    <div className="relative flex cursor-pointer flex-col items-center">
      {isHovered && (
        <span className="pointer-events-none absolute inset-0 -z-10 animate-ping rounded-full opacity-20 ring-1 ring-[#8edfd2]" />
      )}

      <span
        className="font-serif text-3xl md:text-4xl transition-colors duration-200"
        style={{
          color: isHovered ? '#f5f1e9' : 'rgba(203,213,225,0.72)',
          textShadow: isHovered
            ? '0 0 40px rgba(124,214,204,0.36), 0 0 80px rgba(124,214,204,0.16)'
            : 'none',
        }}
      >
        /{node.name.toLowerCase().replaceAll(' ', '-')}
      </span>

      <div
        ref={cardRef}
        className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-5 flex w-72 -translate-x-1/2 flex-col items-start rounded-2xl border border-white/10 bg-[rgba(8,15,20,0.96)] p-5 text-left shadow-[0_24px_60px_rgba(0,0,0,0.78),0_0_0_1px_rgba(124,214,204,0.08)] backdrop-blur-xl"
        style={{
          opacity: 0,
          transform: 'translateY(24px) scale(0.9)',
          transformOrigin: '50% 100%',
        }}
      >
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-[rgba(124,214,204,0.18)]" />
        <div className="absolute top-full left-1/2 mt-px -translate-x-1/2 border-[7px] border-transparent border-t-[rgba(8,15,20,0.96)]" />

        <div className="mb-4 h-px w-full bg-gradient-to-r from-[#8edfd2] via-[rgba(213,187,139,0.3)] to-transparent" />

        <p className="mb-1 font-sans text-sm font-medium leading-relaxed text-slate-100">
          {node.detail}
        </p>
        <p className="mb-4 text-xs text-slate-500">{node.name}</p>

        <div className="flex w-full items-center justify-between border-t border-white/8 pt-3">
          <div className="flex flex-col gap-1">
            <span className="text-[9px] font-bold uppercase tracking-widest text-[#8edfd2]">
              {node.role}
            </span>
            <span className="font-mono text-[10px] tracking-[0.18em] text-[rgba(213,187,139,0.72)]">
              {node.id}
            </span>
          </div>
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[rgba(124,214,204,0.08)] ring-1 ring-[rgba(124,214,204,0.18)]">
            <svg
              width="10"
              height="10"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#8edfd2"
              strokeWidth="2.5"
              aria-hidden="true"
            >
              <path d="M5 12h14" />
              <path d="m12 5 7 7-7 7" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

function MagneticCursor() {
  const dotRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) {
      return;
    }

    let mouseX = 0;
    let mouseY = 0;
    let ringX = 0;
    let ringY = 0;
    let frameId = 0;

    const handleMove = (event: MouseEvent) => {
      mouseX = event.clientX;
      mouseY = event.clientY;
      gsap.set(dot, { x: mouseX, y: mouseY });
    };

    const tick = () => {
      ringX += (mouseX - ringX) * 0.12;
      ringY += (mouseY - ringY) * 0.12;
      gsap.set(ring, { x: ringX, y: ringY });
      frameId = requestAnimationFrame(tick);
    };

    window.addEventListener('mousemove', handleMove);
    frameId = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('mousemove', handleMove);
      cancelAnimationFrame(frameId);
    };
  }, []);

  return (
    <>
      <div
        ref={dotRef}
        className="pointer-events-none fixed z-[9999] -translate-x-1/2 -translate-y-1/2"
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: '#8edfd2',
          boxShadow: '0 0 12px rgba(124,214,204,0.72)',
          top: 0,
          left: 0,
        }}
      />
      <div
        ref={ringRef}
        className="pointer-events-none fixed z-[9998] -translate-x-1/2 -translate-y-1/2 rounded-full border border-[rgba(124,214,204,0.36)]"
        style={{ width: 36, height: 36, top: 0, left: 0 }}
      />
    </>
  );
}
