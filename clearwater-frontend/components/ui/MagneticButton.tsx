'use client';

import Link from 'next/link';
import {
  type ButtonHTMLAttributes,
  type MouseEventHandler,
  type Ref,
  type ReactNode,
  useEffect,
  useEffectEvent,
  useRef,
} from 'react';
import { gsap } from '../../lib/gsap';
import { useReducedMotion } from '../../lib/useReducedMotion';

type Variant = 'primary' | 'secondary' | 'ghost';
type Size = 'sm' | 'md';

type SharedProps = {
  children: ReactNode;
  className?: string;
  label?: string;
  labelStrength?: number;
  size?: Size;
  strength?: number;
  variant?: Variant;
};

type LinkProps = SharedProps & {
  href: string;
  onClick?: MouseEventHandler<HTMLAnchorElement>;
  type?: never;
};

type ActionProps = SharedProps &
  Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> & {
    href?: never;
  };

type MagneticButtonProps = LinkProps | ActionProps;

function joinClasses(...values: Array<string | undefined | false>) {
  return values.filter(Boolean).join(' ');
}

export default function MagneticButton({
  children,
  className,
  href,
  label,
  labelStrength = 0.2,
  size = 'md',
  strength = 0.38,
  variant = 'primary',
  ...rest
}: MagneticButtonProps) {
  const reducedMotion = useReducedMotion();
  const zoneRef = useRef<HTMLSpanElement>(null);
  const actionRef = useRef<HTMLElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);

  const handlePointerMove = useEffectEvent((event: PointerEvent) => {
    if (reducedMotion) {
      return;
    }

    const zone = zoneRef.current;
    const action = actionRef.current;
    const labelNode = labelRef.current;

    if (!zone || !action || !labelNode) {
      return;
    }

    const rect = zone.getBoundingClientRect();
    const x = gsap.utils.mapRange(rect.left, rect.right, -rect.width / 2, rect.width / 2, event.clientX);
    const y = gsap.utils.mapRange(rect.top, rect.bottom, -rect.height / 2, rect.height / 2, event.clientY);

    gsap.to(action, {
      x: x * strength,
      y: y * strength,
      duration: 0.28,
      ease: 'power3.out',
      overwrite: 'auto',
    });

    gsap.to(labelNode, {
      x: x * labelStrength,
      y: y * labelStrength,
      duration: 0.28,
      ease: 'power3.out',
      overwrite: 'auto',
    });
  });

  const resetPosition = useEffectEvent(() => {
    const action = actionRef.current;
    const labelNode = labelRef.current;

    if (!action || !labelNode) {
      return;
    }

    gsap.to([action, labelNode], {
      x: 0,
      y: 0,
      duration: 0.6,
      ease: 'elastic.out(1, 0.45)',
      overwrite: 'auto',
    });
  });

  useEffect(() => {
    const zone = zoneRef.current;

    if (!zone || reducedMotion) {
      return;
    }

    zone.addEventListener('pointermove', handlePointerMove);
    zone.addEventListener('pointerleave', resetPosition);

    return () => {
      zone.removeEventListener('pointermove', handlePointerMove);
      zone.removeEventListener('pointerleave', resetPosition);
    };
  }, [reducedMotion]);

  const classes = joinClasses(
    'magnetic-link',
    `magnetic-link--${variant}`,
    `magnetic-link--${size}`,
    className,
  );

  if (href) {
    const { onClick } = rest as LinkProps;

    return (
      <span className="magnetic-zone" ref={zoneRef}>
        <Link
          href={href}
          className={classes}
          onClick={onClick}
          ref={actionRef as Ref<HTMLAnchorElement>}
        >
          <span className="magnetic-label" ref={labelRef} aria-label={label}>
            {children}
          </span>
        </Link>
      </span>
    );
  }

  const buttonProps = rest as ActionProps;

  return (
    <span className="magnetic-zone" ref={zoneRef}>
      <button
        {...buttonProps}
        className={classes}
        ref={actionRef as Ref<HTMLButtonElement>}
        type={buttonProps.type ?? 'button'}
      >
        <span className="magnetic-label" ref={labelRef} aria-label={label}>
          {children}
        </span>
      </button>
    </span>
  );
}
