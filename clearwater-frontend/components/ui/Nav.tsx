'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import MagneticButton from './MagneticButton';

export default function Nav() {
  const pathname = usePathname();
  const [activeSection, setActiveSection] = useState('story');

  useEffect(() => {
    if (pathname !== '/') {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: '-50% 0px -50% 0px' },
    );

    const sections = document.querySelectorAll<HTMLElement>('section[id]');
    sections.forEach((section) => observer.observe(section));

    return () => observer.disconnect();
  }, [pathname]);

  const getLinkClass = (section: string) => {
    const isActive = pathname === '/' && activeSection === section;

    return isActive
      ? '!min-h-0 !border-transparent !bg-transparent !px-4 !py-2 !text-white !opacity-100 !shadow-none hover:!shadow-none'
      : '!min-h-0 !border-transparent !bg-transparent !px-4 !py-2 !text-slate-500 !opacity-70 !shadow-none hover:!text-white hover:!opacity-100 hover:!shadow-none';
  };

  return (
    <nav
      id="global-nav"
      className="fixed inset-x-0 top-0 z-50 border-b border-white/6 bg-[rgba(4,11,14,0.62)] px-4 py-4 backdrop-blur-xl md:px-6 lg:px-8"
    >
      <div className="mx-auto flex w-full max-w-[1400px] items-center gap-4 md:gap-6">
        <div className="pointer-events-auto flex min-w-0 flex-1 flex-col md:flex-none md:min-w-[220px]">
          <span className="text-lg font-bold tracking-wide text-white">Clearwater</span>
          <span className="text-[10px] uppercase tracking-[0.3em] text-[rgba(213,187,139,0.64)]">Underwater Vision</span>
        </div>

        <div className="hidden flex-1 items-center justify-center gap-2 text-xs uppercase tracking-widest md:flex">
          <MagneticButton
            href="/#story"
            strength={0.18}
            labelStrength={0.08}
            variant="ghost"
            className={getLinkClass('story')}
          >
            Story
          </MagneticButton>

          <MagneticButton
            href="/#team"
            strength={0.18}
            labelStrength={0.08}
            variant="ghost"
            className={getLinkClass('team')}
          >
            Team
          </MagneticButton>

          <MagneticButton
            href="/#upload"
            strength={0.18}
            labelStrength={0.08}
            variant="ghost"
            className={getLinkClass('upload')}
          >
            Lab
          </MagneticButton>
        </div>

        <div className="flex flex-1 justify-end md:min-w-[220px] md:flex-none">
          <MagneticButton
            href="/#upload"
            strength={0.24}
            labelStrength={0.12}
            variant="ghost"
            className="lab-beacon lab-beacon--nav !border-transparent !p-0"
          >
            Enter the Lab &rarr;
          </MagneticButton>
        </div>
      </div>
    </nav>
  );
}
