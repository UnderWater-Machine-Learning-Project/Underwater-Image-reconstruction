import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import MagneticButton from '../../components/ui/MagneticButton';
import { detailPages, getDetailPage } from '../../lib/site-content';

type DetailPageProps = {
  params: Promise<{ slug: string }>;
};

export async function generateStaticParams() {
  return detailPages.map((page) => ({
    slug: page.slug,
  }));
}

export async function generateMetadata({
  params,
}: DetailPageProps): Promise<Metadata> {
  const { slug } = await params;
  const page = getDetailPage(slug);

  if (!page) {
    return {
      title: 'Not Found | Clearwater',
    };
  }

  return {
    title: `${page.navLabel} | Clearwater`,
    description: page.summary,
  };
}

export default async function DetailPage({
  params,
}: DetailPageProps) {
  const { slug } = await params;
  const page = getDetailPage(slug);

  if (!page) {
    notFound();
  }

  const currentIndex = detailPages.findIndex((entry) => entry.slug === slug);
  const nextPage = detailPages[(currentIndex + 1) % detailPages.length];

  return (
    <main className="py-10 md:py-16">
      <section className="mx-auto mt-8 grid w-full max-w-[1180px] grid-cols-1 gap-6 px-4 lg:grid-cols-[1.45fr_0.9fr] md:px-8">
        <div className="relative overflow-hidden rounded-[32px] border border-[#6ec3ff]/16 bg-gradient-to-b from-[#08131e]/78 to-[#091826]/92 p-8 shadow-[0_30px_80px_rgba(0,0,0,0.28)] backdrop-blur-md md:p-11">
          <p className="m-0 font-mono text-xs uppercase tracking-[0.24em] text-[#9ce0ff]">{page.eyebrow}</p>
          <h1 className="mt-4 font-serif text-5xl leading-none tracking-tight md:text-7xl">{page.title}</h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">{page.intro}</p>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-slate-500">{page.summary}</p>
          <div className="mt-8 flex flex-wrap gap-4">
            <MagneticButton href="/">Back Home</MagneticButton>
            <MagneticButton href={`/${nextPage.slug}`} variant="secondary">
              Next: {nextPage.navLabel}
            </MagneticButton>
          </div>
        </div>

        <aside className="relative overflow-hidden rounded-[28px] border border-[#6ec3ff]/16 bg-gradient-to-b from-[#08131e]/78 to-[#091826]/92 p-8 shadow-[0_30px_80px_rgba(0,0,0,0.28)] backdrop-blur-md" aria-label="Key metrics">
          <p className="m-0 font-mono text-xs uppercase tracking-[0.24em] text-[#9ce0ff]">Key metrics</p>
          <div className="mt-6 grid gap-4">
            {page.stats.map((stat) => (
              <article key={stat.label} className="rounded-2xl border border-[#6ec3ff]/14 bg-[#071019]/50 p-5">
                <span className="mb-3 block text-3xl leading-none tracking-tight md:text-4xl">{stat.value}</span>
                <strong className="inline-flex font-mono text-xs uppercase tracking-[0.2em] text-[#9ce0ff]">{stat.label}</strong>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{stat.detail}</p>
              </article>
            ))}
          </div>
        </aside>
      </section>

      <section className="mx-auto mt-6 grid w-full max-w-[1180px] gap-5 px-4 md:px-8">
        {page.sections.map((section, index) => (
          <article key={section.title} className="relative grid overflow-hidden rounded-[28px] border border-[#6ec3ff]/16 bg-gradient-to-b from-[#08131e]/78 to-[#091826]/92 p-8 shadow-[0_30px_80px_rgba(0,0,0,0.28)] backdrop-blur-md grid-cols-1 md:grid-cols-[92px_1fr] gap-6">
            <div className="pt-1 font-mono text-xs uppercase tracking-[0.2em] text-[#9ce0ff]">0{index + 1}</div>
            <div>
              <h2 className="mb-3 mt-0 text-2xl tracking-tight">{section.title}</h2>
              <p className="leading-relaxed text-slate-400">{section.body}</p>
              <ul className="m-0 mt-5 grid list-none gap-3 p-0">
                {section.bullets.map((bullet) => (
                  <li key={bullet} className="relative pl-5 leading-relaxed text-white before:absolute before:left-0 before:top-[11px] before:h-2 before:w-2 before:rounded-full before:bg-gradient-to-br before:from-[#6ec3ff] before:to-[#6ee7b7] before:shadow-[0_0_18px_rgba(110,195,255,0.4)]">{bullet}</li>
                ))}
              </ul>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
