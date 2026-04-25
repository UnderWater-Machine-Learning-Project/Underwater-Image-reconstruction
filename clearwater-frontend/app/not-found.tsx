import MagneticButton from '../components/ui/MagneticButton';

export default function NotFound() {
  return (
    <main className="py-10 md:py-16">
      <section className="mx-auto mt-8 max-w-[1180px] w-full px-4 md:px-8">
        <div className="relative overflow-hidden rounded-[32px] border border-[#6ec3ff]/16 bg-gradient-to-b from-[#08131e]/78 to-[#091826]/92 p-10 shadow-[0_30px_80px_rgba(0,0,0,0.28)] backdrop-blur-md">
          <p className="m-0 font-mono text-xs uppercase tracking-[0.24em] text-[#9ce0ff]">Route missing</p>
          <h1 className="mt-4 font-serif text-4xl leading-none tracking-tight md:text-5xl">That page is not part of the Clearwater information architecture.</h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">
            Use the architecture and subsystem pages for deep technical detail, or jump straight back into the
            one-page funnel.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <MagneticButton href="/">Go Home</MagneticButton>
            <MagneticButton href="/#lab" variant="secondary">
              Enter Lab
            </MagneticButton>
          </div>
        </div>
      </section>
    </main>
  );
}
