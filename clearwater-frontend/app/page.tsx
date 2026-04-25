import HeroSection from '@/components/sections/HeroSection';
import ScrollStorySection from '@/components/sections/ScrollStorySection';
import UploadSection from '@/components/sections/UploadSection';
import TeamConstellation from '@/components/TeamConstellation';

export default function HomePage() {
  return (
    <main className="relative">
      <HeroSection />
      <ScrollStorySection />
      <UploadSection />
      <TeamConstellation />
    </main>
  );
}
