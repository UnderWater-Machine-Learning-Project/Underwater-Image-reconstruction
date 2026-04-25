import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import BackgroundFish from '../components/ui/BackgroundFish';
import Nav from '../components/ui/Nav';
import './fish.scss';
import './globals.css';

const geistSans = Geist({
  variable: '--font-display',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Clearwater | Underwater Vision Interface',
  description:
    'PRD-aligned frontend for underwater image restoration, enhancement, and marine object detection.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body>
        <div className="site-shell">
          <BackgroundFish />
          <Nav />
          {children}
        </div>
      </body>
    </html>
  );
}
