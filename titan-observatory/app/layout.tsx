import type { Metadata } from 'next';
import './globals.css';
import Providers from '@/components/Providers';
import WSInitializer from '@/components/WSInitializer';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import GridBackdrop from '@/components/layout/GridBackdrop';
import MetabolicWrapper from '@/components/layout/MetabolicWrapper';
import ConnectionBanner from '@/components/layout/ConnectionBanner';
import GlobalFreshnessPill from '@/components/layout/GlobalFreshnessPill';

// Force dynamic rendering — pages depend on runtime data (clock, API, WS)
export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  title: 'Titan Observatory',
  description: 'Real-time window into Titan sovereign AI cognitive state',
  icons: {
    icon: '/titan-pfp.png',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-titan antialiased bg-titan-bg text-titan-metal min-h-screen flex flex-col">
        <Providers>
          <WSInitializer />
          <ConnectionBanner />
          <GlobalFreshnessPill />
          <GridBackdrop />
          <MetabolicWrapper>
            <Header />
            <main className="relative z-10 max-w-[1440px] mx-auto px-4 py-6 flex-1 w-full">
              {children}
            </main>
            <Footer />
          </MetabolicWrapper>
        </Providers>
      </body>
    </html>
  );
}
