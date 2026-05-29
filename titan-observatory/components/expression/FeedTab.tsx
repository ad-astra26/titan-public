'use client';

import MoodNarrative from '@/components/feed/MoodNarrative';
import NarratedStream from '@/components/feed/NarratedStream';
import MasonryGallery from '@/components/feed/MasonryGallery';

export default function FeedTab() {
  return (
    <div className="flex flex-col gap-5">
      <MoodNarrative />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <NarratedStream />
        </div>
        <div className="space-y-4">
          <MasonryGallery />
        </div>
      </div>
    </div>
  );
}
