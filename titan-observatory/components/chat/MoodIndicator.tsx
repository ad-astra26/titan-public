'use client';

interface MoodIndicatorProps {
  mood: string;
}

const moodColors: Record<string, string> = {
  Curious: 'bg-titan-pulse',
  Contemplative: 'bg-blue-400',
  Energized: 'bg-titan-growth',
  Creative: 'bg-pink-400',
  Cautious: 'bg-yellow-400',
  Restless: 'bg-orange-400',
  Serene: 'bg-teal-300',
  Focused: 'bg-titan-haze',
  Melancholic: 'bg-indigo-400',
};

export default function MoodIndicator({ mood }: MoodIndicatorProps) {
  if (!mood) return null;

  const dotColor = moodColors[mood] || 'bg-titan-metal';

  return (
    <span className="inline-flex items-center gap-1">
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span className="text-[10px] text-titan-metal/80">{mood}</span>
    </span>
  );
}
