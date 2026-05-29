import { redirect } from 'next/navigation';
// Phase 2 IA — folded into Self · Trinity → Rhythms tab.
export default function RhythmsPage() {
  redirect('/trinity?tab=rhythms');
}
