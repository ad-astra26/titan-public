import { redirect } from 'next/navigation';
// Phase 2 IA — folded into World → System tab.
export default function StatsPage() {
  redirect('/world?tab=system');
}
