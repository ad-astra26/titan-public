import { redirect } from 'next/navigation';
// Phase 2 IA — folded into World → Research tab.
export default function ResearchPage() {
  redirect('/world?tab=research');
}
