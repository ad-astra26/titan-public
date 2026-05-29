import { redirect } from 'next/navigation';

// Legacy route — consolidated under /world (Phase 2 IA, 2026-05-10).
export default function SystemPage() {
  redirect('/world?tab=system');
}
