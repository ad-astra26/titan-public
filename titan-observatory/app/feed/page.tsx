import { redirect } from 'next/navigation';
// Phase 2 IA — folded into Voice · Expression → Feed tab.
export default function FeedPage() {
  redirect('/expression?tab=feed');
}
