import { redirect } from 'next/navigation';
// Phase 2 IA — folded into Voice · Expression → Creative tab.
export default function CreativePage() {
  redirect('/expression?tab=creative');
}
