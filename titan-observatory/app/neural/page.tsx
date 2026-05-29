import { redirect } from 'next/navigation';
// Phase 2 IA — folded into Self · Trinity → Memory tab.
export default function NeuralPage() {
  redirect('/trinity?tab=memory');
}
