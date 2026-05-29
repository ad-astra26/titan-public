import { redirect } from 'next/navigation';

// Phase 2 IA, 2026-05-10: Persona is now a sub-tab of Voice (Expression).
export default function PersonaPage() {
  redirect('/expression?tab=persona');
}
