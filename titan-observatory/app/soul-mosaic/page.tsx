import { redirect } from 'next/navigation';
// Phase 2 IA — folded into World → Soul Mosaic tab.
export default function SoulMosaicPage() {
  redirect('/world?tab=soul-mosaic');
}
