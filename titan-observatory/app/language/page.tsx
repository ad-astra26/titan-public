import { redirect } from 'next/navigation';

export default function LanguagePage() {
  redirect('/expression?tab=language');
}
