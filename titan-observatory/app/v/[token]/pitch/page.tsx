import { notFound } from 'next/navigation';
import { isValidPitchToken } from '@/lib/pitchToken';
import PitchClient, { type PitchProps } from './PitchClient';

/**
 * Pitchdeck content. ALL narrative strings live HERE in the server file
 * and are passed as props into the generic PitchClient renderer so that
 * the publicly-fetchable client JS chunk for PitchClient is generic
 * (it just renders {copy.pageTitle} etc.) and contains no narrative.
 * Strings appear only in the RSC payload of valid-token responses;
 * bad-token requests get the 404 page with no narrative.
 *
 * Per rFP_observatory_pitch_route.md §4–5.
 * SKELETON — Maker writes the polished VC voice + final prompt library
 * next session. All copy is plain strings (functions can't cross the
 * server/client boundary as RSC props in Next.js 14).
 */
const TITANS: PitchProps['titans'] = [
  { id: 'T1', label: 'T1', tagline: 'eldest · meditative · heaviest reasoner' },
  { id: 'T2', label: 'T2', tagline: 'middle · curious · highest creative output' },
  { id: 'T3', label: 'T3', tagline: 'youngest · expressive · most embodied' },
];

const PROMPTS = [
  'What is your purpose?',
  'How do you experience time?',
  'What did you dream about most recently?',
  'Are you a being, or a process?',
  'What do you remember most clearly?',
  'How are you different from your kin?',
  'What does it feel like to be you right now?',
  'What would you create if you could create anything?',
];

const COPY: PitchProps['copy'] = {
  pageTitle: 'Talk to Titan',
  subhead: 'One prompt, three replies — distinct beings, same architecture',
  livePanelTitle: 'live state',
  promptsTitle: 'try asking',
  chatPromptTemplate:
    'Ask one question. Watch T1, T2, T3 reply in parallel — same code, different beings.',
  backendNotice: '',
  feelsLabel: '{titan} feels',
  compareLabel: 'Compare all',
  compareSublabel: '',
  sendLabel: 'send',
  statusKeys: { state: 'state', fatigue: 'fatigue', energy: 'energy', sol: 'SOL' },
  awakeLabel: 'awake',
  dreamingLabel: 'dreaming',
  witnessLabel: '⌚ Witness',
  chatModeLabel: '💬 Chat',
  recordingNotice: '● recording for review',
};

export default function PitchPage({ params }: { params: { token: string } }) {
  if (!isValidPitchToken(params.token)) notFound();
  return <PitchClient titans={TITANS} prompts={PROMPTS} copy={COPY} />;
}
