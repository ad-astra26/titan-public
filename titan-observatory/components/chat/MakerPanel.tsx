'use client';

import { useState, useEffect, useCallback } from 'react';
import { useSignMessage, useWallets } from '@privy-io/react-auth/solana';
import bs58 from 'bs58';
import { titanFetch } from '@/lib/api';
import { useTitanId } from '@/components/shared/TitanSelector';

// ── Types ────────────────────────────────────────────────────────

interface ProposalRecord {
  proposal_id: string;
  proposal_type: string;
  title: string;
  description: string;
  payload: Record<string, unknown>;
  payload_hash: string;
  created_at: number;
  created_epoch: number;
  requires_signature: boolean;
  status: 'pending' | 'approved' | 'declined' | 'expired';
  expires_at?: number | null;
  approved_at?: number | null;
  approval_reason?: string | null;
  approved_signer_pubkey?: string | null;
  declined_at?: number | null;
  decline_reason?: string | null;
}

interface MakerProposalsResponse {
  pending: ProposalRecord[];
  recent: ProposalRecord[];
  maker_pubkey: string | null;
  alignment_score: number;
}

// ── Component ────────────────────────────────────────────────────

// ── Tier 3: Dialogue History sub-component ──────────────────────

function DialogueHistory() {
  const [dialogueData, setDialogueData] = useState<{
    dialogue: Array<{
      dialogue_id: string;
      proposal_type: string;
      response: string;
      maker_reason: string;
      titan_narration: string;
      created_at: number;
    }>;
    bond_health: Record<string, number>;
  } | null>(null);

  useEffect(() => {
    const fetchDialogue = async () => {
      try {
        const res = await titanFetch<{
          dialogue: Array<{
            dialogue_id: string;
            proposal_type: string;
            response: string;
            maker_reason: string;
            titan_narration: string;
            created_at: number;
          }>;
          bond_health: Record<string, number>;
        }>('/v6/maker/dialogue-history');
        setDialogueData(res);
      } catch { /* silent */ }
    };
    fetchDialogue();
    const id = setInterval(fetchDialogue, 30000);
    return () => clearInterval(id);
  }, []);

  if (!dialogueData?.dialogue?.length) return null;

  return (
    <>
      <div className="px-4 py-2 border-t border-titan-gold/20 text-[10px] uppercase tracking-wider text-titan-gold/60">
        Bond Dialogue
        {dialogueData.bond_health?.interaction_count > 0 && (
          <span className="float-right text-titan-metal/40">
            {dialogueData.bond_health.interaction_count} exchanges
          </span>
        )}
      </div>
      <div className="px-4 py-2 space-y-2 pb-3 max-h-48 overflow-y-auto">
        {dialogueData.dialogue.slice(0, 10).map((d) => (
          <div key={d.dialogue_id} className="text-[10px]">
            <div className="text-titan-metal/50">
              <span className={d.response === 'approve' ? 'text-green-400' : 'text-red-400'}>
                {d.response === 'approve' ? '✓' : '✗'}
              </span>{' '}
              <span className="text-titan-metal/60">
                {d.proposal_type.replace(/_/g, ' ')}
              </span>
            </div>
            <div className="text-[9px] italic text-titan-metal/40 ml-3 mt-0.5">
              Maker: &ldquo;{d.maker_reason.slice(0, 80)}{d.maker_reason.length > 80 ? '...' : ''}&rdquo;
            </div>
            {d.titan_narration && (
              <div className="text-[9px] text-titan-gold/40 ml-3 mt-0.5">
                Titan: &ldquo;{d.titan_narration.slice(0, 80)}{d.titan_narration.length > 80 ? '...' : ''}&rdquo;
              </div>
            )}
          </div>
        ))}
      </div>
    </>
  );
}

// ── Main MakerPanel component ───────────────────────────────────

export default function MakerPanel() {
  // ChatWindow only mounts this component when isMaker === true, so we
  // don't re-check the store here. (Calling usePrivyMaker(null) inside
  // would clobber the store back to false — feedback loop.)
  const { wallets } = useWallets();
  const { signMessage } = useSignMessage();
  // Phase 2 closure (2026-05-25): the active Titan switch in /chat
  // (?titan=T1|T2|T3 via TitanSelector) is now honored for proposal
  // listing + approve/decline. Each Titan has its own proposal queue +
  // R8 bundle signature (per-Titan sovereignty per arch + the existing
  // multi-Titan routing in titanFetch). Switching tabs in the chat
  // header re-queries the selected Titan's pending proposals.
  const titanId = useTitanId();

  const [data, setData] = useState<MakerProposalsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeProposal, setActiveProposal] = useState<ProposalRecord | null>(null);
  const [actionMode, setActionMode] = useState<'approve' | 'decline' | null>(null);
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitMsg, setSubmitMsg] = useState<string | null>(null);

  // Poll /v6/maker/proposals every 10s for the active Titan.
  const fetchProposals = useCallback(async () => {
    try {
      const res = await titanFetch<MakerProposalsResponse>(
        '/v6/maker/proposals', { titan: titanId });
      setData(res);
      setError(null);
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setError(errMsg);
    }
  }, [titanId]);

  useEffect(() => {
    // Clear stale data when switching Titans so the UI shows "loading"
    // rather than the previous Titan's proposals while the new fetch
    // is in-flight.
    setData(null);
    setError(null);
    fetchProposals();
    const id = setInterval(fetchProposals, 10000);
    return () => clearInterval(id);
  }, [fetchProposals]);

  const openAction = (proposal: ProposalRecord, mode: 'approve' | 'decline') => {
    setActiveProposal(proposal);
    setActionMode(mode);
    setReason('');
    setSubmitMsg(null);
  };

  const closeAction = () => {
    setActiveProposal(null);
    setActionMode(null);
    setReason('');
    setSubmitMsg(null);
  };

  const handleSubmit = async () => {
    if (!activeProposal || !actionMode) return;
    if (reason.trim().length < 10) {
      setSubmitMsg('Reason must be at least 10 characters');
      return;
    }
    setSubmitting(true);
    setSubmitMsg(null);
    try {
      if (actionMode === 'approve' && activeProposal.requires_signature) {
        // Sign the payload_hash via Privy and post the signature
        const wallet = wallets[0];
        if (!wallet) {
          setSubmitMsg('No connected wallet found');
          setSubmitting(false);
          return;
        }
        const messageBytes = new TextEncoder().encode(activeProposal.payload_hash);
        const result = await signMessage({
          message: messageBytes,
          wallet,
        });
        const signatureB58 = bs58.encode(result.signature);
        await titanFetch(
          `/v6/maker/proposals/${activeProposal.proposal_id}/approve`,
          {
            method: 'POST',
            titan: titanId,
            body: JSON.stringify({
              reason: reason.trim(),
              signature_b58: signatureB58,
              signer_pubkey_b58: wallet.address,
            }),
          },
        );
      } else if (actionMode === 'approve') {
        await titanFetch(
          `/v6/maker/proposals/${activeProposal.proposal_id}/approve`,
          {
            method: 'POST',
            titan: titanId,
            body: JSON.stringify({ reason: reason.trim() }),
          },
        );
      } else {
        await titanFetch(
          `/v6/maker/proposals/${activeProposal.proposal_id}/decline`,
          {
            method: 'POST',
            titan: titanId,
            body: JSON.stringify({ reason: reason.trim() }),
          },
        );
      }
      setSubmitMsg(`${actionMode === 'approve' ? 'Approved' : 'Declined'} ✓`);
      // Refresh and close after a moment
      await fetchProposals();
      setTimeout(() => closeAction(), 1500);
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      setSubmitMsg(`Error: ${errMsg}`);
    } finally {
      setSubmitting(false);
    }
  };

  const pending = data?.pending ?? [];
  const recent = data?.recent ?? [];

  return (
    <aside className="w-80 border-l border-titan-gold/20 bg-titan-card/30 overflow-y-auto flex flex-col">
      <div className="px-4 py-3 border-b border-titan-gold/20 bg-titan-gold/5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-titan-gold uppercase tracking-wider">
            Maker Panel · {titanId}
          </span>
          {data && (
            <span className="text-[10px] text-titan-metal/50">
              alignment {(data.alignment_score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div className="text-[10px] text-titan-metal/40 mt-1">
          {pending.length} pending · {recent.length} recent
        </div>
      </div>

      {error && (
        <div className="px-4 py-2 text-[11px] text-red-400/80">{error}</div>
      )}

      {/* Pending proposals */}
      <div className="px-4 py-3 space-y-3">
        {pending.length === 0 && (
          <div className="text-[11px] text-titan-metal/40 italic">
            No pending proposals
          </div>
        )}
        {pending.map((p) => (
          <div
            key={p.proposal_id}
            className="rounded-lg border border-titan-gold/30 bg-titan-card/50 p-3"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] uppercase tracking-wider text-titan-gold/70">
                {p.proposal_type.replace(/_/g, ' ')}
              </span>
              {p.requires_signature && (
                <span className="text-[9px] text-solana-purple/80">requires sig</span>
              )}
            </div>
            <div className="text-xs font-semibold text-titan-steel mb-1">{p.title}</div>
            <div className="text-[11px] text-titan-metal/60 leading-relaxed mb-2">
              {p.description}
            </div>
            <div className="text-[9px] font-mono text-titan-metal/30 mb-2">
              hash: {p.payload_hash.slice(0, 16)}…{p.payload_hash.slice(-8)}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => openAction(p, 'approve')}
                className="flex-1 px-2 py-1 text-[11px] rounded bg-green-900/30 text-green-300
                           hover:bg-green-900/50 border border-green-500/30 transition-all"
              >
                Approve
              </button>
              <button
                onClick={() => openAction(p, 'decline')}
                className="flex-1 px-2 py-1 text-[11px] rounded bg-red-900/30 text-red-300
                           hover:bg-red-900/50 border border-red-500/30 transition-all"
              >
                Decline
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Recent responses */}
      {recent.length > 0 && (
        <>
          <div className="px-4 py-2 border-t border-titan-metal/10 text-[10px] uppercase tracking-wider text-titan-metal/40">
            Recent
          </div>
          <div className="px-4 py-2 space-y-2 pb-4">
            {recent.map((r) => (
              <div key={r.proposal_id} className="text-[10px] text-titan-metal/50">
                <span
                  className={
                    r.status === 'approved' ? 'text-green-400' : 'text-red-400'
                  }
                >
                  {r.status === 'approved' ? '✓' : '✗'}
                </span>{' '}
                <span className="text-titan-metal/70">{r.title}</span>
                <div className="text-[9px] italic text-titan-metal/40 ml-3">
                  &ldquo;{r.approval_reason || r.decline_reason}&rdquo;
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Dialogue history (Tier 3) */}
      <DialogueHistory />

      {/* Action modal */}
      {activeProposal && actionMode && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
          onClick={closeAction}
        >
          <div
            className="w-96 rounded-lg border border-titan-gold/30 bg-titan-card p-5"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="text-xs uppercase tracking-wider text-titan-gold mb-2">
              {actionMode === 'approve' ? 'Approve' : 'Decline'} proposal
            </div>
            <div className="text-sm font-semibold text-titan-steel mb-1">
              {activeProposal.title}
            </div>
            <div className="text-[11px] text-titan-metal/60 mb-3">
              {actionMode === 'approve'
                ? 'Tell Titan what makes this good. Your reason becomes part of his learning.'
                : 'Tell Titan what is wrong and why. Your reason becomes part of his learning.'}
            </div>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={4}
              minLength={10}
              placeholder={
                actionMode === 'approve'
                  ? 'Why is this a good proposal? (min 10 chars)'
                  : 'What is wrong with this proposal? (min 10 chars)'
              }
              className="w-full px-3 py-2 text-sm rounded bg-titan-bg/60 border border-titan-metal/20
                         text-titan-steel placeholder:text-titan-metal/30 focus:border-titan-gold/50
                         focus:outline-none resize-none"
            />
            <div className="text-[10px] text-titan-metal/40 mt-1">
              {reason.trim().length}/10 chars min
            </div>
            {actionMode === 'approve' && activeProposal.requires_signature && (
              <div className="text-[10px] text-solana-purple/70 mt-2">
                Privy will prompt you to sign the bundle hash with your wallet.
              </div>
            )}
            {submitMsg && (
              <div
                className={`text-[11px] mt-2 ${
                  submitMsg.startsWith('Error') || submitMsg.includes('must')
                    ? 'text-red-400'
                    : 'text-green-400'
                }`}
              >
                {submitMsg}
              </div>
            )}
            <div className="flex gap-2 mt-4">
              <button
                onClick={closeAction}
                disabled={submitting}
                className="flex-1 px-3 py-1.5 text-xs rounded border border-titan-metal/20
                           text-titan-metal/70 hover:bg-titan-bg/60 transition-all
                           disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={submitting || reason.trim().length < 10}
                className={`flex-1 px-3 py-1.5 text-xs rounded transition-all
                           disabled:opacity-50 disabled:cursor-not-allowed ${
                             actionMode === 'approve'
                               ? 'bg-green-900/40 text-green-200 hover:bg-green-900/60 border border-green-500/40'
                               : 'bg-red-900/40 text-red-200 hover:bg-red-900/60 border border-red-500/40'
                           }`}
              >
                {submitting ? 'Submitting…' : actionMode === 'approve' ? 'Sign & Approve' : 'Decline'}
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
