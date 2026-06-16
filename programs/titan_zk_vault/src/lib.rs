use anchor_lang::prelude::*;

// Required for LightHasher derive macro to resolve `::light_hasher::` paths
extern crate light_hasher;

use light_sdk::{
    account::LightAccount,
    address::v1::derive_address,
    cpi::{
        v1::{CpiAccounts, LightSystemProgramCpi},
        InvokeLightSystemProgram, LightCpiInstruction,
    },
    derive_light_cpi_signer,
    instruction::{
        account_meta::CompressedAccountMeta, PackedAddressTreeInfo, PackedAddressTreeInfoExt,
    },
    CpiSigner, LightDiscriminator, LightHasher,
};

declare_id!("52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw");

/// CPI signer PDA for Light Protocol interactions.
pub const LIGHT_CPI_SIGNER: CpiSigner =
    derive_light_cpi_signer!("52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw");

// ---------------------------------------------------------------------------
// Anchor-compatible proof wrapper
// ---------------------------------------------------------------------------

/// Groth16 compressed proof wrapper compatible with Anchor IDL generation.
/// Wraps the Light Protocol proof format (a: 32 bytes, b: 64 bytes, c: 32 bytes).
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct CompressedProofInput {
    pub a: [u8; 32],
    pub b: [u8; 64],
    pub c: [u8; 32],
}

impl Default for CompressedProofInput {
    fn default() -> Self {
        Self {
            a: [0u8; 32],
            b: [0u8; 64],
            c: [0u8; 32],
        }
    }
}

impl CompressedProofInput {
    /// Convert to Light SDK ValidityProof for CPI invocation.
    fn to_validity_proof(&self) -> light_sdk::instruction::ValidityProof {
        use light_sdk::instruction::{CompressedProof, ValidityProof};
        ValidityProof(Some(CompressedProof {
            a: self.a,
            b: self.b,
            c: self.c,
        }))
    }
}

/// Resolve an optional client-supplied proof into a Light `ValidityProof`.
///
/// `compress_memory_batch` / `append_epoch_snapshot` create **output-only,
/// addressless** compressed accounts (no input account, `new_init(.., None, ..)`).
/// The Light system program FORBIDS a proof in that case — passing `Some(..)`
/// (even an all-zero proof) is rejected with `ProofIsSome` (6018). So the proof
/// MUST be `None` for the output-only flow (`compress_memory_batch` /
/// `append_epoch_snapshot`); `Some(realProof)` is used by the input-consuming /
/// address-creating instructions `create_sovereign_state` (non-inclusion) and
/// `update_sovereign_state` (inclusion).
fn resolve_validity_proof(
    proof: Option<CompressedProofInput>,
) -> light_sdk::instruction::ValidityProof {
    match proof {
        Some(p) => p.to_validity_proof(),
        None => light_sdk::instruction::ValidityProof(None),
    }
}

/// Titan ZK-Vault — On-Chain Memory Anchor
///
/// V2.1: Added ZK-compressed batch receipts and epoch snapshots via Light Protocol.
///
/// Account space: 8 (discriminator) + 32 (authority) + 32 (latest_root)
///   + 8 (commit_count) + 8 (last_commit_ts) + 2 (sovereignty_index)
///   + 32 (shadow_url_hash) + 1 (bump) = 123 bytes
#[program]
pub mod titan_zk_vault {
    use super::*;

    /// Creates the Titan's vault PDA. Called once after Genesis Ceremony.
    pub fn initialize_vault(ctx: Context<InitializeVault>) -> Result<()> {
        let vault = &mut ctx.accounts.vault_state;
        vault.authority = ctx.accounts.authority.key();
        vault.latest_root = [0u8; 32];
        vault.commit_count = 0;
        vault.last_commit_ts = Clock::get()?.unix_timestamp;
        vault.sovereignty_index = 0;
        vault.shadow_url_hash = [0u8; 32];
        vault.bump = ctx.bumps.vault_state;

        msg!("Titan ZK-Vault: Initialized for {}", vault.authority);
        Ok(())
    }

    /// Commits a 32-byte Merkle State Root from a Meditation Epoch.
    pub fn commit_state(
        ctx: Context<CommitState>,
        state_root: [u8; 32],
        sovereignty_index: u16,
    ) -> Result<()> {
        let vault = &mut ctx.accounts.vault_state;
        vault.latest_root = state_root;
        vault.commit_count = vault.commit_count.checked_add(1).unwrap();
        vault.last_commit_ts = Clock::get()?.unix_timestamp;
        vault.sovereignty_index = sovereignty_index;

        msg!(
            "Titan ZK-Vault: Commit #{} | Sovereignty: {}bp",
            vault.commit_count,
            sovereignty_index,
        );
        Ok(())
    }

    /// Updates the Shadow Drive URL hash after a Rebirth (Greater Epoch).
    pub fn update_shadow_hash(
        ctx: Context<CommitState>,
        shadow_url_hash: [u8; 32],
    ) -> Result<()> {
        let vault = &mut ctx.accounts.vault_state;
        vault.shadow_url_hash = shadow_url_hash;

        msg!("Titan ZK-Vault: Shadow hash updated");
        Ok(())
    }

    /// Closes the vault and reclaims rent SOL back to the authority.
    pub fn close_vault(_ctx: Context<CloseVault>) -> Result<()> {
        msg!("Titan ZK-Vault: Vault closed, rent reclaimed");
        Ok(())
    }

    /// Creates a ZK-compressed receipt for a batch of memory nodes.
    ///
    /// Called during Meditation Epoch (6-hour cycle) to anchor a batch of
    /// memories on-chain via Light Protocol compressed accounts.
    pub fn compress_memory_batch<'info>(
        ctx: Context<'_, '_, '_, 'info, CompressMemoryBatch<'info>>,
        proof: Option<CompressedProofInput>,
        batch_root: [u8; 32],
        node_count: u16,
        epoch_id: u64,
        sovereignty_score: u16,
        output_tree_index: u8,
    ) -> Result<()> {
        let vault = &ctx.accounts.vault_state;
        require!(
            vault.authority == ctx.accounts.authority.key(),
            TitanError::UnauthorizedAuthority
        );

        let light_cpi_accounts = CpiAccounts::new(
            ctx.accounts.authority.as_ref(),
            ctx.remaining_accounts,
            crate::LIGHT_CPI_SIGNER,
        );

        let timestamp = Clock::get()?.unix_timestamp;

        let mut account = LightAccount::<CompressedMemoryBatch>::new_init(
            &crate::ID,
            None,
            output_tree_index,
        );

        account.authority = ctx.accounts.authority.key().to_bytes();
        account.epoch_id = epoch_id;
        account.timestamp = timestamp;
        account.sovereignty_score = sovereignty_score;
        account.batch_root = batch_root;
        account.node_count = node_count;

        let cpi = LightSystemProgramCpi::new_cpi(crate::LIGHT_CPI_SIGNER, resolve_validity_proof(proof))
            .with_light_account(account)
            .map_err(|_| TitanError::CompressedAccountError)?;
        cpi.invoke(light_cpi_accounts)
            .map_err(|_| TitanError::LightCpiInvokeError)?;

        msg!(
            "Titan ZK-Vault: Compressed batch | epoch={} | nodes={}",
            epoch_id,
            node_count,
        );
        Ok(())
    }

    /// Creates a ZK-compressed epoch snapshot for the Greater Epoch audit trail.
    ///
    /// Called during Rebirth (24-hour cycle) after Shadow Drive upload.
    /// Append-only — each Greater Epoch creates a new compressed account.
    pub fn append_epoch_snapshot<'info>(
        ctx: Context<'_, '_, '_, 'info, AppendEpochSnapshot<'info>>,
        proof: Option<CompressedProofInput>,
        state_root: [u8; 32],
        memory_count: u64,
        sovereignty_score: u16,
        shadow_url_hash: [u8; 32],
        output_tree_index: u8,
    ) -> Result<()> {
        let vault = &ctx.accounts.vault_state;
        require!(
            vault.authority == ctx.accounts.authority.key(),
            TitanError::UnauthorizedAuthority
        );

        let light_cpi_accounts = CpiAccounts::new(
            ctx.accounts.authority.as_ref(),
            ctx.remaining_accounts,
            crate::LIGHT_CPI_SIGNER,
        );

        let timestamp = Clock::get()?.unix_timestamp;
        let epoch_number = vault.commit_count;

        let mut account = LightAccount::<CompressedEpochSnapshot>::new_init(
            &crate::ID,
            None,
            output_tree_index,
        );

        account.authority = ctx.accounts.authority.key().to_bytes();
        account.epoch_number = epoch_number;
        account.state_root = state_root;
        account.memory_count = memory_count;
        account.sovereignty_score = sovereignty_score;
        account.shadow_url_hash = shadow_url_hash;
        account.timestamp = timestamp;

        let cpi = LightSystemProgramCpi::new_cpi(crate::LIGHT_CPI_SIGNER, resolve_validity_proof(proof))
            .with_light_account(account)
            .map_err(|_| TitanError::CompressedAccountError)?;
        cpi.invoke(light_cpi_accounts)
            .map_err(|_| TitanError::LightCpiInvokeError)?;

        msg!(
            "Titan ZK-Vault: Epoch snapshot | epoch={} | memories={}",
            epoch_number,
            memory_count,
        );
        Ok(())
    }

    /// Creates the Titan's single canonical sovereign-state compressed account (GENESIS).
    ///
    /// Addressed create: derives a deterministic Light address from the authority,
    /// registers it with a **non-inclusion validity proof** (the address must not
    /// already exist), and writes the first `SovereignState`. Fires once per Titan;
    /// every subsequent canonical write is `update_sovereign_state`. (E2, INV-ZKW-3.)
    pub fn create_sovereign_state<'info>(
        ctx: Context<'_, '_, '_, 'info, CreateSovereignState<'info>>,
        proof: Option<CompressedProofInput>,
        address_tree_info: PackedAddressTreeInfo,
        output_tree_index: u8,
        state_root: [u8; 32],
        memory_count: u64,
        sovereignty_score: u16,
        shadow_url_hash: [u8; 32],
    ) -> Result<()> {
        let vault = &ctx.accounts.vault_state;
        require!(
            vault.authority == ctx.accounts.authority.key(),
            TitanError::UnauthorizedAuthority
        );
        // INV-ZKW-3: the addressed create carries a real non-inclusion proof.
        require!(proof.is_some(), TitanError::ProofRequired);
        // epoch_number from the vault commit counter (monotonic; mirrors E1, no
        // client-invented value).
        let epoch_number = vault.commit_count;

        let light_cpi_accounts = CpiAccounts::new(
            ctx.accounts.authority.as_ref(),
            ctx.remaining_accounts,
            crate::LIGHT_CPI_SIGNER,
        );

        // Deterministic canonical address: one stable Light address per Titan.
        let (address, address_seed) = derive_address(
            &[b"sovereign_state", ctx.accounts.authority.key().as_ref()],
            &address_tree_info
                .get_tree_pubkey(&light_cpi_accounts)
                .map_err(|_| TitanError::CompressedAccountError)?,
            &crate::ID,
        );
        let new_address_params = address_tree_info.into_new_address_params_packed(address_seed);

        let timestamp = Clock::get()?.unix_timestamp;

        let mut account = LightAccount::<SovereignState>::new_init(
            &crate::ID,
            Some(address),
            output_tree_index,
        );
        account.authority = ctx.accounts.authority.key().to_bytes();
        account.epoch_number = epoch_number;
        account.state_root = state_root;
        account.memory_count = memory_count;
        account.sovereignty_score = sovereignty_score;
        account.shadow_url_hash = shadow_url_hash;
        account.timestamp = timestamp;

        let cpi =
            LightSystemProgramCpi::new_cpi(crate::LIGHT_CPI_SIGNER, resolve_validity_proof(proof))
                .with_light_account(account)
                .map_err(|_| TitanError::CompressedAccountError)?
                .with_new_addresses(&[new_address_params]);
        cpi.invoke(light_cpi_accounts)
            .map_err(|_| TitanError::LightCpiInvokeError)?;

        msg!(
            "Titan ZK-Vault: SovereignState created | epoch={} | memories={}",
            epoch_number,
            memory_count,
        );
        Ok(())
    }

    /// Updates the Titan's canonical sovereign-state account (CONSUME + RECREATE).
    ///
    /// `new_mut` nullifies the prior leaf and writes a new one; the Light system
    /// program verifies a Groth16 **inclusion** proof on-chain before accepting it.
    /// Every canonical sovereign-state write after genesis carries a real SNARK
    /// (INV-ZKW-2). `old_state` MUST equal the current on-chain account data — the
    /// program recomputes the input leaf hash from it.
    pub fn update_sovereign_state<'info>(
        ctx: Context<'_, '_, '_, 'info, UpdateSovereignState<'info>>,
        proof: Option<CompressedProofInput>,
        account_meta: CompressedAccountMeta,
        old_state: SovereignState,
        state_root: [u8; 32],
        memory_count: u64,
        sovereignty_score: u16,
        shadow_url_hash: [u8; 32],
    ) -> Result<()> {
        let vault = &ctx.accounts.vault_state;
        require!(
            vault.authority == ctx.accounts.authority.key(),
            TitanError::UnauthorizedAuthority
        );
        // INV-ZKW-2: every update carries a real inclusion proof.
        require!(proof.is_some(), TitanError::ProofRequired);
        // epoch_number advances with the vault commit counter (monotonic).
        let epoch_number = vault.commit_count;

        let light_cpi_accounts = CpiAccounts::new(
            ctx.accounts.authority.as_ref(),
            ctx.remaining_accounts,
            crate::LIGHT_CPI_SIGNER,
        );

        let timestamp = Clock::get()?.unix_timestamp;

        // 3rd arg = the OLD account DATA (the program re-hashes it to prove the input).
        let mut account = LightAccount::<SovereignState>::new_mut(
            &crate::ID,
            &account_meta,
            old_state,
        )
        .map_err(|_| TitanError::CompressedAccountError)?;

        // authority is immutable identity — left as carried in old_state.
        account.epoch_number = epoch_number;
        account.state_root = state_root;
        account.memory_count = memory_count;
        account.sovereignty_score = sovereignty_score;
        account.shadow_url_hash = shadow_url_hash;
        account.timestamp = timestamp;

        let cpi =
            LightSystemProgramCpi::new_cpi(crate::LIGHT_CPI_SIGNER, resolve_validity_proof(proof))
                .with_light_account(account)
                .map_err(|_| TitanError::CompressedAccountError)?;
        cpi.invoke(light_cpi_accounts)
            .map_err(|_| TitanError::LightCpiInvokeError)?;

        msg!(
            "Titan ZK-Vault: SovereignState updated | epoch={} | memories={}",
            epoch_number,
            memory_count,
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Account Contexts
// ---------------------------------------------------------------------------

#[derive(Accounts)]
pub struct InitializeVault<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + VaultState::INIT_SPACE,
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CommitState<'info> {
    #[account(
        mut,
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CloseVault<'info> {
    #[account(
        mut,
        close = authority,
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Context for compress_memory_batch — reads vault PDA to verify authority.
#[derive(Accounts)]
pub struct CompressMemoryBatch<'info> {
    #[account(
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Context for append_epoch_snapshot — reads vault PDA to verify authority.
#[derive(Accounts)]
pub struct AppendEpochSnapshot<'info> {
    #[account(
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Context for create_sovereign_state — reads vault PDA to verify authority.
#[derive(Accounts)]
pub struct CreateSovereignState<'info> {
    #[account(
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Context for update_sovereign_state — reads vault PDA to verify authority.
#[derive(Accounts)]
pub struct UpdateSovereignState<'info> {
    #[account(
        seeds = [b"titan_vault", authority.key().as_ref()],
        bump = vault_state.bump,
        has_one = authority,
    )]
    pub vault_state: Account<'info, VaultState>,

    #[account(mut)]
    pub authority: Signer<'info>,
}

// ---------------------------------------------------------------------------
// State — Uncompressed (traditional PDA)
// ---------------------------------------------------------------------------

#[account]
#[derive(InitSpace)]
pub struct VaultState {
    /// The Titan's soul keypair pubkey — permanent owner of this vault.
    pub authority: Pubkey,          // 32
    /// Latest Merkle root of the cognitive memory graph.
    pub latest_root: [u8; 32],     // 32
    /// Total number of state root commits (epoch counter).
    pub commit_count: u64,         // 8
    /// Unix timestamp of the last commit.
    pub last_commit_ts: i64,       // 8
    /// Sovereignty index in basis points (0–10000 = 0%–100%).
    pub sovereignty_index: u16,    // 2
    /// SHA-256 hash of the Shadow Drive archive URL.
    pub shadow_url_hash: [u8; 32], // 32
    /// PDA bump seed for deterministic derivation.
    pub bump: u8,                  // 1
}

// ---------------------------------------------------------------------------
// State — ZK-Compressed (Light Protocol)
// ---------------------------------------------------------------------------

/// A compressed receipt for a batch of memory nodes.
/// Created during each Meditation Epoch (6-hour cycle).
#[derive(
    Clone, Debug, Default, LightDiscriminator, LightHasher,
    AnchorSerialize, AnchorDeserialize,
)]
pub struct CompressedMemoryBatch {
    #[hash]
    pub authority: [u8; 32],        // 32 bytes — Titan's wallet
    pub epoch_id: u64,              // 8 bytes — meditation cycle number
    pub timestamp: i64,             // 8 bytes — Unix timestamp
    pub sovereignty_score: u16,     // 2 bytes — basis points at creation
    #[hash]
    pub batch_root: [u8; 32],       // 32 bytes — Merkle root of memory hashes
    pub node_count: u16,            // 2 bytes — how many memories in this batch
}

/// A compressed epoch snapshot for the Greater Epoch audit trail.
/// Created during each Rebirth (24-hour cycle).
#[derive(
    Clone, Debug, Default, LightDiscriminator, LightHasher,
    AnchorSerialize, AnchorDeserialize,
)]
pub struct CompressedEpochSnapshot {
    #[hash]
    pub authority: [u8; 32],        // 32 bytes
    pub epoch_number: u64,          // 8 bytes — Greater Epoch sequence
    #[hash]
    pub state_root: [u8; 32],       // 32 bytes — full cognitive state root
    pub memory_count: u64,          // 8 bytes — total memories at this point
    pub sovereignty_score: u16,     // 2 bytes
    #[hash]
    pub shadow_url_hash: [u8; 32],  // 32 bytes — Shadow Drive archive hash
    pub timestamp: i64,             // 8 bytes
}

/// The Titan's single canonical sovereign-state compressed account (E2).
///
/// ADDRESSED — one stable, deterministic Light address per Titan
/// (`derive_address([b"sovereign_state", authority], …)`), reusable as the
/// on-chain anchor for future Titan artifacts. MUTABLE — each daily backup
/// event consume+recreates it via `new_mut`, carrying a real Groth16 proof
/// (INV-ZKW-2). Decode-distinct discriminator from the append-only
/// `CompressedEpochSnapshot` audit trail (INV-ZKW-1).
#[derive(
    Clone, Debug, Default, LightDiscriminator, LightHasher,
    AnchorSerialize, AnchorDeserialize,
)]
pub struct SovereignState {
    #[hash]
    pub authority: [u8; 32],        // 32 bytes — Titan's wallet (immutable identity)
    pub epoch_number: u64,          // 8 bytes — Greater Epoch sequence
    #[hash]
    pub state_root: [u8; 32],       // 32 bytes — == backup event_merkle_root (INV-ZKW-4)
    pub memory_count: u64,          // 8 bytes — total memories at this point
    pub sovereignty_score: u16,     // 2 bytes — basis points
    #[hash]
    pub shadow_url_hash: [u8; 32],  // 32 bytes — Arweave archive hash
    pub timestamp: i64,             // 8 bytes
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[error_code]
pub enum TitanError {
    #[msg("Authority does not match vault owner")]
    UnauthorizedAuthority,
    #[msg("Failed to create or modify compressed account")]
    CompressedAccountError,
    #[msg("Light system program CPI invocation failed")]
    LightCpiInvokeError,
    #[msg("A validity proof is required for this instruction")]
    ProofRequired,
}
