//! middle_path — Trinity middle-path loss function (shared).
//!
//! Faithful port of `titan_hcl/logic/middle_path.py:51-107`. Used as the
//! reward signal for both filter_down tiers (unified `FilterDownV5Engine` +
//! per-half `SmallFilterDownEngine`): `reward = −middle_path_loss(...)`.
//!
//! - `layer_loss(tensor)` = L2 distance from center (0.5).
//! - `middle_path_loss(body, mind, spirit, weights)` = weighted average of
//!   layer_losses, normalized to [0, 1] by the theoretical max for 5-dim
//!   tensors (sqrt(5 × 0.25) ≈ 1.118).
//! - Default weights (1.0, 1.0, 1.2) — spirit weighted 20% higher per
//!   `titan_params.toml [middle_path]`.

/// Center value — perfect equilibrium per `middle_path.py:35`.
pub const CENTER: f64 = 0.5;
/// Default body weight in middle-path loss.
pub const WEIGHT_BODY: f64 = 1.0;
/// Default mind weight.
pub const WEIGHT_MIND: f64 = 1.0;
/// Default spirit weight (slightly elevated per Python).
pub const WEIGHT_SPIRIT: f64 = 1.2;

/// L2 distance from center for a single layer's tensor. Returns ≥ 0.0;
/// 0.0 = perfect equilibrium. Max for a 5-dim tensor at extremes ≈ 1.118.
pub fn layer_loss(tensor: &[f64]) -> f64 {
    tensor
        .iter()
        .map(|v| (v - CENTER).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Combined weighted middle-path loss across the trinity. Returns
/// normalized value in `[0, 1]` (0.0 = perfect equilibrium, 1.0 = max
/// distress).
pub fn middle_path_loss(body: &[f64], mind: &[f64], spirit: &[f64]) -> f64 {
    middle_path_loss_with_weights(body, mind, spirit, WEIGHT_BODY, WEIGHT_MIND, WEIGHT_SPIRIT)
}

/// Same as [`middle_path_loss`] but with explicit weights.
pub fn middle_path_loss_with_weights(
    body: &[f64],
    mind: &[f64],
    spirit: &[f64],
    wb: f64,
    wm: f64,
    ws: f64,
) -> f64 {
    let total_weight = wb + wm + ws;
    let body_l = layer_loss(body);
    let mind_l = layer_loss(mind);
    let spirit_l = layer_loss(spirit);
    // Theoretical max L2 for a 5-dim tensor at all-0 or all-1.
    let max_l2 = (5.0_f64 * 0.25_f64).sqrt(); // ~1.118
    let raw = (wb * body_l + wm * mind_l + ws * spirit_l) / total_weight;
    1.0_f64.min(raw / max_l2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_loss_zero_at_center() {
        let v = [CENTER; 5];
        assert_eq!(layer_loss(&v), 0.0);
    }

    #[test]
    fn layer_loss_max_at_extreme() {
        let v0 = [0.0_f64; 5];
        let v1 = [1.0_f64; 5];
        let max_l2 = (5.0_f64 * 0.25_f64).sqrt();
        assert!((layer_loss(&v0) - max_l2).abs() < 1e-12);
        assert!((layer_loss(&v1) - max_l2).abs() < 1e-12);
    }

    #[test]
    fn middle_path_loss_zero_at_center() {
        let body = [CENTER; 5];
        let mind = [CENTER; 5];
        let spirit = [CENTER; 5];
        assert_eq!(middle_path_loss(&body, &mind, &spirit), 0.0);
    }

    #[test]
    fn middle_path_loss_clamped_at_one() {
        let body = [0.0_f64; 5];
        let mind = [1.0_f64; 5];
        let spirit = [0.0_f64; 5];
        let loss = middle_path_loss(&body, &mind, &spirit);
        assert!((loss - 1.0).abs() < 1e-12, "loss {loss} should be 1.0");
    }
}
