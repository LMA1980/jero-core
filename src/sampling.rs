//! Shared deterministic & stochastic logit sampling and multi-stop decoding contracts.
//!
//! Provides [`SamplingParams`], [`TokenSampler`] (with `NaN`-safe greedy argmax and
//! portable `xoshiro256**` PRNG state checkpointable into `KvBlockMeta::sampler_rng_state`),
//! [`StopCriteria`], and [`StopReason`] for unified decoding-stage semantics across
//! `jero-engine`, `jero-diag`, and `jero-worker`.

use core::cmp::Ordering;

/// Canonical sampling configuration shared across engine, diagnostic, and IPC layers.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature (`<= 0.0` or non-finite enables deterministic greedy argmax).
    pub temperature: f32,
    /// Maximum number of highest-logit candidates retained (`0` disables top-k cutoff).
    pub top_k: u32,
    /// Cumulative nucleus probability threshold in `(0.0, 1.0)` (`>= 1.0` disables top-p).
    pub top_p: f32,
    /// Relative probability floor against the top token in `(0.0, 1.0)` (`<= 0.0` disables min-p).
    pub min_p: f32,
    /// Penalty factor applied once per unique token in the trailing `repeat_last_n` window
    /// (`1.0` disables repetition penalty).
    pub repeat_penalty: f32,
    /// Number of trailing history tokens considered for `repeat_penalty` (`0` disables).
    pub repeat_last_n: usize,
    /// Deterministic PRNG seed initializing `xoshiro256**` state.
    pub seed: u64,
}

impl SamplingParams {
    /// Returns deterministic greedy decoding parameters (`temperature = 0.0`, no penalties).
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Deterministic `SplitMix64` mixer for seeding `xoshiro256**` without all-zero fixed points.
const fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const fn seed_xoshiro256(seed: u64) -> [u64; 4] {
    let mut sm = seed;
    let s0 = splitmix64_step(&mut sm);
    let s1 = splitmix64_step(&mut sm);
    let s2 = splitmix64_step(&mut sm);
    let s3 = splitmix64_step(&mut sm);
    if s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0 {
        [1, 0, 0, 0]
    } else {
        [s0, s1, s2, s3]
    }
}

/// Deterministic and stochastic logit sampler backed by checkpointable `xoshiro256**` state.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSampler {
    params: SamplingParams,
    rng_state: [u64; 4],
    candidates: Vec<(u32, f32)>,
    probs: Vec<f64>,
    recent_scratch: Vec<u32>,
}

impl TokenSampler {
    /// Constructs a [`TokenSampler`] initialized from `params.seed`.
    #[must_use]
    pub fn new(params: SamplingParams) -> Self {
        Self {
            params,
            rng_state: seed_xoshiro256(params.seed),
            candidates: Vec::new(),
            probs: Vec::new(),
            recent_scratch: Vec::new(),
        }
    }

    /// Restores a [`TokenSampler`] from a checkpointed `[u64; 4]` RNG state
    /// (for example `KvBlockMeta::sampler_rng_state`).
    ///
    /// If `rng_state == [0; 4]` (invalid absorbing zero state), falls back to
    /// seeding from `params.seed`.
    #[must_use]
    pub fn from_rng_state(params: SamplingParams, rng_state: [u64; 4]) -> Self {
        let state = if rng_state == [0; 4] {
            seed_xoshiro256(params.seed)
        } else {
            rng_state
        };
        Self {
            params,
            rng_state: state,
            candidates: Vec::new(),
            probs: Vec::new(),
            recent_scratch: Vec::new(),
        }
    }

    /// Returns the active [`SamplingParams`].
    #[must_use]
    pub const fn params(&self) -> SamplingParams {
        self.params
    }

    /// Returns the current 256-bit `xoshiro256**` PRNG state for checkpointing.
    #[must_use]
    pub const fn rng_state(&self) -> [u64; 4] {
        self.rng_state
    }

    /// Replaces the internal `xoshiro256**` state (re-seeding from `params.seed` if all zero).
    pub fn set_rng_state(&mut self, rng_state: [u64; 4]) {
        self.rng_state = if rng_state == [0; 4] {
            seed_xoshiro256(self.params.seed)
        } else {
            rng_state
        };
    }

    /// Selects the lowest-index maximal finite logit in `logits`, ignoring `NaN` and `±inf`.
    ///
    /// Returns `None` when `logits` is empty or contains no finite entries.
    #[must_use]
    pub fn greedy_argmax(logits: &[f32]) -> Option<u32> {
        let mut best: Option<(usize, f32)> = None;
        for (idx, &score) in logits.iter().enumerate() {
            if !score.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((idx, score));
            }
        }
        best.and_then(|(idx, _)| u32::try_from(idx).ok())
    }

    /// Samples the next token ID from `logits`, applying repetition penalty against `history`.
    ///
    /// When `params.temperature <= 0.0` (or non-finite), runs deterministic `NaN`-safe greedy
    /// selection (with optional repetition penalty when `repeat_penalty != 1.0`) without
    /// consuming PRNG state. Otherwise runs `repeat_penalty -> top_k -> softmax -> min_p ->
    /// top_p -> categorical draw`, advancing `xoshiro256**` state by one step.
    ///
    /// Returns `None` if `logits` is empty or contains no finite candidates.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }

        let has_repeat_penalty = self.prepare_repeat_set(logits.len(), history);

        if !self.params.temperature.is_finite() || self.params.temperature <= 0.0 {
            if !has_repeat_penalty {
                return Self::greedy_argmax(logits);
            }
            let penalty = self.params.repeat_penalty;
            let mut best: Option<(usize, f32)> = None;
            for (idx, &raw_score) in logits.iter().enumerate() {
                if !raw_score.is_finite() {
                    continue;
                }
                let Ok(token_id) = u32::try_from(idx) else {
                    break;
                };
                let penalized = if self.recent_scratch.binary_search(&token_id).is_ok() {
                    apply_penalty(raw_score, penalty)
                } else {
                    raw_score
                };
                if !penalized.is_finite() {
                    continue;
                }
                if best.is_none_or(|(_, best_score)| penalized > best_score) {
                    best = Some((idx, penalized));
                }
            }
            return best.and_then(|(idx, _)| u32::try_from(idx).ok());
        }

        self.collect_penalized_candidates(logits, has_repeat_penalty);
        if self.candidates.is_empty() {
            return None;
        }

        // Sort descending by score; break exact ties deterministically by lowest token ID.
        self.candidates.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        if self.params.top_k > 0 {
            let k = usize::try_from(self.params.top_k).unwrap_or(usize::MAX);
            if self.candidates.len() > k {
                self.candidates.truncate(k);
            }
        }

        self.apply_softmax_and_probability_filters();

        let kept = self.probs.len();
        if kept == 0 {
            return None;
        }

        // Draw unconditionally on stochastic steps so RNG state progresses predictably.
        let uniform = self.next_unit_f64();
        let sum_w: f64 = self.probs.iter().sum();
        if !sum_w.is_finite() || sum_w <= 0.0 {
            return Some(self.candidates[0].0);
        }

        let target = uniform * sum_w;
        let mut cumulative = 0.0_f64;
        for (idx, &weight) in self.probs.iter().enumerate() {
            cumulative += weight;
            if cumulative > target {
                return Some(self.candidates[idx].0);
            }
        }
        Some(self.candidates[kept - 1].0)
    }

    fn prepare_repeat_set(&mut self, vocab_len: usize, history: &[u32]) -> bool {
        self.recent_scratch.clear();
        let penalty = self.params.repeat_penalty;
        if self.params.repeat_last_n == 0
            || history.is_empty()
            || !penalty.is_finite()
            || penalty <= 0.0
            || (penalty - 1.0).abs() <= f32::EPSILON
        {
            return false;
        }
        let start = history.len().saturating_sub(self.params.repeat_last_n);
        for &tok in &history[start..] {
            if usize::try_from(tok).is_ok_and(|idx| idx < vocab_len) {
                self.recent_scratch.push(tok);
            }
        }
        if self.recent_scratch.is_empty() {
            return false;
        }
        self.recent_scratch.sort_unstable();
        self.recent_scratch.dedup();
        true
    }

    fn collect_penalized_candidates(&mut self, logits: &[f32], has_repeat_penalty: bool) {
        self.candidates.clear();
        let penalty = self.params.repeat_penalty;
        for (idx, &raw_score) in logits.iter().enumerate() {
            if !raw_score.is_finite() {
                continue;
            }
            let Ok(token_id) = u32::try_from(idx) else {
                break;
            };
            let score =
                if has_repeat_penalty && self.recent_scratch.binary_search(&token_id).is_ok() {
                    apply_penalty(raw_score, penalty)
                } else {
                    raw_score
                };
            if score.is_finite() {
                self.candidates.push((token_id, score));
            }
        }
    }

    fn apply_softmax_and_probability_filters(&mut self) {
        self.probs.clear();
        let max_logit = f64::from(self.candidates[0].1);
        let temp = f64::from(self.params.temperature);

        let mut total = 0.0_f64;
        for &(_, logit) in &self.candidates {
            let scaled = (f64::from(logit) - max_logit) / temp;
            let weight = scaled.exp();
            self.probs.push(weight);
            total += weight;
        }

        if total > 0.0 && total.is_finite() {
            for p in &mut self.probs {
                *p /= total;
            }
        }

        // Apply min_p relative floor (`p >= p_max * min_p`), keeping at least the top token.
        let min_p = self.params.min_p;
        if min_p.is_finite() && min_p > 0.0 && !self.probs.is_empty() {
            let threshold = self.probs[0] * f64::from(min_p.min(1.0));
            let mut keep = 1_usize;
            while keep < self.probs.len() && self.probs[keep] >= threshold {
                keep += 1;
            }
            self.candidates.truncate(keep);
            self.probs.truncate(keep);
            renormalize(&mut self.probs);
        }

        // Apply top_p cumulative nucleus cutoff, keeping at least the top token.
        let top_p = self.params.top_p;
        if top_p.is_finite() && top_p > 0.0 && top_p < 1.0 && self.probs.len() > 1 {
            let cutoff = f64::from(top_p);
            let mut cumulative = 0.0_f64;
            let mut keep = 0_usize;
            for &p in &self.probs {
                cumulative += p;
                keep += 1;
                if cumulative >= cutoff {
                    break;
                }
            }
            keep = keep.max(1);
            self.candidates.truncate(keep);
            self.probs.truncate(keep);
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.rng_state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.rng_state[1] << 17;

        self.rng_state[2] ^= self.rng_state[0];
        self.rng_state[3] ^= self.rng_state[1];
        self.rng_state[1] ^= self.rng_state[2];
        self.rng_state[0] ^= self.rng_state[3];

        self.rng_state[2] ^= t;
        self.rng_state[3] = self.rng_state[3].rotate_left(45);

        result
    }

    fn next_unit_f64(&mut self) -> f64 {
        let upper_53 = self.next_u64() >> 11;
        #[expect(
            clippy::cast_precision_loss,
            reason = "upper 53 bits fit losslessly in f64 mantissa"
        )]
        let mantissa = upper_53 as f64;
        mantissa / 9_007_199_254_740_992.0_f64
    }
}

fn apply_penalty(score: f32, penalty: f32) -> f32 {
    if score > 0.0 {
        score / penalty
    } else {
        score * penalty
    }
}

fn renormalize(probs: &mut [f64]) {
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in probs {
            *p /= sum;
        }
    }
}

/// Reason a multi-step generation loop terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    /// Model emitted one of the configured EOS/EOT stop token IDs.
    EosToken(u32),
    /// Generated token count reached `max_tokens`.
    MaxTokens,
    /// Sequence cursor reached or saturated the configured `n_ctx` ceiling.
    ContextCeiling,
}

/// Stopping criteria evaluated after sampling each token in a decoding loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopCriteria<'a> {
    /// Model-specific EOS/EOT token IDs that terminate generation cleanly.
    pub stop_token_ids: &'a [u32],
    /// Maximum number of tokens permitted for this generation call.
    pub max_tokens: usize,
    /// Fixed context window capacity (`n_ctx`) configured for the placement.
    pub n_ctx: usize,
}

impl<'a> StopCriteria<'a> {
    /// Constructs a new [`StopCriteria`] bound to `stop_token_ids`, `max_tokens`, and `n_ctx`.
    #[must_use]
    pub const fn new(stop_token_ids: &'a [u32], max_tokens: usize, n_ctx: usize) -> Self {
        Self {
            stop_token_ids,
            max_tokens,
            n_ctx,
        }
    }

    /// Evaluates stopping conditions after emitting `token_id`, reaching `generated_count`
    /// emitted tokens and advancing the KV cursor to `next_pos`.
    ///
    /// Precedence:
    /// 1. Explicit EOS/EOT token match (`StopReason::EosToken(token_id)`)
    /// 2. Token generation cap (`generated_count >= self.max_tokens` -> [`StopReason::MaxTokens`])
    /// 3. Context ceiling (`next_pos >= self.n_ctx` -> [`StopReason::ContextCeiling`])
    #[must_use]
    pub fn evaluate(
        &self,
        token_id: u32,
        generated_count: usize,
        next_pos: usize,
    ) -> Option<StopReason> {
        if self.stop_token_ids.contains(&token_id) {
            return Some(StopReason::EosToken(token_id));
        }
        if generated_count >= self.max_tokens {
            return Some(StopReason::MaxTokens);
        }
        if next_pos >= self.n_ctx {
            return Some(StopReason::ContextCeiling);
        }
        None
    }

    /// Alias for [`Self::evaluate`], returning `Some(StopReason)` when decoding must halt.
    #[must_use]
    #[inline]
    pub fn check(
        &self,
        token_id: u32,
        generated_count: usize,
        next_pos: usize,
    ) -> Option<StopReason> {
        self.evaluate(token_id, generated_count, next_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::KvBlockMeta;

    #[test]
    fn greedy_sampler_is_nan_safe_and_preserves_lowest_tie_index() {
        let logits = [f32::NAN, -5.0, -1.0, -1.0, f32::INFINITY];
        let mut sampler = TokenSampler::new(SamplingParams::greedy());
        let initial_state = sampler.rng_state();
        assert_eq!(sampler.sample(&logits, &[]), Some(2));
        // Greedy path does not advance PRNG state when unpenalized.
        assert_eq!(sampler.rng_state(), initial_state);

        assert_eq!(TokenSampler::greedy_argmax(&[f32::NAN, f32::NAN]), None);
        assert_eq!(
            TokenSampler::greedy_argmax(&[f32::NEG_INFINITY, f32::INFINITY]),
            None
        );
        assert_eq!(TokenSampler::greedy_argmax(&[-0.0_f32, 0.0]), Some(0));
        assert_eq!(sampler.sample(&[], &[]), None);
    }

    #[test]
    fn repeat_penalty_applies_once_per_unique_recent_token_and_respects_window() {
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 2,
            ..SamplingParams::greedy()
        };
        let mut sampler = TokenSampler::new(params);

        // Logit 0 = 3.0, Logit 1 = 2.0.
        // Repeating token 0 twice inside the window applies /2.0 only once (3.0 -> 1.5 < 2.0).
        let logits = [3.0_f32, 2.0];
        assert_eq!(sampler.sample(&logits, &[0, 0]), Some(1));

        // Negative logits get multiplied by penalty (-1.0 * 2.0 = -2.0 < -1.5).
        let neg_logits = [-1.0_f32, -1.5];
        assert_eq!(sampler.sample(&neg_logits, &[0]), Some(1));

        // Token 0 outside trailing window of length 2 is NOT penalized.
        assert_eq!(sampler.sample(&logits, &[0, 99, 98]), Some(0));
    }

    #[test]
    fn seeded_stochastic_sampling_checkpoints_and_resumes_identically() {
        let params = SamplingParams {
            temperature: 0.85,
            top_k: 4,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 4,
            seed: 777_999,
        };
        let logits = [0.2_f32, 1.4, f32::NAN, 1.9, 1.1, f32::NEG_INFINITY, 0.7];

        // Baseline uninterrupted run of 12 steps.
        let mut baseline = TokenSampler::new(params);
        let mut baseline_seq = Vec::new();
        for step in 0..12 {
            let tok = baseline
                .sample(&logits, &baseline_seq)
                .expect("finite logits must sample");
            baseline_seq.push(tok);
            assert!(
                tok != 2 && tok != 5,
                "non-finite masked token {tok} sampled at step {step}"
            );
        }

        // Checkpoint after 5 steps into KvBlockMeta::sampler_rng_state and resume.
        let mut first_half = TokenSampler::new(params);
        let mut resumed_seq = Vec::new();
        for _ in 0..5 {
            resumed_seq.push(first_half.sample(&logits, &resumed_seq).unwrap());
        }

        let mut meta = KvBlockMeta::new(1, 5);
        meta.sampler_rng_state = first_half.rng_state();
        assert_ne!(meta.sampler_rng_state, [0; 4]);

        let mut resumed = TokenSampler::from_rng_state(params, meta.sampler_rng_state);
        for _ in 5..12 {
            resumed_seq.push(resumed.sample(&logits, &resumed_seq).unwrap());
        }

        assert_eq!(baseline_seq, resumed_seq);
        assert_eq!(baseline.rng_state(), resumed.rng_state());

        // Seed 0 avoids absorbing zero state and produces valid non-zero RNG words.
        let zero_seed = TokenSampler::new(SamplingParams { seed: 0, ..params });
        assert_ne!(zero_seed.rng_state(), [0; 4]);

        // Zero state passed to from_rng_state recovers safely from params.seed.
        let recovered = TokenSampler::from_rng_state(params, [0; 4]);
        assert_eq!(recovered.rng_state(), TokenSampler::new(params).rng_state());
    }

    #[test]
    fn top_k_min_p_and_top_p_filters_constrain_candidates_deterministically() {
        // top_k=1 selects the single highest candidate deterministically even at temp > 0.
        let mut top1 = TokenSampler::new(SamplingParams {
            temperature: 1.5,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 123,
        });
        for _ in 0..8 {
            assert_eq!(top1.sample(&[0.1, 4.2, 3.9, 1.0], &[]), Some(1));
        }

        // min_p = 0.5 with top logit = 10.0 and others = 0.0 (exp(-10) << 0.5) filters to token 0.
        let mut minp = TokenSampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.5,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 55,
        });
        for _ in 0..8 {
            assert_eq!(minp.sample(&[10.0, 0.0, 0.0, 0.0], &[]), Some(0));
        }

        // top_p = 0.5 with token 2 >> others keeps strictly token 2 in nucleus.
        let mut nucleus = TokenSampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 99,
        });
        for _ in 0..8 {
            assert_eq!(nucleus.sample(&[0.0, 0.0, 8.0, 0.0], &[]), Some(2));
        }
    }

    #[test]
    fn stop_criteria_evaluates_multi_eos_max_tokens_and_n_ctx_boundaries() {
        let stop_tokens = [2_u32, 128_009, 32_000];
        let criteria = StopCriteria::new(&stop_tokens, 16, 64);

        // Normal continuation below limits.
        assert_eq!(criteria.evaluate(42, 1, 10), None);
        assert_eq!(criteria.evaluate(42, 15, 63), None);
        assert_eq!(criteria.check(42, 15, 63), None);

        // Multi-EOS/EOT hits match regardless of token position/count, taking precedence.
        assert_eq!(
            criteria.evaluate(128_009, 1, 10),
            Some(StopReason::EosToken(128_009))
        );
        assert_eq!(
            criteria.check(128_009, 1, 10),
            Some(StopReason::EosToken(128_009))
        );
        assert_eq!(
            criteria.evaluate(32_000, 16, 64),
            Some(StopReason::EosToken(32_000))
        );

        // max_tokens boundary triggers when generated_count reaches max_tokens.
        assert_eq!(criteria.evaluate(42, 16, 20), Some(StopReason::MaxTokens));
        assert_eq!(criteria.check(42, 16, 20), Some(StopReason::MaxTokens));

        // n_ctx ceiling triggers when next_pos reaches n_ctx while still below max_tokens.
        assert_eq!(
            criteria.evaluate(42, 5, 64),
            Some(StopReason::ContextCeiling)
        );
        assert_eq!(criteria.check(42, 5, 64), Some(StopReason::ContextCeiling));
    }

    #[test]
    fn sampling_params_and_stop_reason_serde_roundtrip() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.92,
            min_p: 0.05,
            repeat_penalty: 1.15,
            repeat_last_n: 32,
            seed: 987_654_321,
        };
        let json = serde_json::to_string(&params).expect("serialize params");
        let parsed: SamplingParams = serde_json::from_str(&json).expect("deserialize params");
        assert_eq!(parsed, params);

        let reason = StopReason::EosToken(128_001);
        let r_json = serde_json::to_string(&reason).expect("serialize reason");
        let r_parsed: StopReason = serde_json::from_str(&r_json).expect("deserialize reason");
        assert_eq!(r_parsed, reason);
    }
}
