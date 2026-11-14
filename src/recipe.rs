//! Single-prompt execution context and architecture recipe traits (ADR-005, ADR-023, ADR-024).

use core::ops::Range;

use crate::error::K3Error;
use crate::gguf::ModelMetadata;
use crate::tensor::{ModelWeights, QuantType};

/// Canonical layout dimensions for a contiguous `[layer][K|V][pos][kv_dim]` KV cache buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvGeometry {
    /// Number of transformer layers owning KV cache planes.
    pub num_layers: usize,
    /// Number of key/value attention heads per layer.
    pub n_heads_kv: usize,
    /// Scalar dimension per key/value attention head.
    pub head_dim: usize,
    /// Maximum sequence capacity (`n_ctx`) along the sequence position axis.
    pub seq_capacity: usize,
    /// Optional sliding-window span limiting readable attention history.
    pub sliding_window: Option<usize>,
}

impl KvGeometry {
    /// Validates and constructs a [`KvGeometry`] descriptor.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `num_layers`, `n_heads_kv`, `head_dim`,
    /// or `seq_capacity` is zero, if `sliding_window == Some(0)`, or if the required buffer
    /// element count overflows `usize`.
    pub fn new(
        num_layers: usize,
        n_heads_kv: usize,
        head_dim: usize,
        seq_capacity: usize,
        sliding_window: Option<usize>,
    ) -> Result<Self, K3Error> {
        let geom = Self {
            num_layers,
            n_heads_kv,
            head_dim,
            seq_capacity,
            sliding_window,
        };
        geom.required_f32_len()?;
        Ok(geom)
    }

    /// Returns the per-position vector width `n_heads_kv * head_dim`, checked against overflow.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if any dimension is zero or overflows `usize`.
    pub fn kv_dim(&self) -> Result<usize, K3Error> {
        if self.num_layers == 0
            || self.n_heads_kv == 0
            || self.head_dim == 0
            || self.seq_capacity == 0
            || self.sliding_window == Some(0)
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "KvGeometry dimensions and optional sliding_window must be non-zero"
                    .to_owned(),
            });
        }
        self.n_heads_kv
            .checked_mul(self.head_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: "KvGeometry kv_dim overflowed usize".to_owned(),
            })
    }

    /// Returns the number of `f32` elements in a single K or V plane (`seq_capacity * kv_dim`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on zero dimensions or `usize` overflow.
    pub fn plane_len(&self) -> Result<usize, K3Error> {
        let kv_dim = self.kv_dim()?;
        self.seq_capacity
            .checked_mul(kv_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: "KvGeometry plane_len overflowed usize".to_owned(),
            })
    }

    /// Returns the exact total `f32` element count (`num_layers * 2 * seq_capacity * kv_dim`).
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] on zero dimensions or `usize` overflow.
    pub fn required_f32_len(&self) -> Result<usize, K3Error> {
        let plane = self.plane_len()?;
        let layer_len = plane
            .checked_mul(2)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: "KvGeometry layer_len overflowed usize".to_owned(),
            })?;
        self.num_layers
            .checked_mul(layer_len)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: 0,
                reason: "KvGeometry required_f32_len overflowed usize".to_owned(),
            })
    }
}

/// Bounds-checked mutable view over a contiguous `[layer][K|V][pos][kv_dim]` KV cache buffer.
#[derive(Debug)]
pub struct KvCacheView<'a> {
    buffer: &'a mut [f32],
    geometry: KvGeometry,
    kv_dim: usize,
    plane_len: usize,
}

impl<'a> KvCacheView<'a> {
    /// Creates a validated [`KvCacheView`] over `buffer` matching `geometry`.
    ///
    /// If `buffer` is larger than `geometry.required_f32_len()`, the view borrows only the exact
    /// canonical prefix `[0..required_f32_len]`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `geometry` is invalid or `buffer.len()` is
    /// smaller than the required element count.
    pub fn new(buffer: &'a mut [f32], geometry: KvGeometry) -> Result<Self, K3Error> {
        let kv_dim = geometry.kv_dim()?;
        let plane_len = geometry.plane_len()?;
        let required = geometry.required_f32_len()?;
        if buffer.len() < required {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: required.saturating_mul(size_of::<f32>()),
                buffer_len: size_of_val(buffer),
                reason: format!(
                    "kv_cache buffer length {} is smaller than required {} f32 elements",
                    buffer.len(),
                    required
                ),
            });
        }
        Ok(Self {
            buffer: &mut buffer[..required],
            geometry,
            kv_dim,
            plane_len,
        })
    }

    /// Returns the underlying [`KvGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    /// Returns the maximum sequence token capacity (`seq_capacity`) of this KV cache view.
    #[must_use]
    pub const fn seq_capacity(&self) -> usize {
        self.geometry.seq_capacity
    }

    /// Returns the per-position vector width `n_heads_kv * head_dim`.
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Returns the length of a single K or V plane (`seq_capacity * kv_dim`).
    #[must_use]
    pub const fn plane_len(&self) -> usize {
        self.plane_len
    }

    fn layer_offsets(&self, layer: usize) -> Result<(usize, usize, usize), K3Error> {
        if layer >= self.geometry.num_layers {
            return Err(K3Error::TensorBoundsError {
                tensor_name: format!("kv_cache.layer.{layer}"),
                offset: 0,
                byte_len: self.plane_len * 2 * size_of::<f32>(),
                buffer_len: size_of_val(self.buffer),
                reason: format!(
                    "layer index {layer} out of bounds for num_layers={}",
                    self.geometry.num_layers
                ),
            });
        }
        let k_start = layer * 2 * self.plane_len;
        let v_start = k_start + self.plane_len;
        let layer_end = v_start + self.plane_len;
        Ok((k_start, v_start, layer_end))
    }

    /// Returns immutable `(k_plane, v_plane)` slices for `layer`, each of length `seq_capacity * kv_dim`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `layer >= self.geometry().num_layers`.
    pub fn layer_kv(&self, layer: usize) -> Result<(&[f32], &[f32]), K3Error> {
        let (k_start, v_start, layer_end) = self.layer_offsets(layer)?;
        Ok((
            &self.buffer[k_start..v_start],
            &self.buffer[v_start..layer_end],
        ))
    }

    /// Returns simultaneous mutable `(k_plane, v_plane)` slices for `layer`, each of length
    /// `seq_capacity * kv_dim`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `layer >= self.geometry().num_layers`.
    pub fn layer_kv_mut(&mut self, layer: usize) -> Result<(&mut [f32], &mut [f32]), K3Error> {
        let (k_start, _, _) = self.layer_offsets(layer)?;
        let (_, layer_tail) = self.buffer.split_at_mut(k_start);
        let (k_plane, v_tail) = layer_tail.split_at_mut(self.plane_len);
        let (v_plane, _) = v_tail.split_at_mut(self.plane_len);
        Ok((k_plane, v_plane))
    }

    /// Returns mutable `(k_slot, v_slot)` slices of length `kv_dim` for writing sequence position `pos`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `layer >= num_layers`, or
    /// [`K3Error::ContextLengthExceeded`] if `pos >= seq_capacity`.
    pub fn write_pos_range(
        &mut self,
        layer: usize,
        pos: usize,
    ) -> Result<(&mut [f32], &mut [f32]), K3Error> {
        if pos >= self.geometry.seq_capacity {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: pos.saturating_add(1),
                max_context_tokens: self.geometry.seq_capacity,
            });
        }
        let kv_dim = self.kv_dim;
        let offset = pos * kv_dim;
        let end = offset + kv_dim;
        let (k_plane, v_plane) = self.layer_kv_mut(layer)?;
        Ok((&mut k_plane[offset..end], &mut v_plane[offset..end]))
    }

    /// Returns the valid attention token index range up to `end_pos`, clamped by `sliding_window`
    /// if configured (`[max(0, end_pos - window)..end_pos]`).
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `end_pos > seq_capacity`.
    pub const fn valid_token_range(&self, end_pos: usize) -> Result<Range<usize>, K3Error> {
        if end_pos > self.geometry.seq_capacity {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: self.geometry.seq_capacity,
            });
        }
        let start_pos = match self.geometry.sliding_window {
            Some(window) => end_pos.saturating_sub(window),
            None => 0,
        };
        Ok(start_pos..end_pos)
    }

    /// Returns immutable `(k_valid, v_valid)` slices covering the active attention history up to
    /// `end_pos` (respecting `sliding_window` if configured) for `layer`.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `end_pos > seq_capacity`, or
    /// [`K3Error::TensorBoundsError`] if `layer >= num_layers`.
    pub fn valid_kv_range(
        &self,
        layer: usize,
        end_pos: usize,
    ) -> Result<(&[f32], &[f32]), K3Error> {
        let token_range = self.valid_token_range(end_pos)?;
        let (k_plane, v_plane) = self.layer_kv(layer)?;
        let elem_start = token_range.start * self.kv_dim;
        let elem_end = token_range.end * self.kv_dim;
        Ok((
            &k_plane[elem_start..elem_end],
            &v_plane[elem_start..elem_end],
        ))
    }
}

/// Single-prompt execution context enforcing strict cycle exclusivity (no cross-prompt batching
/// per ADR-005) and fixed `n_ctx` context window bounds (per ADR-027).
#[derive(Debug)]
pub struct ExecutionContext<'a> {
    prompt_tokens: &'a [u32],
    decode_token: Option<u32>,
    jump_forward_tokens: Option<&'a [u32]>,
    position: usize,
    n_ctx: usize,
    compute_logits: bool,
    logits: &'a mut [f32],
    kv_cache: &'a mut [f32],
    kv_geometry: Option<KvGeometry>,
}

impl<'a> ExecutionContext<'a> {
    /// Creates a new single-prompt [`ExecutionContext`] validated against `n_ctx`.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `prompt_tokens.len() > n_ctx` or `n_ctx == 0`.
    pub fn new(
        prompt_tokens: &'a [u32],
        n_ctx: usize,
        logits: &'a mut [f32],
        kv_cache: &'a mut [f32],
    ) -> Result<Self, K3Error> {
        if n_ctx == 0 || prompt_tokens.len() > n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: prompt_tokens.len(),
                max_context_tokens: n_ctx,
            });
        }

        Ok(Self {
            prompt_tokens,
            decode_token: None,
            jump_forward_tokens: None,
            position: 0,
            n_ctx,
            compute_logits: true,
            logits,
            kv_cache,
            kv_geometry: None,
        })
    }

    /// Creates a zero-offset single-prompt prefill [`ExecutionContext`] expecting output logits,
    /// delegating to [`Self::new_prefill_chunk`] with `start_position = 0` and
    /// `compute_logits = true`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `prompt_tokens` or `logits` is empty, or
    /// [`K3Error::ContextLengthExceeded`] if `n_ctx == 0` or `prompt_tokens.len() > n_ctx`.
    pub fn new_prefill(
        prompt_tokens: &'a [u32],
        n_ctx: usize,
        logits: &'a mut [f32],
        kv_cache: &'a mut [f32],
    ) -> Result<Self, K3Error> {
        Self::new_prefill_chunk(prompt_tokens, n_ctx, 0, true, logits, kv_cache)
    }

    /// Creates a bounded prefill chunk [`ExecutionContext`] over `[start_position..end_position]`
    /// aligned with [`crate::ipc::TokenInputSlotHeader::start_position`].
    ///
    /// Intermediate chunks may set `compute_logits = false` (allowing empty `logits`) so recipes
    /// skip the vocabulary LM-head projection until the final prompt chunk.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `prompt_tokens` is empty or if
    /// `compute_logits == true` with an empty `logits` buffer, and returns
    /// [`K3Error::ContextLengthExceeded`] if `n_ctx == 0`, `start_position + prompt_tokens.len()`
    /// overflows `usize`, or the chunk end position exceeds `n_ctx`.
    pub fn new_prefill_chunk(
        prompt_tokens: &'a [u32],
        n_ctx: usize,
        start_position: usize,
        compute_logits: bool,
        logits: &'a mut [f32],
        kv_cache: &'a mut [f32],
    ) -> Result<Self, K3Error> {
        if n_ctx == 0 {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: prompt_tokens.len(),
                max_context_tokens: 0,
            });
        }
        if prompt_tokens.is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "prompt_tokens".to_owned(),
                offset: start_position,
                byte_len: 0,
                buffer_len: 0,
                reason: "prefill chunk requires at least one token".to_owned(),
            });
        }
        let end_pos = validate_context_range(start_position, prompt_tokens.len(), n_ctx)?;
        if compute_logits && logits.is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "logits".to_owned(),
                offset: end_pos,
                byte_len: size_of::<f32>(),
                buffer_len: 0,
                reason: "logits buffer must be non-empty when compute_logits is enabled".to_owned(),
            });
        }

        Ok(Self {
            prompt_tokens,
            decode_token: None,
            jump_forward_tokens: None,
            position: start_position,
            n_ctx,
            compute_logits,
            logits,
            kv_cache,
            kv_geometry: None,
        })
    }

    const fn effective_context_limit(&self) -> usize {
        match self.kv_geometry {
            Some(geom) if geom.seq_capacity < self.n_ctx => geom.seq_capacity,
            _ => self.n_ctx,
        }
    }

    /// Attaches a canonical [`KvGeometry`] layout descriptor and verifies the backing `kv_cache`
    /// buffer is large enough to hold `[num_layers][K|V][seq_capacity][n_heads_kv * head_dim]`
    /// and that the active chunk end position fits within `kv_geometry.seq_capacity`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `kv_geometry` has zero/overflowing dimensions or
    /// if `self.kv_cache().len()` is smaller than `kv_geometry.required_f32_len()`, or
    /// [`K3Error::ContextLengthExceeded`] if `self.end_position() > kv_geometry.seq_capacity`.
    pub fn with_kv_geometry(mut self, kv_geometry: KvGeometry) -> Result<Self, K3Error> {
        let required = kv_geometry.required_f32_len()?;
        if self.kv_cache.len() < required {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "kv_cache".to_owned(),
                offset: 0,
                byte_len: required.saturating_mul(size_of::<f32>()),
                buffer_len: size_of_val(self.kv_cache),
                reason: format!(
                    "kv_cache buffer length {} is smaller than required {} f32 elements",
                    self.kv_cache.len(),
                    required
                ),
            });
        }
        validate_context_range(
            self.position,
            self.prompt_tokens.len(),
            kv_geometry.seq_capacity,
        )?;
        self.kv_geometry = Some(kv_geometry);
        Ok(self)
    }

    /// Sets the starting KV sequence position cursor for this [`ExecutionContext`].
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `position + self.prompt_tokens.len()` exceeds
    /// `self.n_ctx` or attached `KvGeometry::seq_capacity`.
    pub fn with_position(mut self, position: usize) -> Result<Self, K3Error> {
        validate_context_range(
            position,
            self.prompt_tokens.len(),
            self.effective_context_limit(),
        )?;
        self.position = position;
        Ok(self)
    }

    /// Returns the input prompt (or chunk) token sequence for the single active evaluation slot.
    #[must_use]
    pub const fn prompt_tokens(&self) -> &'a [u32] {
        self.prompt_tokens
    }

    /// Returns the number of tokens in the active prompt or chunk (`self.prompt_tokens().len()`).
    #[must_use]
    pub const fn batch_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    /// Returns the current token ID scheduled for the next single-token [`ArchRecipe::decode_step`].
    #[must_use]
    pub const fn decode_token(&self) -> Option<u32> {
        self.decode_token
    }

    /// Returns the current KV sequence position cursor (`start_position` at prefill chunk entry).
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the starting KV sequence position cursor for the current prefill chunk
    /// (alias for [`Self::position`]).
    #[must_use]
    pub const fn start_position(&self) -> usize {
        self.position
    }

    /// Returns the exclusive ending KV sequence position for the current chunk
    /// (`start_position + prompt_tokens.len()`).
    #[must_use]
    pub const fn end_position(&self) -> usize {
        self.position.saturating_add(self.prompt_tokens.len())
    }

    /// Returns the exclusive ending KV sequence position for the current chunk
    /// (alias for [`Self::end_position`]).
    #[must_use]
    pub const fn batch_end_position(&self) -> usize {
        self.end_position()
    }

    /// Returns `true` when the active step must materialize final vocabulary logits into `logits`.
    #[must_use]
    pub const fn compute_logits(&self) -> bool {
        self.compute_logits
    }

    /// Returns `true` when the active step must materialize final vocabulary logits into `logits`
    /// (alias for [`Self::compute_logits`]).
    #[must_use]
    pub const fn needs_logits(&self) -> bool {
        self.compute_logits
    }

    /// Returns the fixed maximum context length (`n_ctx`) for this execution slot.
    #[must_use]
    pub const fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Returns an immutable view of the output logits buffer.
    #[must_use]
    pub const fn logits(&self) -> &[f32] {
        self.logits
    }

    /// Returns a mutable view of the output logits buffer.
    pub const fn logits_mut(&mut self) -> &mut [f32] {
        self.logits
    }

    /// Returns an immutable view of the single-slot KV cache buffer.
    #[must_use]
    pub const fn kv_cache(&self) -> &[f32] {
        self.kv_cache
    }

    /// Returns a mutable view of the single-slot KV cache buffer.
    pub const fn kv_cache_mut(&mut self) -> &mut [f32] {
        self.kv_cache
    }

    /// Returns the staged forced token sequence for a jump-forward prefill burst, if any.
    #[must_use]
    pub const fn jump_forward_tokens(&self) -> Option<&'a [u32]> {
        self.jump_forward_tokens
    }

    /// Returns simultaneous mutable views of the logits and KV cache buffers.
    pub fn buffers_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (self.logits, self.kv_cache)
    }

    /// Returns the configured [`KvGeometry`], if attached via [`Self::with_kv_geometry`].
    #[must_use]
    pub const fn kv_geometry(&self) -> Option<KvGeometry> {
        self.kv_geometry
    }

    /// Returns a bounds-checked [`KvCacheView`] over the single-slot KV cache buffer when
    /// [`Self::with_kv_geometry`] has been configured.
    pub fn kv_view_mut(&mut self) -> Option<KvCacheView<'_>> {
        let geom = self.kv_geometry?;
        KvCacheView::new(self.kv_cache, geom).ok()
    }

    /// Returns simultaneous mutable views of the logits buffer and optional [`KvCacheView`].
    pub fn logits_and_kv_view_mut(&mut self) -> (&mut [f32], Option<KvCacheView<'_>>) {
        let kv_view = self
            .kv_geometry
            .and_then(|geom| KvCacheView::new(self.kv_cache, geom).ok());
        (self.logits, kv_view)
    }

    /// Stages the next generated token for a single-token decode pass, verifying `position < n_ctx`.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + 1` exceeds `self.n_ctx` or
    /// attached `KvGeometry::seq_capacity`.
    pub fn prepare_decode(&mut self, token_id: u32) -> Result<(), K3Error> {
        validate_context_range(self.position, 1, self.effective_context_limit())?;
        self.jump_forward_tokens = None;
        self.compute_logits = true;
        self.decode_token = Some(token_id);
        Ok(())
    }

    /// Stages a deterministic `forced_tokens` slice for an atomic jump-forward prefill burst.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + forced_tokens.len()` exceeds
    /// `self.n_ctx` or attached `KvGeometry::seq_capacity`.
    pub fn prepare_jump_forward(&mut self, forced_tokens: &'a [u32]) -> Result<(), K3Error> {
        validate_context_range(
            self.position,
            forced_tokens.len(),
            self.effective_context_limit(),
        )?;
        self.decode_token = None;
        self.jump_forward_tokens = Some(forced_tokens);
        Ok(())
    }

    /// Clears any staged jump-forward forced token sequence.
    pub fn clear_jump_forward(&mut self) {
        self.jump_forward_tokens = None;
    }

    /// Advances the KV sequence position cursor by `token_count` tokens after a forward step.
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] if `self.position + token_count` exceeds
    /// `self.n_ctx` or attached `KvGeometry::seq_capacity`.
    pub fn advance_position(&mut self, token_count: usize) -> Result<usize, K3Error> {
        let new_pos =
            validate_context_range(self.position, token_count, self.effective_context_limit())?;
        self.position = new_pos;
        Ok(self.position)
    }
}

const fn validate_context_range(
    start_position: usize,
    token_count: usize,
    limit: usize,
) -> Result<usize, K3Error> {
    let Some(end_pos) = start_position.checked_add(token_count) else {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: usize::MAX,
            max_context_tokens: limit,
        });
    };
    if end_pos > limit {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: end_pos,
            max_context_tokens: limit,
        });
    }
    Ok(end_pos)
}

/// Architecture recipe trait compiled in-process per ADR-023 and ADR-024.
///
/// Each recipe maps a specific `general.architecture` identifier (treating minor versions
/// such as `qwen3` vs. `qwen3.8` as distinct algorithms) to single-prompt prefill and decode steps.
pub trait ArchRecipe: Send + Sync + core::fmt::Debug {
    /// Returns the exact GGUF `general.architecture` identifier handled by this recipe.
    fn architecture_id(&self) -> &'static str;

    /// Returns the exact recipe revision string (e.g., `"1.0.0"`) per ADR-023.
    fn recipe_version(&self) -> &'static str {
        "1.0.0"
    }

    /// Returns the supported target weight layouts (e.g., `&["native", "ime_tile_8x16"]`) per ADR-037.
    fn target_layouts(&self) -> &'static [&'static str] {
        &["native"]
    }

    /// Returns `true` if this recipe maintains an autoregressive KV cache across decode steps.
    ///
    /// Zero-KV encoder architectures (such as `ModernBERT`, `DeBERTa-v3`, and `bert`) override
    /// this to return `false` so admission and arena setup skip KV page reservation.
    fn requires_kv_cache(&self) -> bool {
        true
    }

    /// Returns `true` if this recipe and its linked kernel backend support `quant`.
    fn supports_quant(&self, quant: QuantType) -> bool;

    /// Validates per-tensor quantization admission and resident weight invariants before execution.
    ///
    /// Checks every tensor in `metadata.tensors()` (and any resident weight tensors resolvable
    /// when `weights.tensor_count() > 0`) against [`Self::supports_quant`], returning
    /// [`K3Error::CapabilityMismatch`] naming the tensor and unsupported [`QuantType`].
    ///
    /// # Fail-Closed Weight Invariant
    /// Recipes validated against loaded model weights (`weights.tensor_count() > 0`) **must**
    /// fail closed on missing, mismatched, or unsupported tensors and must **never** silently
    /// synthesize fallback weights or identity projections to keep running.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any tensor uses an unsupported quantization format.
    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0
                && let Ok(view) = weights.tensor(desc.name())
                && !self.supports_quant(view.quant())
            {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    view.name(),
                    view.quant(),
                ));
            }
        }
        Ok(())
    }

    /// Executes a single-pass classification or logit-harvest pass over [`ExecutionContext::prompt_tokens`].
    ///
    /// Defaults to delegating to [`ArchRecipe::prefill_step`]. Encoder-only recipes override this
    /// to compute bidirectional embeddings, pooling, and classification logits without a KV cache.
    ///
    /// # Errors
    /// Returns [`K3Error`] if weight lookup, buffer validation, or forward execution fails.
    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.prefill_step(ctx, weights)
    }

    /// Executes the prompt or bounded chunk prefill forward pass over
    /// [`ExecutionContext::prompt_tokens`] starting at [`ExecutionContext::start_position`] for a
    /// single prompt (ADR-005).
    ///
    /// Populates KV cache positions `[start_position..end_position)` and writes next-token logits
    /// when [`ExecutionContext::compute_logits`] (`needs_logits`) is `true` (skipping LM-head
    /// projection on intermediate prefill chunks where `compute_logits` is `false`).
    ///
    /// # Errors
    /// Returns [`K3Error`] if a required tensor is missing, has an unsupported quantization
    /// layout, or exceeds context window limits.
    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error>;

    /// Executes a single-token autoregressive decode step using [`ExecutionContext::decode_token`],
    /// updating the KV cache at [`ExecutionContext::position`] and writing next-token logits.
    ///
    /// # Errors
    /// Returns [`K3Error`] if no decode token is staged, context bounds are exceeded, or weight
    /// lookup fails.
    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenInputSlotHeader;
    use crate::gguf::{GgufFile, GgufValue, test_helpers::create_synthetic_gguf_v3_with_kvs};
    use crate::tensor::TensorView;

    #[derive(Debug)]
    struct Q4OnlyMockRecipe;

    impl ArchRecipe for Q4OnlyMockRecipe {
        fn architecture_id(&self) -> &'static str {
            "llama"
        }

        fn supports_quant(&self, quant: QuantType) -> bool {
            matches!(quant, QuantType::F32 | QuantType::Q4_0)
        }

        fn prefill_step(
            &self,
            _ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            Ok(())
        }

        fn decode_step(
            &self,
            _ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            Ok(())
        }
    }

    struct EmptyMockWeights;

    impl ModelWeights for EmptyMockWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "empty mock".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_validate_model_rejects_mixed_quant_tensor_fail_closed() {
        let kvs = [
            ("llama.block_count", GgufValue::Uint32(1)),
            ("llama.embedding_length", GgufValue::Uint32(32)),
            ("llama.attention.head_count", GgufValue::Uint32(2)),
        ];

        // 1. Accepted: F32 norms + Q4_0 projection on Q4OnlyMockRecipe
        let supported_gguf_bytes = create_synthetic_gguf_v3_with_kvs(
            "llama",
            32,
            &kvs,
            &[
                ("blk.0.attn_norm.weight", QuantType::F32, &[32]),
                ("blk.0.attn_q.weight", QuantType::Q4_0, &[32, 32]),
            ],
        );
        let supported_gguf = GgufFile::parse(&supported_gguf_bytes).expect("parse supported gguf");
        Q4OnlyMockRecipe
            .validate_model(supported_gguf.metadata(), &supported_gguf)
            .expect("F32 + Q4_0 should pass Q4OnlyMockRecipe");

        // 2. Rejected: F32 norms + Q8_0 weights on Q4OnlyMockRecipe
        let mixed_gguf_bytes = create_synthetic_gguf_v3_with_kvs(
            "llama",
            32,
            &kvs,
            &[
                ("blk.0.attn_norm.weight", QuantType::F32, &[32]),
                ("blk.0.attn_q.weight", QuantType::Q8_0, &[32, 32]),
            ],
        );
        let mixed_gguf = GgufFile::parse(&mixed_gguf_bytes).expect("parse mixed gguf");

        for weights in [
            &mixed_gguf as &dyn ModelWeights,
            &EmptyMockWeights as &dyn ModelWeights,
        ] {
            let err = Q4OnlyMockRecipe
                .validate_model(mixed_gguf.metadata(), weights)
                .expect_err("Q8_0 weight must fail Q4OnlyMockRecipe");
            match err {
                K3Error::CapabilityMismatch {
                    architecture,
                    reason,
                } => {
                    assert_eq!(architecture, "llama");
                    assert!(
                        reason.contains("blk.0.attn_q.weight"),
                        "expected tensor name in reason: {reason}"
                    );
                    assert!(
                        reason.contains("Q8_0"),
                        "expected quant type in reason: {reason}"
                    );
                }
                other => panic!("expected CapabilityMismatch, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_kv_cache_view_layer_pos_and_sliding_window_miri_clean() {
        let geom = KvGeometry::new(2, 2, 4, 6, Some(3)).expect("valid kv geometry");
        assert_eq!(geom.kv_dim().unwrap(), 8);
        assert_eq!(geom.plane_len().unwrap(), 48);
        assert_eq!(geom.required_f32_len().unwrap(), 192);

        let mut raw_kv = vec![0.0_f32; geom.required_f32_len().unwrap() + 16];
        let mut logits = [0.0_f32; 8];
        let mut ctx = ExecutionContext::new(&[1, 2], 6, &mut logits, &mut raw_kv)
            .expect("execution context")
            .with_kv_geometry(geom)
            .expect("attach kv geometry");

        assert_eq!(ctx.kv_geometry(), Some(geom));
        {
            let mut kv_view = ctx.kv_view_mut().expect("kv view must exist");
            for layer in 0..2 {
                for pos in 0..5 {
                    let (k_slot, v_slot) = kv_view
                        .write_pos_range(layer, pos)
                        .expect("write position slot");
                    assert_eq!(k_slot.len(), 8);
                    assert_eq!(v_slot.len(), 8);
                    #[allow(clippy::cast_precision_loss)]
                    let base = (layer * 100 + pos * 10) as f32;
                    k_slot.fill(base + 1.0);
                    v_slot.fill(base + 2.0);
                }
            }

            // Sliding window W=3 at end_pos=5 keeps positions 2, 3, 4 (24 floats per plane)
            assert_eq!(kv_view.valid_token_range(5).unwrap(), 2..5);
            let (k_valid, v_valid) = kv_view.valid_kv_range(1, 5).expect("valid kv range");
            assert_eq!(k_valid.len(), 3 * 8);
            assert_eq!(v_valid.len(), 3 * 8);
            assert!((k_valid[0] - 121.0).abs() < f32::EPSILON);
            assert!((k_valid[8] - 131.0).abs() < f32::EPSILON);
            assert!((k_valid[16] - 141.0).abs() < f32::EPSILON);
            assert!((v_valid[16] - 142.0).abs() < f32::EPSILON);

            // At end_pos=2 < W=3, window starts at 0
            assert_eq!(kv_view.valid_token_range(2).unwrap(), 0..2);
            let (k_early, _) = kv_view.valid_kv_range(0, 2).expect("early kv range");
            assert_eq!(k_early.len(), 16);
            assert!((k_early[0] - 1.0).abs() < f32::EPSILON);

            // Bounds checks reject out-of-range layers and positions
            assert!(matches!(
                kv_view.layer_kv(2),
                Err(K3Error::TensorBoundsError { .. })
            ));
            assert!(matches!(
                kv_view.write_pos_range(0, 6),
                Err(K3Error::ContextLengthExceeded { .. })
            ));
            assert!(matches!(
                kv_view.valid_kv_range(0, 7),
                Err(K3Error::ContextLengthExceeded { .. })
            ));
        }

        // Backward compatibility: raw kv_cache() and kv_cache_mut() remain usable
        assert_eq!(ctx.kv_cache().len(), 192 + 16);
        ctx.kv_cache_mut()[0] = 42.0;
        assert!((ctx.kv_cache()[0] - 42.0).abs() < f32::EPSILON);

        // Zero dimensions and short backing buffers fail closed
        assert!(matches!(
            KvGeometry::new(0, 2, 4, 6, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            KvGeometry::new(2, 2, 4, 6, Some(0)),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut too_short = [0.0_f32; 16];
        let short_ctx =
            ExecutionContext::new(&[1], 6, &mut logits, &mut too_short).expect("execution context");
        assert!(matches!(
            short_ctx.with_kv_geometry(geom),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[derive(Debug)]
    struct ReferenceCausalRecipe {
        vocab_size: usize,
    }

    impl ReferenceCausalRecipe {
        #[allow(clippy::cast_precision_loss)]
        fn causal_layer_step(
            kv_view: &mut KvCacheView<'_>,
            layer: usize,
            tokens: &[u32],
            start_pos: usize,
            kv_dim: usize,
            hidden: &[f32],
        ) -> Result<Vec<f32>, K3Error> {
            let mut next_hidden = vec![0.0_f32; hidden.len()];
            for (offset, &tok) in tokens.iter().enumerate() {
                let pos = start_pos + offset;
                let h_slice = &hidden[offset * kv_dim..(offset + 1) * kv_dim];
                let k_vals: Vec<f32> = h_slice
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| v * 0.5 + (layer as f32) * 0.125 + (d as f32) * 0.0625)
                    .collect();
                let v_vals: Vec<f32> = h_slice
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| v * 0.25 + ((tok % 7) as f32) * 0.0625 - (d as f32) * 0.031_25)
                    .collect();
                {
                    let (k_dst, v_dst) = kv_view.write_pos_range(layer, pos)?;
                    k_dst.copy_from_slice(&k_vals);
                    v_dst.copy_from_slice(&v_vals);
                }

                let (k_valid, v_valid) = kv_view.valid_kv_range(layer, pos + 1)?;
                let mut attn_acc = vec![0.0_f32; kv_dim];
                let mut weight_sum = 0.0_f32;
                for (k_past, v_past) in k_valid
                    .chunks_exact(kv_dim)
                    .zip(v_valid.chunks_exact(kv_dim))
                {
                    let mut dot = 0.0_f32;
                    for d in 0..kv_dim {
                        dot += k_vals[d] * k_past[d];
                    }
                    let score = if dot >= 0.0 {
                        1.0_f32 + dot
                    } else {
                        1.0_f32 / (1.0_f32 - dot)
                    };
                    weight_sum += score;
                    for d in 0..kv_dim {
                        attn_acc[d] += score * v_past[d];
                    }
                }
                let inv = 1.0_f32 / weight_sum.max(1.0e-6);
                let out_slice = &mut next_hidden[offset * kv_dim..(offset + 1) * kv_dim];
                for d in 0..kv_dim {
                    out_slice[d] = h_slice[d] + attn_acc[d] * inv;
                }
            }
            Ok(next_hidden)
        }

        #[allow(clippy::cast_precision_loss)]
        fn run_tokens(
            &self,
            ctx: &mut ExecutionContext<'_>,
            tokens: &[u32],
            start_pos: usize,
            write_logits: bool,
        ) -> Result<(), K3Error> {
            let geom = ctx
                .kv_geometry()
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: "kv_cache".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "ReferenceCausalRecipe requires KvGeometry".to_owned(),
                })?;
            let kv_dim = geom.kv_dim()?;
            let mut hidden = vec![0.0_f32; tokens.len() * kv_dim];
            for (offset, &tok) in tokens.iter().enumerate() {
                let pos = start_pos + offset;
                for d in 0..kv_dim {
                    hidden[offset * kv_dim + d] = ((tok as f32) * 0.031_25)
                        + ((pos as f32) * 0.007_812_5)
                        + ((d as f32) * 0.015_625);
                }
            }

            {
                let mut kv_view = ctx
                    .kv_view_mut()
                    .ok_or_else(|| K3Error::TensorBoundsError {
                        tensor_name: "kv_cache".to_owned(),
                        offset: 0,
                        byte_len: 0,
                        buffer_len: 0,
                        reason: "ReferenceCausalRecipe requires valid KvCacheView".to_owned(),
                    })?;
                for layer in 0..geom.num_layers {
                    hidden = Self::causal_layer_step(
                        &mut kv_view,
                        layer,
                        tokens,
                        start_pos,
                        kv_dim,
                        &hidden,
                    )?;
                }
            }

            if write_logits {
                let last_offset = tokens.len() - 1;
                let last_h = &hidden[last_offset * kv_dim..(last_offset + 1) * kv_dim];
                let logits = ctx.logits_mut();
                if logits.len() < self.vocab_size {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "logits".to_owned(),
                        offset: 0,
                        byte_len: self.vocab_size * size_of::<f32>(),
                        buffer_len: size_of_val(logits),
                        reason: "logits slice smaller than vocab_size".to_owned(),
                    });
                }
                for (v, slot) in logits.iter_mut().take(self.vocab_size).enumerate() {
                    let mut sum = 0.0_f32;
                    for (d, &hv) in last_h.iter().enumerate() {
                        sum += hv * (((v as f32) + 1.0) * 0.031_25 - (d as f32) * 0.015_625);
                    }
                    *slot = sum;
                }
            }

            ctx.advance_position(tokens.len())?;
            Ok(())
        }
    }

    impl ArchRecipe for ReferenceCausalRecipe {
        fn architecture_id(&self) -> &'static str {
            "reference-causal"
        }

        fn supports_quant(&self, quant: QuantType) -> bool {
            matches!(quant, QuantType::F32)
        }

        fn prefill_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let tokens = ctx.prompt_tokens().to_vec();
            let start_pos = ctx.start_position();
            let write_logits = ctx.compute_logits();
            self.run_tokens(ctx, &tokens, start_pos, write_logits)
        }

        fn decode_step(
            &self,
            ctx: &mut ExecutionContext<'_>,
            _weights: &dyn ModelWeights,
        ) -> Result<(), K3Error> {
            let token = ctx
                .decode_token()
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: "decode_token".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "decode_step called without staged decode_token".to_owned(),
                })?;
            let pos = ctx.position();
            let write_logits = ctx.compute_logits();
            self.run_tokens(ctx, &[token], pos, write_logits)
        }
    }

    #[test]
    fn test_wave3_prefill_chunk_validation_and_slot_alignment() {
        let mut logits = [0.0_f32; 8];
        let mut kv_buf = [0.0_f32; 256];

        // new_prefill defaults to start_position=0, compute_logits=true
        let ctx = ExecutionContext::new_prefill(&[10, 20, 30], 16, &mut logits, &mut kv_buf)
            .expect("new_prefill should succeed");
        assert_eq!(ctx.start_position(), 0);
        assert_eq!(ctx.batch_len(), 3);
        assert_eq!(ctx.end_position(), 3);
        assert_eq!(ctx.batch_end_position(), 3);
        assert!(ctx.compute_logits());
        assert!(ctx.needs_logits());

        // Intermediate chunk with compute_logits=false permits empty &mut [] logits slice
        let mut empty_logits: [f32; 0] = [];
        let slot_hdr = TokenInputSlotHeader {
            slot_seq_id: 9,
            token_len: 3,
            start_position: 4,
        };
        let chunk_ctx = ExecutionContext::new_prefill_chunk(
            &[1, 2, 3],
            16,
            slot_hdr.start_position as usize,
            false,
            &mut empty_logits,
            &mut kv_buf,
        )
        .expect("intermediate chunk without logits should succeed");
        assert_eq!(chunk_ctx.start_position(), 4);
        assert_eq!(chunk_ctx.batch_len(), 3);
        assert_eq!(chunk_ctx.end_position(), 7);
        assert!(!chunk_ctx.compute_logits());
        assert!(!chunk_ctx.needs_logits());

        // compute_logits=true rejects empty logits
        assert!(matches!(
            ExecutionContext::new_prefill_chunk(
                &[1, 2],
                16,
                0,
                true,
                &mut empty_logits,
                &mut kv_buf
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Empty prompt fails closed
        assert!(matches!(
            ExecutionContext::new_prefill_chunk(&[], 16, 0, false, &mut empty_logits, &mut kv_buf),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Zero n_ctx, overflow, and end_position > n_ctx fail closed
        assert!(matches!(
            ExecutionContext::new_prefill_chunk(&[1], 0, 0, true, &mut logits, &mut kv_buf),
            Err(K3Error::ContextLengthExceeded { .. })
        ));
        assert!(matches!(
            ExecutionContext::new_prefill_chunk(
                &[1, 2],
                16,
                usize::MAX - 1,
                false,
                &mut empty_logits,
                &mut kv_buf
            ),
            Err(K3Error::ContextLengthExceeded { .. })
        ));
        assert!(matches!(
            ExecutionContext::new_prefill_chunk(&[1, 2, 3], 8, 6, true, &mut logits, &mut kv_buf),
            Err(K3Error::ContextLengthExceeded { .. })
        ));

        // KvGeometry capacity shorter than end_position fails closed in with_kv_geometry
        let small_geom = KvGeometry::new(2, 2, 4, 6, None).expect("geometry");
        let over_cap =
            ExecutionContext::new_prefill_chunk(&[1, 2, 3], 16, 4, true, &mut logits, &mut kv_buf)
                .expect("context fits n_ctx=16");
        assert!(matches!(
            over_cap.with_kv_geometry(small_geom),
            Err(K3Error::ContextLengthExceeded { .. })
        ));
    }

    fn execute_chunked_prefill(
        recipe: &ReferenceCausalRecipe,
        prompt: &[u32],
        n_ctx: usize,
        geom: KvGeometry,
        u_batch: usize,
        final_logits: &mut [f32],
        kv_buf: &mut [f32],
    ) {
        let mut sentinel_logits = [-999.0_f32; 8];
        for (idx, chunk) in prompt.chunks(u_batch).enumerate() {
            let start_pos = idx * u_batch;
            let is_last = start_pos + chunk.len() == prompt.len();
            if is_last {
                let mut ctx = ExecutionContext::new_prefill_chunk(
                    chunk,
                    n_ctx,
                    start_pos,
                    true,
                    final_logits,
                    kv_buf,
                )
                .expect("final chunk context")
                .with_kv_geometry(geom)
                .expect("final chunk geometry");
                recipe
                    .prefill_step(&mut ctx, &EmptyMockWeights)
                    .expect("final prefill step");
                assert_eq!(ctx.position(), prompt.len());
            } else {
                let mut ctx = ExecutionContext::new_prefill_chunk(
                    chunk,
                    n_ctx,
                    start_pos,
                    false,
                    &mut sentinel_logits,
                    kv_buf,
                )
                .expect("intermediate chunk context")
                .with_kv_geometry(geom)
                .expect("intermediate chunk geometry");
                recipe
                    .prefill_step(&mut ctx, &EmptyMockWeights)
                    .expect("intermediate prefill step");
                assert_eq!(ctx.position(), start_pos + chunk.len());
                assert!(
                    sentinel_logits
                        .iter()
                        .all(|v| v.to_bits() == (-999.0_f32).to_bits()),
                    "intermediate chunk with compute_logits=false must not write logits"
                );
            }
        }
    }

    fn assert_bit_identical(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: float mismatch at index {idx} ({a} != {e})"
            );
        }
    }

    #[test]
    fn test_wave3_chunked_prefill_equivalence_and_sliding_window() {
        let prompt = [11_u32, 22, 33, 44, 55, 66, 77, 88, 99, 101];
        let n_ctx = 16;
        let vocab_size = 8;
        let recipe = ReferenceCausalRecipe { vocab_size };

        for sliding_window in [None, Some(6)] {
            let geom = KvGeometry::new(2, 2, 4, n_ctx, sliding_window).expect("valid KvGeometry");
            let buf_len = geom.required_f32_len().expect("kv buffer length");

            let mut ref_kv = vec![12345.0_f32; buf_len];
            let mut ref_logits = vec![0.0_f32; vocab_size];
            let mut ref_ctx =
                ExecutionContext::new_prefill(&prompt, n_ctx, &mut ref_logits, &mut ref_kv)
                    .expect("reference prefill context")
                    .with_kv_geometry(geom)
                    .expect("reference geometry");
            recipe
                .prefill_step(&mut ref_ctx, &EmptyMockWeights)
                .expect("reference prefill");
            assert_eq!(ref_ctx.position(), prompt.len());

            for u_batch in [1_usize, 3, 10, 16] {
                let mut kv = vec![12345.0_f32; buf_len];
                let mut logits = vec![0.0_f32; vocab_size];
                execute_chunked_prefill(
                    &recipe,
                    &prompt,
                    n_ctx,
                    geom,
                    u_batch,
                    &mut logits,
                    &mut kv,
                );
                assert_bit_identical(&kv, &ref_kv, &format!("kv u_batch={u_batch}"));
                assert_bit_identical(&logits, &ref_logits, &format!("logits u_batch={u_batch}"));
            }
        }
    }

    #[test]
    fn test_wave3_empty_intermediate_logits_and_decode_continuation() {
        let prompt = [11_u32, 22, 33, 44, 55, 66, 77, 88, 99, 101];
        let n_ctx = 16;
        let vocab_size = 8;
        let recipe = ReferenceCausalRecipe { vocab_size };
        let geom = KvGeometry::new(2, 2, 4, n_ctx, Some(6)).expect("valid KvGeometry");
        let buf_len = geom.required_f32_len().expect("kv buffer length");

        let mut ref_kv = vec![12345.0_f32; buf_len];
        let mut ref_logits = vec![0.0_f32; vocab_size];
        let mut ref_ctx =
            ExecutionContext::new_prefill(&prompt, n_ctx, &mut ref_logits, &mut ref_kv)
                .expect("reference context")
                .with_kv_geometry(geom)
                .expect("reference geometry");
        recipe
            .prefill_step(&mut ref_ctx, &EmptyMockWeights)
            .expect("reference prefill");

        let mut chunk_kv = vec![12345.0_f32; buf_len];
        let mut chunk_logits = vec![0.0_f32; vocab_size];
        for (idx, chunk) in prompt.chunks(3).enumerate() {
            let start_pos = idx * 3;
            let is_last = start_pos + chunk.len() == prompt.len();
            let mut empty: [f32; 0] = [];
            let target_logits: &mut [f32] = if is_last {
                &mut chunk_logits
            } else {
                &mut empty
            };
            let mut ctx = ExecutionContext::new_prefill_chunk(
                chunk,
                n_ctx,
                start_pos,
                is_last,
                target_logits,
                &mut chunk_kv,
            )
            .expect("chunk ctx")
            .with_kv_geometry(geom)
            .expect("chunk geom");
            recipe
                .prefill_step(&mut ctx, &EmptyMockWeights)
                .expect("chunk step");
            if is_last {
                ctx.prepare_decode(202).expect("prepare_decode at pos 10");
                assert_eq!(ctx.position(), prompt.len());
                assert!(ctx.compute_logits());
                recipe
                    .decode_step(&mut ctx, &EmptyMockWeights)
                    .expect("decode step at pos 10");
                assert_eq!(ctx.position(), prompt.len() + 1);
            }
        }

        ref_ctx
            .prepare_decode(202)
            .expect("ref prepare_decode at pos 10");
        recipe
            .decode_step(&mut ref_ctx, &EmptyMockWeights)
            .expect("ref decode step at pos 10");

        assert_bit_identical(&chunk_kv, &ref_kv, "kv after prefill+decode");
        assert_bit_identical(&chunk_logits, &ref_logits, "logits after prefill+decode");
    }
}
