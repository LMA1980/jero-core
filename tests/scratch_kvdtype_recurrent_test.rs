use std::time::Instant;

use jero_core::{
    ArchRecipe, ArenaHeader, ExecutionContext, K3Error, KV_Q8_0_BLOCK_BYTES,
    KV_Q8_0_BLOCK_ELEMENTS, KvCacheView, KvDtype, KvGeometry, ModelWeights, QuantType,
    RecurrentStateSlotHeader, SCRATCH_ALIGNMENT, ScratchWorkspace, TensorView,
    estimate_recurrent_state_bytes,
};
use zerocopy::{FromBytes, IntoBytes};

struct MockWeights;

impl ModelWeights for MockWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "unused in scratch mock".to_owned(),
        })
    }

    fn tensor_count(&self) -> usize {
        0
    }

    fn total_weight_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct ScratchDecodeRecipe {
    hidden_dim: usize,
    ffn_dim: usize,
}

impl ArchRecipe for ScratchDecodeRecipe {
    fn architecture_id(&self) -> &'static str {
        "scratch_test"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::Q8_0)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let len = ctx.prompt_tokens().len();
        ctx.advance_position(len)?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let pos = ctx.position();
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "scratch_test".to_owned(),
                reason: "missing decode token".to_owned(),
            })?;

        let (logits, kv_opt, scratch_opt) = ctx.logits_kv_and_scratch_mut();
        let mut kv_view = kv_opt.ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "scratch_test".to_owned(),
            reason: "missing KvCacheView".to_owned(),
        })?;
        let scratch = scratch_opt.ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "scratch_test".to_owned(),
            reason: "missing ScratchWorkspace".to_owned(),
        })?;

        scratch.reset();
        {
            let scope = scratch.scope();
            let norm = scope.alloc_f32_zeroed(self.hidden_dim)?;
            let q = scope.alloc_f32_zeroed(self.hidden_dim)?;
            let k = scope.alloc_f32_zeroed(kv_view.kv_dim())?;
            let v = scope.alloc_f32_zeroed(kv_view.kv_dim())?;
            let q8_tmp = scope.alloc_i8_zeroed(self.hidden_dim)?;
            let attn_out = scope.alloc_f32_zeroed(self.hidden_dim)?;
            let scores = scope.alloc_f32_zeroed(pos + 1)?;
            let gate = scope.alloc_f32_zeroed(self.ffn_dim)?;
            let up = scope.alloc_f32_zeroed(self.ffn_dim)?;

            for slice_ptr in [
                norm.as_ptr() as usize,
                q.as_ptr() as usize,
                k.as_ptr() as usize,
                v.as_ptr() as usize,
                attn_out.as_ptr() as usize,
                scores.as_ptr() as usize,
                gate.as_ptr() as usize,
                up.as_ptr() as usize,
            ] {
                assert_eq!(slice_ptr % SCRATCH_ALIGNMENT, 0);
            }
            assert_eq!(q8_tmp.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);

            norm[0] = f32::from(u16::try_from(token).unwrap_or(u16::MAX)) * 0.25;
            q[0] = norm[0];
            k.fill(norm[0]);
            v.fill(norm[0]);
            q8_tmp[0] = 12;
            scores[pos] = 1.0;
            attn_out[0] = q[0];
            gate[0] = norm[0];
            up[0] = gate[0];

            let (k_slot, v_slot) = kv_view.write_pos_range(0, pos)?;
            k_slot.copy_from_slice(k);
            v_slot.copy_from_slice(v);
        }

        assert_eq!(scratch.used_bytes(), 0);
        if let Some(slot) = logits.get_mut(token as usize) {
            *slot = 1.0;
        }
        ctx.advance_position(1)?;
        Ok(())
    }
}

#[test]
fn multi_step_decode_reuses_single_scratch_workspace_without_growth() {
    let geom = KvGeometry::new(2, 4, 64, 64, None).expect("geometry");
    let mut kv_buf = vec![0.0f32; geom.required_f32_len().expect("f32 len")];
    let mut logits = vec![0.0f32; 64];
    // Sufficient for norm + q + k + v + q8_tmp + attn_out + scores + gate + up at 128B alignment.
    let mut scratch = ScratchWorkspace::new(256 * 1024).expect("workspace");

    let recipe = ScratchDecodeRecipe {
        hidden_dim: 512,
        ffn_dim: 2048,
    };

    let mut ctx = ExecutionContext::new_prefill(&[1, 2, 3], 64, &mut logits, &mut kv_buf)
        .expect("prefill ctx")
        .with_kv_geometry(geom)
        .expect("geom");
    recipe
        .prefill_step(&mut ctx, &MockWeights)
        .expect("prefill");

    let mut first_peak = 0usize;
    for step in 0..32u32 {
        let mut step_ctx = ExecutionContext::new(&[1], 64, &mut logits, &mut kv_buf)
            .expect("decode ctx")
            .with_kv_geometry(geom)
            .expect("geom")
            .with_position(3 + step as usize)
            .expect("pos")
            .with_scratch(&mut scratch);
        step_ctx.prepare_decode(step + 4).expect("prepare");
        recipe
            .decode_step(&mut step_ctx, &MockWeights)
            .expect("decode step");
        if step == 0 {
            first_peak = step_ctx.scratch().expect("scratch").peak_used_bytes();
        }
    }

    assert!(first_peak > 0);
    // Scores grow linearly with pos; capacity remained strictly bounded with zero reallocation.
    assert!(scratch.peak_used_bytes() < scratch.capacity_bytes());
    assert_eq!(scratch.used_bytes(), 0);
}

#[test]
fn kv_dtype_compression_ratios_and_arena_header_roundtrip() {
    // 8 heads * 128 head_dim = 1024 kv_dim
    let geom = KvGeometry::new(32, 8, 128, 4096, None).expect("4k geometry");
    let f32_bytes = geom.total_bytes(KvDtype::F32).expect("f32");
    let f16_bytes = geom.total_bytes(KvDtype::F16).expect("f16");
    let q8_bytes = geom.total_bytes(KvDtype::Q8_0).expect("q8");

    assert_eq!(geom.row_bytes(KvDtype::F32).unwrap(), 4096);
    assert_eq!(geom.row_bytes(KvDtype::F16).unwrap(), 2048);
    assert_eq!(
        geom.row_bytes(KvDtype::Q8_0).unwrap(),
        (1024 / KV_Q8_0_BLOCK_ELEMENTS) * KV_Q8_0_BLOCK_BYTES
    );
    assert_eq!(f32_bytes / 2, f16_bytes);
    assert!(q8_bytes * 3 < f32_bytes); // 34 bytes/block vs 128 bytes in F32 (~3.76x compression)

    let header = ArenaHeader {
        num_layers: 32,
        head_count_kv: 8,
        head_dim: 128,
        n_ctx: 4096,
        ..Default::default()
    }
    .with_kv_dtype(KvDtype::Q8_0)
    .with_seq_capacity(4096);

    assert!(header.validate().is_ok());
    assert_eq!(header.kv_row_bytes().unwrap(), 1088);

    let roundtrip = ArenaHeader::read_from_bytes(header.as_bytes()).expect("zerocopy roundtrip");
    assert_eq!(roundtrip.kv_dtype().unwrap(), KvDtype::Q8_0);
    assert_eq!(roundtrip.seq_capacity(), 4096);

    let mut kv_buf = vec![
        0.0f32;
        KvGeometry::new(2, 2, 32, 8, None)
            .unwrap()
            .required_f32_len()
            .unwrap()
    ];
    let tiny_geom = KvGeometry::new(2, 2, 32, 8, None).unwrap();
    let view = KvCacheView::new(&mut kv_buf, tiny_geom).unwrap();
    assert_eq!(
        view.row_bytes(KvDtype::Q8_0).unwrap(),
        2 * KV_Q8_0_BLOCK_BYTES
    );
}

#[test]
fn recurrent_state_sizing_is_o1_in_context_length_and_ping_pong_safe() {
    let small_ctx_bytes = estimate_recurrent_state_bytes(24, 1536, 32768).expect("24 layers");
    let large_ctx_bytes = estimate_recurrent_state_bytes(24, 1536, 32768).expect("o1 invariant");
    assert_eq!(small_ctx_bytes, large_ctx_bytes);
    assert_eq!(small_ctx_bytes, 64 + 2 * 24 * (1536 + 32768));

    let bank_stride = 24 * (1536 + 32768);
    let mut hdr = RecurrentStateSlotHeader::new(24, 1536, 32768, bank_stride);
    assert_eq!(hdr.total_region_bytes(), small_ctx_bytes);

    let t0 = Instant::now();
    for step in 1..=64u32 {
        let bank = hdr.commit_step(step).expect("commit step");
        assert_eq!(bank, step % 2);
        assert_eq!(hdr.committed_step, step);
        let recovered =
            RecurrentStateSlotHeader::read_from_bytes(hdr.as_bytes()).expect("zerocopy roundtrip");
        assert_eq!(recovered.active_bank, bank);
        assert_eq!(recovered.committed_step, step);
    }
    assert!(t0.elapsed().as_secs() < 5);
}
