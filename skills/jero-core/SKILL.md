---
name: jero-core
description: "Implement ArchRecipe traits, zero-copy TensorView, GGUF v3 parsing, SpacemiT K3 cluster affinity, and SCM_RIGHTS FD passing with jero-core."
---

# `jero-core` Hardware Topology, Zero-Copy Tensor & IPC Invariants

## 1. SpacemiT K3 Canonical Cluster Mapping & `/proc/set_ai_thread`

- **Cluster 0 (`X100`, cores `0–7`, mask `0x00FF`)**: Out-of-Order general-purpose cores (`2.15 GHz`), `VLEN = 256` (`vlenb = 32`), supports RISC-V Hypervisor (`H`) extension (`/dev/kvm`). Does **not** support `XSMTVDotII` matrix instructions (`SIGILL` if executed on Cluster 0).
- **Cluster 1 (`A100`, cores `8–15`, mask `0xFF00`)**: In-Order AI/Matrix cores (`1.80 GHz`), `VLEN = 1024` (`vlenb = 128`), supports `XSMTVDotII` matrix instructions.
- **Conditional `/proc/set_ai_thread` Registration**: On SpacemiT-patched kernels (`spacemit-hmp`), any worker thread or forked capability-probe process pinning to Cluster 1 (`A100`, cores `8–15`) **must** write its kernel thread ID (`SYS_gettid`) to `/proc/set_ai_thread` before calling `sched_setaffinity`. Treat `/proc/set_ai_thread` as conditional (write only when the procfs node exists) so vanilla Linux kernels remain supported.

## 2. Rust Integer Sizing (`usize` vs. `u32`) & Strict Lints

- Use `usize` strictly for indexing collections, slice/array lengths, and memory allocation offsets (`TensorView` byte offsets).
- Use `u32` for domain-specific model metadata, token IDs, GGUF header counters, and wire protocol fields where 32-bit width must be architecture-independent.
- Every `unsafe` block must carry a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks = "warn"`, `unsafe_op_in_unsafe_fn = "deny"`).

## 3. Non-Tokenized Feature Matrices, Soft Prompts & Upfront Sanitization

- **`ExecutionContext` Continuous Inputs**: Use `ExecutionContext::new_feature_input`, `with_feature_input`, or `with_soft_prompt_embeds` to pass validated non-tokenized `f32` matrices or soft-prompt embeddings alongside empty `prompt_tokens`. All matrix constructors validate non-zero dimensions, `usize` multiplication bounds, exact slice lengths, and finite elements (`!x.is_finite()` fails closed with `K3Error::TensorBoundsError`) before attaching views.
- **Z-Score Normalization (`sanitize_and_normalize_features`)**: Reject non-finite (`NaN`, `±Inf`) feature elements, non-finite means, non-positive or non-finite standard deviations, non-positive `clip_sigma`, empty slices, or shape mismatches upfront prior to writing `out`.
- **Cacheline-Aligned Feature IPC Slots**: Serialize numerical feature batches (`FeatureDtype::{F32, F16, I32, I8}`) in shared memory using the 64-byte aligned `FeatureInputSlotHeader` (`SharedMemoryArena::write_feature_slot`, `write_f32_feature_slot`, `read_feature_slot_bytes`, `read_feature_slot_as_f32`) with upfront `NaN`/`±Inf` validation on both write and read paths.

## 4. Zero-Copy SafeTensors & Multimodal `ExecutionContext` Contracts

- **SafeTensors Parsing (`SafeTensorsFile`)**: Parse SafeTensors buffers via `SafeTensorsFile::from_bytes` (or `SafeTensorsFile::parse`), enforcing the `100 MiB` header cap (`MAX_SAFETENSORS_HEADER_BYTES`), UTF-8/JSON object integrity, `"F32"`/`"F16"`/`"BF16"` dtype mapping (`K3Error::CapabilityMismatch` on unsupported dtypes), checked shape products, alignment, and strictly non-overlapping `data_offsets` before exposing zero-copy `TensorView` slices via `ModelWeights`.
- **Multimodal `ExecutionContext`**: Attach validated multimodal buffers with `with_embedding_output` (`EmbeddingPooling::{None, Mean, Cls, LastToken}`, optional `matryoshka_dim`), `with_vision_input`, `with_audio_mel`, and `with_cross_attn_kv` (`with_cross_attn_kv_shaped`, `with_cross_attn_kv_readonly`), enforcing non-zero dimensions, checked products, and upfront finiteness checks (`is_finite()`).
