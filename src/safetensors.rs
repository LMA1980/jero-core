//! Zero-copy `SafeTensors` binary parser, metadata extraction, and `ModelWeights` adapter.
//!
//! Conforms to the Hugging Face `SafeTensors` binary layout:
//! - 8-byte little-endian `u64` header byte length `N`, bounded by
//!   [`MAX_SAFETENSORS_HEADER_BYTES`] (`100 MiB`) to prevent header-allocation denial of service.
//! - `[8 .. 8 + N]` UTF-8 JSON object containing optional `"__metadata__"` and tensor entries.
//! - `[8 + N ..]` raw little-endian tensor payload buffer sliced zero-copy via [`TensorView`].

use core::fmt;
use std::collections::BTreeMap;

use serde::Deserialize;
use serde::de::{MapAccess, Visitor};

use crate::error::K3Error;
use crate::gguf::{ModelMetadata, ModelMetadataParts, TensorDescriptor};
use crate::tensor::{ModelWeights, QuantType, TensorView};

/// Maximum permitted `SafeTensors` JSON header size in bytes (`100 MiB`) to guard against
/// unbounded allocation on malformed or adversarial inputs.
pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 100 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct ParsedHeader {
    custom_metadata: BTreeMap<String, String>,
    raw_tensors: Vec<(String, RawTensorEntry)>,
}

struct HeaderMapVisitor;

impl<'de> Visitor<'de> for HeaderMapVisitor {
    type Value = ParsedHeader;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a SafeTensors JSON header object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut custom_metadata: Option<BTreeMap<String, String>> = None;
        let mut saw_metadata_key = false;
        let mut raw_tensors = Vec::new();
        let mut seen_names = std::collections::BTreeSet::new();

        while let Some(key) = map.next_key::<String>()? {
            if key == "__metadata__" {
                if saw_metadata_key {
                    return Err(serde::de::Error::duplicate_field("__metadata__"));
                }
                saw_metadata_key = true;
                custom_metadata = map.next_value::<Option<BTreeMap<String, String>>>()?;
            } else {
                if key.is_empty() {
                    return Err(serde::de::Error::custom("tensor name must not be empty"));
                }
                if !seen_names.insert(key.clone()) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate tensor entry '{key}'"
                    )));
                }
                let entry = map.next_value::<RawTensorEntry>()?;
                raw_tensors.push((key, entry));
            }
        }

        Ok(ParsedHeader {
            custom_metadata: custom_metadata.unwrap_or_default(),
            raw_tensors,
        })
    }
}

impl<'de> Deserialize<'de> for ParsedHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(HeaderMapVisitor)
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorInterval {
    begin: usize,
    end: usize,
    desc_idx: usize,
}

/// Zero-copy `SafeTensors` file view backed by a contiguous byte slice (e.g. an `mmap` buffer).
#[derive(Debug, Clone)]
pub struct SafeTensorsFile<'a> {
    buffer: &'a [u8],
    data_offset: usize,
    metadata: ModelMetadata,
    custom_metadata: BTreeMap<String, String>,
    tensor_index: BTreeMap<String, usize>,
}

impl<'a> SafeTensorsFile<'a> {
    /// Parses a `SafeTensors` payload from a contiguous byte slice without copying tensor bytes.
    ///
    /// Enforces:
    /// - Header length `1 <= N <= MAX_SAFETENSORS_HEADER_BYTES` (`100 MiB`) and `8 + N <= data.len()`
    /// - Valid UTF-8 JSON object header without duplicate keys or empty tensor names
    /// - Supported dtypes (`"F32"`, `"F16"`, `"BF16"`); unsupported dtypes fail closed with
    ///   [`K3Error::CapabilityMismatch`]
    /// - Non-empty shapes with non-zero dimensions and checked `usize` element products
    /// - Valid `data_offsets: [begin, end]` with `begin <= end`, exact byte length matching
    ///   `quant.storage_bytes(...)`, `end` within `[8 + N .. data.len()]`, `begin` aligned to
    ///   `quant.alignment()`, and strictly non-overlapping byte intervals across all tensors
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if the header, JSON structure, tensor shapes,
    /// alignment, or `data_offsets` fail validation, or [`K3Error::CapabilityMismatch`] if any
    /// tensor uses an unsupported dtype.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, K3Error> {
        let (header_bytes, data_offset, tensor_buffer_len) = parse_header_framing(data)?;
        let header_str =
            core::str::from_utf8(header_bytes).map_err(|err| K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: 8,
                byte_len: header_bytes.len(),
                buffer_len: data.len(),
                reason: format!("invalid UTF-8 in SafeTensors JSON header: {err}"),
            })?;

        let parsed_header: ParsedHeader =
            serde_json::from_str(header_str).map_err(|err| K3Error::TensorBoundsError {
                tensor_name: "header".to_owned(),
                offset: 8,
                byte_len: header_bytes.len(),
                buffer_len: data.len(),
                reason: format!("malformed SafeTensors JSON header: {err}"),
            })?;

        let architecture = resolve_architecture(&parsed_header.custom_metadata);
        let validated = build_and_validate_tensors(
            parsed_header.raw_tensors,
            &architecture,
            data_offset,
            tensor_buffer_len,
            data.len(),
        )?;

        let total_weight_bytes = validate_non_overlapping(
            &validated.descriptors,
            validated.intervals,
            tensor_buffer_len,
        )?;

        let metadata = build_model_metadata(
            architecture,
            &parsed_header.custom_metadata,
            validated.descriptors,
            total_weight_bytes,
        );

        Ok(Self {
            buffer: data,
            data_offset,
            metadata,
            custom_metadata: parsed_header.custom_metadata,
            tensor_index: validated.tensor_index,
        })
    }

    /// Alias for [`Self::from_bytes`] matching [`crate::GgufFile::parse`].
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] on invalid input.
    pub fn parse(data: &'a [u8]) -> Result<Self, K3Error> {
        Self::from_bytes(data)
    }

    /// Returns the extracted [`ModelMetadata`] for this `SafeTensors` checkpoint.
    #[must_use]
    pub const fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Returns the validated list of tensor descriptors.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDescriptor] {
        self.metadata.tensors()
    }

    /// Alias for [`Self::tensors`].
    #[must_use]
    pub fn tensor_descriptors(&self) -> &[TensorDescriptor] {
        self.metadata.tensors()
    }

    /// Returns the parsed `"__metadata__"` string map from the `SafeTensors` header.
    #[must_use]
    pub const fn custom_metadata(&self) -> &BTreeMap<String, String> {
        &self.custom_metadata
    }

    /// Looks up a string entry inside the `"__metadata__"` header map, if present.
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.custom_metadata.get(key).map(String::as_str)
    }

    /// Returns the resolved model architecture identifier.
    #[must_use]
    pub fn architecture(&self) -> &str {
        self.metadata.architecture()
    }

    /// Returns the starting byte offset (`8 + header_len`) of the tensor data payload.
    #[must_use]
    pub const fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Returns the full underlying `SafeTensors` byte slice.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.buffer
    }

    /// Returns the zero-copy tensor payload section `&buffer[data_offset..]`.
    #[must_use]
    pub fn tensor_data(&self) -> &'a [u8] {
        &self.buffer[self.data_offset..]
    }
}

impl ModelWeights for SafeTensorsFile<'_> {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let &idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.buffer.len(),
                reason: format!("tensor '{name}' not found in SafeTensors weight table"),
            })?;

        let desc = &self.metadata.tensors()[idx];
        TensorView::from_bytes(
            desc.name(),
            desc.quant(),
            desc.shape(),
            desc.absolute_offset(),
            self.buffer,
        )
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    fn tensor_count(&self) -> usize {
        self.metadata.tensor_count()
    }

    fn total_weight_bytes(&self) -> usize {
        self.metadata.total_weight_bytes()
    }
}

fn parse_header_framing(data: &[u8]) -> Result<(&[u8], usize, usize), K3Error> {
    if data.len() < 8 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 0,
            byte_len: 8,
            buffer_len: data.len(),
            reason: format!(
                "SafeTensors buffer length {} is smaller than 8-byte header length prefix",
                data.len()
            ),
        });
    }

    let len_bytes: [u8; 8] = data[0..8]
        .try_into()
        .map_err(|_| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 0,
            byte_len: 8,
            buffer_len: data.len(),
            reason: "failed to read 8-byte SafeTensors header length prefix".to_owned(),
        })?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);

    if header_len_u64 == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 8,
            byte_len: 0,
            buffer_len: data.len(),
            reason: "SafeTensors header length must be greater than zero".to_owned(),
        });
    }

    if header_len_u64 > MAX_SAFETENSORS_HEADER_BYTES as u64 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 8,
            byte_len: usize::try_from(header_len_u64).unwrap_or(usize::MAX),
            buffer_len: data.len(),
            reason: format!(
                "SafeTensors header length {header_len_u64} bytes exceeds {MAX_SAFETENSORS_HEADER_BYTES} byte limit"
            ),
        });
    }

    let header_len = usize::try_from(header_len_u64).map_err(|_| K3Error::TensorBoundsError {
        tensor_name: "header".to_owned(),
        offset: 8,
        byte_len: usize::MAX,
        buffer_len: data.len(),
        reason: format!("SafeTensors header length {header_len_u64} overflows usize"),
    })?;

    let data_offset = 8usize
        .checked_add(header_len)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 8,
            byte_len: header_len,
            buffer_len: data.len(),
            reason: "SafeTensors header offset calculation overflowed usize".to_owned(),
        })?;

    if data_offset > data.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "header".to_owned(),
            offset: 8,
            byte_len: header_len,
            buffer_len: data.len(),
            reason: format!(
                "SafeTensors header range [8..{data_offset}] exceeds buffer length {}",
                data.len()
            ),
        });
    }

    let tensor_buffer_len = data.len() - data_offset;
    Ok((&data[8..data_offset], data_offset, tensor_buffer_len))
}

fn map_safetensors_dtype(
    dtype: &str,
    tensor_name: &str,
    architecture: &str,
) -> Result<QuantType, K3Error> {
    match dtype {
        "F32" => Ok(QuantType::F32),
        "F16" => Ok(QuantType::F16),
        "BF16" => Ok(QuantType::BF16),
        other => Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: format!("tensor '{tensor_name}' uses unsupported SafeTensors dtype '{other}'"),
        }),
    }
}

struct ValidatedTensors {
    descriptors: Vec<TensorDescriptor>,
    intervals: Vec<TensorInterval>,
    tensor_index: BTreeMap<String, usize>,
}

fn build_and_validate_tensors(
    raw_tensors: Vec<(String, RawTensorEntry)>,
    architecture: &str,
    data_offset: usize,
    tensor_buffer_len: usize,
    total_buffer_len: usize,
) -> Result<ValidatedTensors, K3Error> {
    let mut descriptors = Vec::with_capacity(raw_tensors.len());
    let mut intervals = Vec::with_capacity(raw_tensors.len());
    let mut tensor_index = BTreeMap::new();

    for (idx, (name, raw)) in raw_tensors.into_iter().enumerate() {
        let quant = map_safetensors_dtype(&raw.dtype, &name, architecture)?;
        let (shape, num_elements) = validate_shape(&name, &raw.shape, tensor_buffer_len)?;
        let expected_bytes = quant.storage_bytes(&name, num_elements)?;

        let (begin, end, absolute_offset) = validate_offsets(
            &name,
            quant,
            raw.data_offsets,
            expected_bytes,
            data_offset,
            tensor_buffer_len,
            total_buffer_len,
        )?;

        tensor_index.insert(name.clone(), idx);
        intervals.push(TensorInterval {
            begin,
            end,
            desc_idx: idx,
        });
        descriptors.push(TensorDescriptor::new(
            name,
            quant,
            shape,
            num_elements,
            begin,
            absolute_offset,
            expected_bytes,
        ));
    }

    Ok(ValidatedTensors {
        descriptors,
        intervals,
        tensor_index,
    })
}

fn validate_shape(
    name: &str,
    raw_shape: &[u64],
    tensor_buffer_len: usize,
) -> Result<(Vec<usize>, usize), K3Error> {
    if raw_shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: tensor_buffer_len,
            reason: "tensor shape must contain at least one dimension".to_owned(),
        });
    }

    let mut shape = Vec::with_capacity(raw_shape.len());
    let mut num_elements = 1usize;

    for &dim_u64 in raw_shape {
        let dim = usize::try_from(dim_u64).map_err(|_| K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: tensor_buffer_len,
            reason: format!("tensor dimension {dim_u64} overflows usize"),
        })?;
        if dim == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: tensor_buffer_len,
                reason: "tensor shape dimensions must be non-zero".to_owned(),
            });
        }
        num_elements = num_elements
            .checked_mul(dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: tensor_buffer_len,
                reason: "tensor shape element product overflowed usize".to_owned(),
            })?;
        shape.push(dim);
    }

    Ok((shape, num_elements))
}

fn validate_offsets(
    name: &str,
    quant: QuantType,
    data_offsets: [u64; 2],
    expected_bytes: usize,
    data_offset: usize,
    tensor_buffer_len: usize,
    total_buffer_len: usize,
) -> Result<(usize, usize, usize), K3Error> {
    let begin = usize::try_from(data_offsets[0]).map_err(|_| K3Error::TensorBoundsError {
        tensor_name: name.to_owned(),
        offset: usize::MAX,
        byte_len: expected_bytes,
        buffer_len: tensor_buffer_len,
        reason: format!("data_offsets[0] ({}) overflows usize", data_offsets[0]),
    })?;
    let end = usize::try_from(data_offsets[1]).map_err(|_| K3Error::TensorBoundsError {
        tensor_name: name.to_owned(),
        offset: begin,
        byte_len: expected_bytes,
        buffer_len: tensor_buffer_len,
        reason: format!("data_offsets[1] ({}) overflows usize", data_offsets[1]),
    })?;

    if begin > end {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: begin,
            byte_len: expected_bytes,
            buffer_len: tensor_buffer_len,
            reason: format!("inverted SafeTensors data_offsets [{begin}, {end}]"),
        });
    }

    let actual_bytes = end - begin;
    if actual_bytes != expected_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: begin,
            byte_len: actual_bytes,
            buffer_len: tensor_buffer_len,
            reason: format!(
                "data_offsets span [{begin}..{end}) ({actual_bytes} bytes) does not match expected {expected_bytes} bytes for {quant}"
            ),
        });
    }

    let required_align = quant.alignment();
    if required_align > 1 && begin % required_align != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: begin,
            byte_len: actual_bytes,
            buffer_len: tensor_buffer_len,
            reason: format!(
                "data_offsets start {begin} is not aligned to {required_align}-byte requirement for {quant}"
            ),
        });
    }

    if end > tensor_buffer_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: begin,
            byte_len: actual_bytes,
            buffer_len: tensor_buffer_len,
            reason: format!(
                "tensor range [{begin}..{end}) exceeds tensor buffer length {tensor_buffer_len}"
            ),
        });
    }

    let absolute_offset =
        data_offset
            .checked_add(begin)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: begin,
                byte_len: actual_bytes,
                buffer_len: total_buffer_len,
                reason: "tensor absolute offset overflowed usize".to_owned(),
            })?;

    Ok((begin, end, absolute_offset))
}

fn validate_non_overlapping(
    descriptors: &[TensorDescriptor],
    mut intervals: Vec<TensorInterval>,
    tensor_buffer_len: usize,
) -> Result<usize, K3Error> {
    intervals.sort_unstable_by_key(|item| (item.begin, item.end));

    let mut total_weight_bytes = 0usize;
    let mut prev: Option<TensorInterval> = None;

    for cur in intervals {
        if let Some(p) = prev
            && cur.begin < p.end
        {
            let cur_name = descriptors[cur.desc_idx].name();
            let prev_name = descriptors[p.desc_idx].name();
            return Err(K3Error::TensorBoundsError {
                tensor_name: cur_name.to_owned(),
                offset: cur.begin,
                byte_len: cur.end - cur.begin,
                buffer_len: tensor_buffer_len,
                reason: format!(
                    "tensor range [{}..{}) overlaps with tensor '{prev_name}' [{}..{})",
                    cur.begin, cur.end, p.begin, p.end
                ),
            });
        }

        total_weight_bytes = total_weight_bytes
            .checked_add(cur.end - cur.begin)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: descriptors[cur.desc_idx].name().to_owned(),
                offset: cur.begin,
                byte_len: cur.end - cur.begin,
                buffer_len: tensor_buffer_len,
                reason: "total SafeTensors weight bytes overflowed usize".to_owned(),
            })?;
        prev = Some(cur);
    }

    Ok(total_weight_bytes)
}

fn resolve_architecture(custom_metadata: &BTreeMap<String, String>) -> String {
    for key in ["general.architecture", "model_type", "architecture"] {
        if let Some(val) = custom_metadata.get(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return trimmed.to_owned();
            }
        }
    }
    "unknown".to_owned()
}

fn lookup_meta_usize(
    custom_metadata: &BTreeMap<String, String>,
    architecture: &str,
    arch_suffix: Option<&str>,
    aliases: &[&str],
) -> Option<usize> {
    if let Some(suffix) = arch_suffix {
        let arch_key = format!("{architecture}.{suffix}");
        if let Some(val) = custom_metadata
            .get(&arch_key)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
        {
            return Some(val);
        }
    }
    for alias in aliases {
        if let Some(val) = custom_metadata
            .get(*alias)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
        {
            return Some(val);
        }
    }
    None
}

fn lookup_meta_f32(
    custom_metadata: &BTreeMap<String, String>,
    architecture: &str,
    arch_suffix: Option<&str>,
    aliases: &[&str],
) -> Option<f32> {
    if let Some(suffix) = arch_suffix {
        let arch_key = format!("{architecture}.{suffix}");
        if let Some(val) = custom_metadata
            .get(&arch_key)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            return Some(val);
        }
    }
    for alias in aliases {
        if let Some(val) = custom_metadata
            .get(*alias)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            return Some(val);
        }
    }
    None
}

fn extract_layer_index(name: &str) -> Option<usize> {
    const PREFIXES: &[&str] = &[
        "blk.",
        "model.layers.",
        "layers.",
        "encoder.layer.",
        "encoder.layers.",
        "decoder.layers.",
        "transformer.h.",
        "blocks.",
        "h.",
    ];
    for prefix in PREFIXES {
        if let Some(rest) = name.strip_prefix(prefix)
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            return Some(idx);
        }
    }
    None
}

fn infer_layer_count(descriptors: &[TensorDescriptor]) -> usize {
    let mut max_idx: Option<usize> = None;
    for desc in descriptors {
        if let Some(idx) = extract_layer_index(desc.name()) {
            max_idx = Some(max_idx.map_or(idx, |cur| cur.max(idx)));
        }
    }
    max_idx.map_or(0, |idx| idx.saturating_add(1))
}

fn infer_embedding_dim(descriptors: &[TensorDescriptor]) -> usize {
    const NORM_NAMES: &[&str] = &[
        "model.norm.weight",
        "norm.weight",
        "ln_f.weight",
        "transformer.ln_f.weight",
        "encoder.layer_norm.weight",
        "decoder.layer_norm.weight",
        "blk.0.attn_norm.weight",
        "output_norm.weight",
    ];
    const HF_EMBED_NAMES: &[&str] = &[
        "model.embed_tokens.weight",
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
        "wte.weight",
        "transformer.wte.weight",
        "decoder.embed_tokens.weight",
        "embed_tokens.weight",
    ];

    for norm_name in NORM_NAMES {
        if let Some(desc) = descriptors.iter().find(|d| d.name() == *norm_name)
            && desc.shape().len() == 1
        {
            return desc.shape()[0];
        }
    }

    for embed_name in HF_EMBED_NAMES {
        if let Some(desc) = descriptors.iter().find(|d| d.name() == *embed_name)
            && desc.shape().len() == 2
        {
            return desc.shape()[1];
        }
    }

    if let Some(desc) = descriptors.iter().find(|d| d.name() == "token_embd.weight")
        && desc.shape().len() == 2
    {
        return desc.shape()[0];
    }

    0
}

fn infer_vocab_size(descriptors: &[TensorDescriptor]) -> Option<usize> {
    const HF_VOCAB_NAMES: &[&str] = &[
        "model.embed_tokens.weight",
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
        "lm_head.weight",
        "wte.weight",
        "transformer.wte.weight",
        "decoder.embed_tokens.weight",
        "embed_tokens.weight",
    ];
    for name in HF_VOCAB_NAMES {
        if let Some(desc) = descriptors.iter().find(|d| d.name() == *name)
            && desc.shape().len() == 2
        {
            return Some(desc.shape()[0]);
        }
    }
    for name in ["token_embd.weight", "output.weight"] {
        if let Some(desc) = descriptors.iter().find(|d| d.name() == name)
            && desc.shape().len() == 2
        {
            return Some(desc.shape()[1]);
        }
    }
    None
}

fn build_model_metadata(
    architecture: String,
    custom_metadata: &BTreeMap<String, String>,
    descriptors: Vec<TensorDescriptor>,
    total_weight_bytes: usize,
) -> ModelMetadata {
    let u = |suffix: &str, aliases: &[&str]| -> Option<usize> {
        lookup_meta_usize(custom_metadata, &architecture, Some(suffix), aliases)
    };
    let f = |suffix: &str, aliases: &[&str]| -> Option<f32> {
        lookup_meta_f32(custom_metadata, &architecture, Some(suffix), aliases)
    };

    let layer_count = u(
        "block_count",
        &[
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "encoder_layers",
        ],
    )
    .unwrap_or_else(|| infer_layer_count(&descriptors));
    let embedding_dim = u(
        "embedding_length",
        &["hidden_size", "d_model", "n_embd", "dim"],
    )
    .unwrap_or_else(|| infer_embedding_dim(&descriptors));
    let head_count = u(
        "attention.head_count",
        &["num_attention_heads", "n_head", "encoder_attention_heads"],
    )
    .unwrap_or(0);
    let head_count_kv = u(
        "attention.head_count_kv",
        &["num_key_value_heads", "n_head_kv"],
    )
    .unwrap_or(head_count);
    let context_length = u(
        "context_length",
        &["max_position_embeddings", "n_positions", "n_ctx"],
    )
    .unwrap_or(2048);
    let alignment =
        lookup_meta_usize(custom_metadata, "general", Some("alignment"), &[]).unwrap_or(1);
    let vocab_size = u("vocab_size", &["vocab_size"]).or_else(|| infer_vocab_size(&descriptors));
    let tensor_count = descriptors.len();

    ModelMetadata::from_parts(ModelMetadataParts {
        layer_count,
        embedding_dim,
        head_count,
        head_count_kv,
        context_length,
        alignment,
        tensor_count,
        total_weight_bytes,
        tensors: descriptors,
        qk_nope_head_dim: u("attention.key_length", &["head_dim"]),
        v_head_dim: u("attention.value_length", &["head_dim"]),
        expert_count: u("expert_count", &["num_local_experts", "num_experts"]),
        expert_used_count: u("expert_used_count", &["num_experts_per_tok"]),
        feed_forward_length: u("feed_forward_length", &["intermediate_size", "n_inner"]),
        rope_freq_base: f("rope.freq_base", &["rope_theta"]),
        rope_dimension_count: u("rope.dimension_count", &["rotary_dim"]),
        rms_norm_eps: f(
            "attention.layer_norm_rms_epsilon",
            &["rms_norm_eps", "layer_norm_eps"],
        ),
        sliding_window: u("attention.sliding_window", &["sliding_window"]),
        vocab_size,
        architecture,
    })
}

/// Test helpers for synthesizing `SafeTensors` binaries in unit and integration tests.
#[doc(hidden)]
pub mod test_helpers {
    use crate::tensor::QuantType;

    /// Constructs a valid `SafeTensors` byte buffer from optional `__metadata__` pairs and
    /// `(name, quant, shape, raw_bytes)` tensor definitions.
    ///
    /// Automatically pads tensor offsets to `quant.alignment()` within the data payload section.
    ///
    /// # Panics
    /// Panics if a tensor uses a quantized format unsupported by `SafeTensors`, or if `raw_bytes`
    /// length does not match `quant.storage_bytes(name, shape.iter().product())`.
    #[must_use]
    pub fn create_synthetic_safetensors(
        custom_metadata: &[(&str, &str)],
        tensors: &[(&str, QuantType, &[usize], &[u8])],
    ) -> Vec<u8> {
        let mut root = serde_json::Map::new();
        if !custom_metadata.is_empty() {
            let mut meta_map = serde_json::Map::new();
            for &(k, v) in custom_metadata {
                meta_map.insert(k.to_owned(), serde_json::Value::String(v.to_owned()));
            }
            root.insert(
                "__metadata__".to_owned(),
                serde_json::Value::Object(meta_map),
            );
        }

        let mut payload = Vec::new();
        for &(name, quant, shape, bytes) in tensors {
            assert!(
                matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16),
                "synthetic SafeTensors helper supports F32/F16/BF16 only, got {quant}"
            );
            let num_elems: usize = shape.iter().product();
            let expected = quant
                .storage_bytes(name, num_elems)
                .expect("valid tensor shape");
            assert_eq!(
                bytes.len(),
                expected,
                "raw bytes length mismatch for tensor '{name}'"
            );

            let align = quant.alignment();
            if align > 1 {
                let rem = payload.len() % align;
                if rem != 0 {
                    payload.resize(payload.len() + (align - rem), 0);
                }
            }

            let begin = payload.len();
            payload.extend_from_slice(bytes);
            let end = payload.len();

            let mut entry = serde_json::Map::new();
            entry.insert(
                "dtype".to_owned(),
                serde_json::Value::String(quant.as_str().to_owned()),
            );
            entry.insert(
                "shape".to_owned(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|&d| {
                            let d_u64 = u64::try_from(d).expect("shape dimension fits u64");
                            serde_json::Value::Number(serde_json::Number::from(d_u64))
                        })
                        .collect(),
                ),
            );
            let begin_u64 = u64::try_from(begin).expect("begin offset fits u64");
            let end_u64 = u64::try_from(end).expect("end offset fits u64");
            entry.insert(
                "data_offsets".to_owned(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number(serde_json::Number::from(begin_u64)),
                    serde_json::Value::Number(serde_json::Number::from(end_u64)),
                ]),
            );
            root.insert(name.to_owned(), serde_json::Value::Object(entry));
        }

        let header_json = serde_json::to_vec(&serde_json::Value::Object(root))
            .expect("serialize SafeTensors header");
        let header_len = u64::try_from(header_json.len()).expect("header length fits u64");

        let mut out = Vec::with_capacity(8 + header_json.len() + payload.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_json);
        out.extend_from_slice(&payload);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::create_synthetic_safetensors;
    use super::*;

    fn build_raw_safetensors(header_json: &str, payload: &[u8]) -> Vec<u8> {
        let len = u64::try_from(header_json.len()).expect("header len fits u64");
        let mut buf = Vec::with_capacity(8 + header_json.len() + payload.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(header_json.as_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn safetensors_zero_copy_roundtrip_and_metadata_extraction() {
        let mut f32_bytes = Vec::new();
        for val in [1.0_f32, -2.5, 3.25, 0.5, 4.0, -1.0] {
            f32_bytes.extend_from_slice(&val.to_le_bytes());
        }
        let f16_bytes = [0x00_u8, 0x3C, 0x00, 0x40, 0x00, 0x42]; // 3 * F16 = 6 bytes
        let bf16_bytes = [0x80_u8, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]; // 4 * BF16 = 8 bytes

        let st_bytes = create_synthetic_safetensors(
            &[
                ("general.architecture", "modernbert"),
                ("num_hidden_layers", "6"),
                ("num_attention_heads", "12"),
                ("max_position_embeddings", "8192"),
                ("intermediate_size", "3072"),
                ("rms_norm_eps", "1e-5"),
                ("rope_theta", "160000.0"),
                ("format", "pt"),
            ],
            &[
                ("model.norm.weight", QuantType::BF16, &[4], &bf16_bytes),
                (
                    "model.layers.1.attn_q.weight",
                    QuantType::F32,
                    &[2, 3],
                    &f32_bytes,
                ),
                (
                    "model.embed_tokens.weight",
                    QuantType::F16,
                    &[3, 1],
                    &f16_bytes,
                ),
            ],
        );

        let parsed = SafeTensorsFile::from_bytes(&st_bytes).expect("valid SafeTensors file");
        assert_eq!(parsed.architecture(), "modernbert");
        assert_eq!(parsed.metadata_value("format"), Some("pt"));
        assert_eq!(
            parsed.custom_metadata().get("format").map(String::as_str),
            Some("pt")
        );

        let meta = parsed.metadata();
        assert_eq!(meta.architecture(), "modernbert");
        assert_eq!(meta.layer_count(), 6);
        assert_eq!(meta.embedding_dim(), 4);
        assert_eq!(meta.head_count(), 12);
        assert_eq!(meta.head_count_kv(), 12);
        assert_eq!(meta.context_length(), 8192);
        assert_eq!(meta.feed_forward_length(), Some(3072));
        assert_eq!(meta.vocab_size(), Some(3));
        assert!(meta.rms_norm_eps().is_some_and(|v| (v - 1e-5).abs() < 1e-9));
        assert!(
            meta.rope_freq_base()
                .is_some_and(|v| (v - 160_000.0).abs() < 1e-3)
        );
        assert_eq!(parsed.tensor_count(), 3);
        assert_eq!(parsed.total_weight_bytes(), 8 + 24 + 6);
        assert_eq!(parsed.tensors().len(), 3);
        assert_eq!(parsed.tensor_descriptors().len(), 3);
        assert_eq!(parsed.as_bytes().as_ptr(), st_bytes.as_ptr());

        assert!(parsed.contains_tensor("model.layers.1.attn_q.weight"));
        assert!(!parsed.contains_tensor("missing.weight"));

        let q_view = parsed
            .tensor("model.layers.1.attn_q.weight")
            .expect("lookup F32 tensor");
        assert_eq!(q_view.quant(), QuantType::F32);
        assert_eq!(q_view.shape(), &[2, 3]);
        assert_eq!(q_view.num_elements(), 6);
        assert_eq!(q_view.byte_len(), 24);
        assert_eq!(q_view.as_bytes(), &f32_bytes[..]);
        // Zero-copy pointer invariant: slice points strictly within original st_bytes buffer.
        let buf_range = st_bytes.as_ptr_range();
        assert!(buf_range.contains(&q_view.as_bytes().as_ptr()));
        assert_eq!(
            q_view.as_bytes().as_ptr(),
            st_bytes[q_view.offset()..].as_ptr()
        );

        let norm_view = parsed
            .tensor("model.norm.weight")
            .expect("lookup BF16 tensor");
        assert_eq!(norm_view.quant(), QuantType::BF16);
        assert_eq!(norm_view.as_bytes(), &bf16_bytes[..]);

        let emb_view = parsed
            .tensor("model.embed_tokens.weight")
            .expect("lookup F16 tensor");
        assert_eq!(emb_view.quant(), QuantType::F16);
        assert_eq!(emb_view.as_bytes(), &f16_bytes[..]);
    }

    #[test]
    fn safetensors_infers_architecture_and_dimensions_without_metadata() {
        let mut payload = [0_u8; 4 * 16 + 2 * 16]; // 4 F32 (16B) + 8 F16 (16B)
        payload[..16].fill(0x11);
        payload[16..32].fill(0x22);
        let header = r#"{
            "model.layers.2.self_attn.q_proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
            "model.embed_tokens.weight":{"dtype":"F16","shape":[4,2],"data_offsets":[16,32]}
        }"#;
        let bytes = build_raw_safetensors(header, &payload[..32]);
        let st = SafeTensorsFile::parse(&bytes).expect("parse SafeTensors without __metadata__");
        assert_eq!(st.architecture(), "unknown");
        assert_eq!(st.metadata().layer_count(), 3);
        assert_eq!(st.metadata().embedding_dim(), 2);
        assert_eq!(st.metadata().vocab_size(), Some(4));
        assert_eq!(st.metadata().context_length(), 2048);
        assert!(st.custom_metadata().is_empty());

        // `model_type` fallback works when `general.architecture` is absent.
        let header_mt = r#"{
            "__metadata__":{"model_type":"qwen2"},
            "w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}
        }"#;
        let bytes_mt = build_raw_safetensors(header_mt, &[0_u8; 16]);
        let st_mt = SafeTensorsFile::from_bytes(&bytes_mt).expect("model_type fallback");
        assert_eq!(st_mt.architecture(), "qwen2");
    }

    #[test]
    fn safetensors_rejects_oversized_malformed_and_truncated_headers() {
        // 1. Truncated < 8 bytes
        assert!(matches!(
            SafeTensorsFile::from_bytes(&[1, 2, 3]),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 2. Zero header length
        let mut zero_hdr = vec![0_u8; 16];
        zero_hdr[0..8].copy_from_slice(&0_u64.to_le_bytes());
        assert!(matches!(
            SafeTensorsFile::from_bytes(&zero_hdr),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 3. Oversized > 100 MiB header DoS cap (rejects before allocating)
        let mut dos_hdr = [0_u8; 16];
        let oversized = (MAX_SAFETENSORS_HEADER_BYTES as u64) + 1;
        dos_hdr[0..8].copy_from_slice(&oversized.to_le_bytes());
        let err_dos = SafeTensorsFile::from_bytes(&dos_hdr).expect_err("must reject > 100 MiB");
        assert!(matches!(err_dos, K3Error::TensorBoundsError { .. }));
        assert!(err_dos.to_string().contains("exceeds"));

        // 4. Header length exceeds backing slice
        let mut eof_hdr = Vec::new();
        eof_hdr.extend_from_slice(&64_u64.to_le_bytes());
        eof_hdr.extend_from_slice(b"{}");
        assert!(matches!(
            SafeTensorsFile::from_bytes(&eof_hdr),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 5. Invalid UTF-8 in header
        let mut bad_utf8 = Vec::new();
        bad_utf8.extend_from_slice(&4_u64.to_le_bytes());
        bad_utf8.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(matches!(
            SafeTensorsFile::from_bytes(&bad_utf8),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 6. Malformed JSON / non-object JSON root / duplicate tensor keys
        for bad_json in [
            "{not valid json",
            "[1, 2, 3]",
            r#"{"__metadata__":[1,2]}"#,
            r#"{"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        ] {
            let bytes = build_raw_safetensors(bad_json, &[0_u8; 16]);
            assert!(
                matches!(
                    SafeTensorsFile::from_bytes(&bytes),
                    Err(K3Error::TensorBoundsError { .. })
                ),
                "expected TensorBoundsError for header: {bad_json}"
            );
        }
    }

    #[test]
    fn safetensors_rejects_bad_offsets_overlaps_alignment_shapes_and_dtypes() {
        // 1. Overlapping intervals [0..16) and [8..24)
        let overlap_hdr = r#"{
            "a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
            "b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}
        }"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(overlap_hdr, &[0_u8; 32])),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 2. Inverted offsets [16..0]
        let inverted_hdr = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[16,0]}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(inverted_hdr, &[0_u8; 32])),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 3. Out-of-bounds end offset
        let oob_hdr = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(oob_hdr, &[0_u8; 15])),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 4. Byte-length mismatch vs shape * dtype_size
        let mismatch_hdr = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,12]}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(mismatch_hdr, &[0_u8; 32])),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 5. Misaligned start offset (2 is not divisible by 4 for F32; 1 is not divisible by 2 for F16)
        let unaligned_f32 = r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[2,18]}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(unaligned_f32, &[0_u8; 32])),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let unaligned_f16 = r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[1,9]}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(unaligned_f16, &[0_u8; 16])),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // 6. Empty shape, zero dimension, and shape product overflow
        for bad_shape_hdr in [
            r#"{"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4,0],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,16]}}"#,
        ] {
            assert!(matches!(
                SafeTensorsFile::from_bytes(&build_raw_safetensors(bad_shape_hdr, &[0_u8; 16])),
                Err(K3Error::TensorBoundsError { .. })
            ));
        }

        // 7. Unsupported dtypes fail closed with K3Error::CapabilityMismatch
        for dtype in ["I8", "U8", "I32", "F64", "BOOL", "F8_E4M3"] {
            let hdr = format!(
                r#"{{"__metadata__":{{"general.architecture":"bert"}},"a":{{"dtype":"{dtype}","shape":[4],"data_offsets":[0,4]}}}}"#
            );
            let err = SafeTensorsFile::from_bytes(&build_raw_safetensors(&hdr, &[0_u8; 32]))
                .expect_err("unsupported dtype must fail closed");
            assert!(
                matches!(
                    err,
                    K3Error::CapabilityMismatch {
                        ref architecture,
                        ..
                    } if architecture == "bert"
                ),
                "unexpected error for dtype {dtype}: {err:?}"
            );
        }
    }

    #[test]
    fn safetensors_padding_gaps_slice_rows_and_empty_object() {
        // 1. Empty JSON object `{}` with zero tensors succeeds cleanly.
        let empty_st = build_raw_safetensors("{}", &[]);
        let parsed_empty = SafeTensorsFile::from_bytes(&empty_st).expect("empty safetensors");
        assert_eq!(parsed_empty.tensor_count(), 0);
        assert_eq!(parsed_empty.total_weight_bytes(), 0);
        assert_eq!(parsed_empty.tensor_data(), [0_u8; 0]);

        // 2. Non-overlapping tensors with alignment padding gaps + zero-copy slice_rows.
        let mut payload = vec![0_u8; 32];
        payload[0..4].fill(0xAA); // 2 x F16 = 4 bytes at [0..4)
        // [4..8) padding gap
        payload[8..24].fill(0x55); // [2, 2] F32 = 16 bytes at [8..24)
        payload[8..16].fill(0x11);
        let hdr = r#"{
            "__metadata__": {"general.architecture": "bert"},
            "norm": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
            "proj": {"dtype": "F32", "shape": [2, 2], "data_offsets": [8, 24]}
        }"#;
        let bytes = build_raw_safetensors(hdr, &payload);
        let st = SafeTensorsFile::from_bytes(&bytes).expect("valid padded safetensors");
        assert_eq!(st.total_weight_bytes(), 20);
        assert_eq!(st.data_offset(), 8 + hdr.len());

        let proj_view = st.tensor("proj").expect("proj tensor");
        let row1 = proj_view.slice_rows(1, 1).expect("zero-copy row slice");
        assert_eq!(row1.shape(), &[2, 1]);
        assert_eq!(row1.as_bytes(), &[0x55_u8; 8]);
        assert!(bytes.as_ptr_range().contains(&row1.as_bytes().as_ptr()));

        // 3. Duplicate __metadata__ rejected.
        let dup_meta_hdr = r#"{"__metadata__":{"a":"1"},"__metadata__":{"b":"2"}}"#;
        assert!(matches!(
            SafeTensorsFile::from_bytes(&build_raw_safetensors(dup_meta_hdr, &[])),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
