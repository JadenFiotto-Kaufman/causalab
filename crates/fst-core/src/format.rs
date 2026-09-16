//! The safetensors container grammar.
//!
//! An object is an 8-byte little-endian header length `N`, `N` bytes of JSON,
//! then the data section. The JSON maps tensor names to `{dtype, shape,
//! data_offsets}` with offsets relative to the data section, plus an optional
//! `__metadata__` map of strings. Format reference:
//! <https://github.com/huggingface/safetensors#format>.
//!
//! Two things make output from this module indistinguishable from the
//! reference library's: tensors are laid out by **descending dtype then
//! ascending name** (the reference sorts by its `Dtype` enum's order, which
//! [`Dtype`] reproduces variant for variant), and the header JSON is padded
//! with spaces to a multiple of eight bytes. Reading is strict: sizes must
//! match dtype and shape, and the ranges must tile the data section from
//! zero, exactly as the reference validates.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::FormatError;

/// Bytes of the little-endian header-length prefix.
pub const LENGTH_PREFIX_BYTES: usize = 8;
/// The reference library refuses headers larger than this.
pub const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;
/// The one key of the header that is not a tensor.
pub const METADATA_KEY: &str = "__metadata__";

/// Element types the format defines.
///
/// Declared in the reference library's order so that `Ord` *is* its layout
/// rank: the reference sorts tensors by descending dtype, then name.
#[allow(non_camel_case_types, missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Dtype {
    BOOL,
    F4,
    F6_E2M3,
    F6_E3M2,
    U8,
    I8,
    F8_E5M2,
    F8_E4M3,
    F8_E8M0,
    F8_E4M3FNUZ,
    F8_E5M2FNUZ,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    C64,
    F64,
    I64,
    U64,
}

impl Dtype {
    /// Width of one element in bits. `BOOL` is stored as a byte.
    pub const fn bits(self) -> u32 {
        match self {
            Dtype::F4 => 4,
            Dtype::F6_E2M3 | Dtype::F6_E3M2 => 6,
            Dtype::BOOL
            | Dtype::U8
            | Dtype::I8
            | Dtype::F8_E5M2
            | Dtype::F8_E4M3
            | Dtype::F8_E8M0
            | Dtype::F8_E4M3FNUZ
            | Dtype::F8_E5M2FNUZ => 8,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 16,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 32,
            Dtype::C64 | Dtype::F64 | Dtype::I64 | Dtype::U64 => 64,
        }
    }

    /// The name the header spells this dtype with.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::BOOL => "BOOL",
            Dtype::F4 => "F4",
            Dtype::F6_E2M3 => "F6_E2M3",
            Dtype::F6_E3M2 => "F6_E3M2",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::F8_E5M2 => "F8_E5M2",
            Dtype::F8_E4M3 => "F8_E4M3",
            Dtype::F8_E8M0 => "F8_E8M0",
            Dtype::F8_E4M3FNUZ => "F8_E4M3FNUZ",
            Dtype::F8_E5M2FNUZ => "F8_E5M2FNUZ",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::F32 => "F32",
            Dtype::C64 => "C64",
            Dtype::F64 => "F64",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
        }
    }

    /// Every dtype, in layout-rank order.
    pub const ALL: [Dtype; 22] = [
        Dtype::BOOL,
        Dtype::F4,
        Dtype::F6_E2M3,
        Dtype::F6_E3M2,
        Dtype::U8,
        Dtype::I8,
        Dtype::F8_E5M2,
        Dtype::F8_E4M3,
        Dtype::F8_E8M0,
        Dtype::F8_E4M3FNUZ,
        Dtype::F8_E5M2FNUZ,
        Dtype::I16,
        Dtype::U16,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I32,
        Dtype::U32,
        Dtype::F32,
        Dtype::C64,
        Dtype::F64,
        Dtype::I64,
        Dtype::U64,
    ];

    /// Parse a header dtype string.
    pub fn parse(name: &str) -> Option<Dtype> {
        Dtype::ALL.iter().copied().find(|d| d.name() == name)
    }

    /// Bytes `numel` elements occupy, or the element count that does not
    /// fill whole bytes (sub-byte dtypes only).
    pub fn nbytes(self, numel: u64) -> Result<u64, u64> {
        let bits = u64::from(self.bits()) * numel;
        if bits.is_multiple_of(8) {
            Ok(bits / 8)
        } else {
            Err(numel)
        }
    }
}

/// Product of a shape's extents; an empty shape is one scalar.
pub fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// One tensor's header entry as written: dtype, shape, and its byte range
/// **relative to the data section**.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Element type.
    pub dtype: Dtype,
    /// Extents, outermost first.
    pub shape: Vec<u64>,
    /// `[start, end)` within the data section.
    pub data_offsets: [u64; 2],
}

impl TensorInfo {
    /// Bytes this tensor occupies.
    pub fn nbytes(&self) -> u64 {
        self.data_offsets[1] - self.data_offsets[0]
    }

    /// Elements this tensor holds.
    pub fn numel(&self) -> u64 {
        numel(&self.shape)
    }
}

/// A parsed header: where every tensor is, what it is, and the metadata.
///
/// Tensor ranges are kept relative to the data section as the header spells
/// them; [`Layout::data_start`] converts to absolute object offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Length of the JSON header, padding included (the prefix's value).
    pub header_len: u64,
    /// Tensors by name.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// The `__metadata__` entries in the order the header lists them. On
    /// output this crate keeps the writer's order; the reference library
    /// serializes a `HashMap`, so its order varies per process and byte
    /// equality with multi-key metadata is not well defined. `None` when the
    /// header has no `__metadata__` key at all, which the reference writes
    /// differently from an empty map.
    pub metadata: Option<Vec<(String, String)>>,
}

impl Layout {
    /// Absolute offset of the data section's first byte.
    pub fn data_start(&self) -> u64 {
        LENGTH_PREFIX_BYTES as u64 + self.header_len
    }

    /// Bytes the data section holds (the end of the last tensor).
    pub fn data_len(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| t.data_offsets[1])
            .max()
            .unwrap_or(0)
    }

    /// Total object length, header and data.
    pub fn object_len(&self) -> u64 {
        self.data_start() + self.data_len()
    }

    /// Absolute `[start, end)` of one tensor's bytes within the object.
    pub fn absolute_range(&self, name: &str) -> Option<[u64; 2]> {
        let info = self.tensors.get(name)?;
        let base = self.data_start();
        Some([base + info.data_offsets[0], base + info.data_offsets[1]])
    }

    /// One metadata value by key.
    pub fn metadata_get(&self, key: &str) -> Option<&str> {
        self.metadata
            .as_ref()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Tensors in the order their bytes appear in the data section. Empty
    /// tensors share an offset with their successor; ties fall back to the
    /// reference's layout rank (descending dtype, then name).
    pub fn in_data_order(&self) -> Vec<(&str, &TensorInfo)> {
        let mut entries: Vec<(&str, &TensorInfo)> = self
            .tensors
            .iter()
            .map(|(name, info)| (name.as_str(), info))
            .collect();
        entries.sort_by(|(a_name, a), (b_name, b)| {
            a.data_offsets[0]
                .cmp(&b.data_offsets[0])
                .then(a.data_offsets[1].cmp(&b.data_offsets[1]))
                .then(b.dtype.cmp(&a.dtype))
                .then(a_name.cmp(b_name))
        });
        entries
    }
}

/// Parse the header-length prefix. `bytes` must hold at least eight bytes.
pub fn parse_length_prefix(bytes: &[u8]) -> Result<u64, FormatError> {
    let prefix: [u8; LENGTH_PREFIX_BYTES] = bytes
        .get(..LENGTH_PREFIX_BYTES)
        .and_then(|p| p.try_into().ok())
        .ok_or(FormatError::TruncatedLength)?;
    let len = u64::from_le_bytes(prefix);
    if len > MAX_HEADER_BYTES {
        return Err(FormatError::HeaderTooLarge {
            len,
            max: MAX_HEADER_BYTES,
        });
    }
    Ok(len)
}

/// Parse a header from its JSON bytes (the `header_len` bytes after the
/// prefix, padding included) and validate it the way the reference does.
pub fn parse_header(json: &[u8]) -> Result<Layout, FormatError> {
    let value: serde_json::Value = serde_json::from_slice(json)?;
    let serde_json::Value::Object(object) = value else {
        return Err(FormatError::NotAnObject);
    };
    let mut tensors = BTreeMap::new();
    let mut metadata = None;
    for (name, entry) in object {
        if name == METADATA_KEY {
            let serde_json::Value::Object(map) = entry else {
                return Err(FormatError::MetadataNotStrings);
            };
            let mut entries = Vec::with_capacity(map.len());
            for (key, value) in map {
                let serde_json::Value::String(value) = value else {
                    return Err(FormatError::MetadataNotStrings);
                };
                entries.push((key, value));
            }
            metadata = Some(entries);
            continue;
        }
        let info = parse_entry(&name, entry)?;
        tensors.insert(name, info);
    }
    validate_tiling(&tensors)?;
    Ok(Layout {
        header_len: json.len() as u64,
        tensors,
        metadata,
    })
}

/// Parse one object: prefix, header, and enough of the data section to know
/// it is all there. `bytes` is the whole object or at least its header;
/// `object_len` is the full object length when known (a file's size), so a
/// truncated data section is refused here rather than at the first read.
pub fn parse_object(bytes: &[u8], object_len: Option<u64>) -> Result<Layout, FormatError> {
    let header_len = parse_length_prefix(bytes)?;
    let end = LENGTH_PREFIX_BYTES as u64 + header_len;
    let available = bytes.len() as u64;
    if available < end {
        return Err(FormatError::TruncatedHeader {
            declared: header_len,
            available,
        });
    }
    let layout = parse_header(&bytes[LENGTH_PREFIX_BYTES..end as usize])?;
    let total = object_len.unwrap_or(available);
    if total < layout.object_len() {
        return Err(FormatError::TruncatedData {
            promised: layout.data_len(),
            available: total.saturating_sub(layout.data_start()),
        });
    }
    Ok(layout)
}

fn parse_entry(name: &str, entry: serde_json::Value) -> Result<TensorInfo, FormatError> {
    #[derive(Deserialize)]
    struct Raw {
        dtype: String,
        shape: Vec<u64>,
        data_offsets: [u64; 2],
    }
    let raw: Raw = serde_json::from_value(entry).map_err(|err| FormatError::InvalidEntry {
        name: name.to_owned(),
        reason: err.to_string(),
    })?;
    let dtype = Dtype::parse(&raw.dtype).ok_or_else(|| FormatError::UnknownDtype {
        name: name.to_owned(),
        dtype: raw.dtype.clone(),
    })?;
    let [start, end] = raw.data_offsets;
    if end < start {
        return Err(FormatError::InvalidEntry {
            name: name.to_owned(),
            reason: format!("data_offsets end {end} precedes start {start}"),
        });
    }
    let count = numel(&raw.shape);
    let expected = dtype
        .nbytes(count)
        .map_err(|numel| FormatError::PartialByte {
            name: name.to_owned(),
            dtype,
            numel,
            bits: dtype.bits(),
        })?;
    if end - start != expected {
        return Err(FormatError::SizeMismatch {
            name: name.to_owned(),
            dtype,
            numel: count,
            expected,
            span: end - start,
        });
    }
    Ok(TensorInfo {
        dtype,
        shape: raw.shape,
        data_offsets: raw.data_offsets,
    })
}

/// The reference's `validate`: sorted by start, the ranges tile `[0, len)`.
fn validate_tiling(tensors: &BTreeMap<String, TensorInfo>) -> Result<(), FormatError> {
    let mut entries: Vec<(&String, &TensorInfo)> = tensors.iter().collect();
    entries.sort_by_key(|(name, info)| (info.data_offsets[0], info.data_offsets[1], *name));
    let mut expected = 0;
    for (name, info) in entries {
        let [start, end] = info.data_offsets;
        if start != expected {
            return Err(FormatError::OffsetGap {
                name: name.clone(),
                start,
                expected,
            });
        }
        expected = end;
    }
    Ok(())
}

/// What a caller wants written: a name, a dtype, a shape, and how many bytes
/// its buffer holds (the buffer itself stays with the caller — the header is
/// built without touching data).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Extents, outermost first.
    pub shape: Vec<u64>,
    /// Bytes the caller's buffer holds; must equal `dtype x shape`.
    pub nbytes: u64,
}

/// A built header and the order the caller's buffers follow it in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltHeader {
    /// Length prefix, JSON, padding: the object's first `bytes.len()` bytes.
    pub bytes: Vec<u8>,
    /// Indices into the caller's [`TensorSpec`] slice, in data-section order.
    pub order: Vec<usize>,
    /// The layout the header describes, as a reader would parse it back.
    pub layout: Layout,
}

/// Build the header for `specs` and `metadata`, byte-identical to the
/// reference library's for the same tensors.
pub fn build_header(
    specs: &[TensorSpec],
    metadata: Option<&[(String, String)]>,
) -> Result<BuiltHeader, FormatError> {
    let mut seen = std::collections::HashSet::with_capacity(specs.len());
    for spec in specs {
        if spec.name == METADATA_KEY {
            return Err(FormatError::ReservedName);
        }
        if !seen.insert(spec.name.as_str()) {
            return Err(FormatError::DuplicateName {
                name: spec.name.clone(),
            });
        }
        let count = numel(&spec.shape);
        let expected = spec
            .dtype
            .nbytes(count)
            .map_err(|numel| FormatError::PartialByte {
                name: spec.name.clone(),
                dtype: spec.dtype,
                numel,
                bits: spec.dtype.bits(),
            })?;
        if expected != spec.nbytes {
            return Err(FormatError::BufferMismatch {
                name: spec.name.clone(),
                dtype: spec.dtype,
                shape: spec.shape.clone(),
                expected,
                actual: spec.nbytes,
            });
        }
    }
    // the reference's order: descending dtype, then ascending name
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.sort_by(|&a, &b| {
        specs[b]
            .dtype
            .cmp(&specs[a].dtype)
            .then_with(|| specs[a].name.cmp(&specs[b].name))
    });

    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        // insertion order kept: serde_json's `preserve_order` feature
        let map: serde_json::Map<String, serde_json::Value> = metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        header.insert(METADATA_KEY.to_owned(), serde_json::Value::Object(map));
    }
    let mut tensors = BTreeMap::new();
    let mut offset = 0u64;
    for &index in &order {
        let spec = &specs[index];
        let info = TensorInfo {
            dtype: spec.dtype,
            shape: spec.shape.clone(),
            data_offsets: [offset, offset + spec.nbytes],
        };
        offset += spec.nbytes;
        header.insert(spec.name.clone(), serde_json::to_value(&info)?);
        tensors.insert(spec.name.clone(), info);
    }
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let padding = (8 - json.len() % 8) % 8;
    json.extend(std::iter::repeat_n(b' ', padding));

    let mut bytes = Vec::with_capacity(LENGTH_PREFIX_BYTES + json.len());
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&json);
    let layout = Layout {
        header_len: json.len() as u64,
        tensors,
        metadata: metadata.map(<[_]>::to_vec),
    };
    Ok(BuiltHeader {
        bytes,
        order,
        layout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    use crate::fixtures::{MIXED, SINGLE_U8};

    fn specs_of(layout: &Layout) -> Vec<TensorSpec> {
        // in name order, deliberately not data order: build_header must sort
        layout
            .tensors
            .iter()
            .map(|(name, info)| TensorSpec {
                name: name.clone(),
                dtype: info.dtype,
                shape: info.shape.clone(),
                nbytes: info.nbytes(),
            })
            .collect()
    }

    #[test]
    fn dtype_order_is_the_reference_enum_order() {
        // safetensors 0.8.0, `Dtype` variants top to bottom
        let names: Vec<&str> = Dtype::ALL.iter().map(|d| d.name()).collect();
        assert_eq!(
            names,
            [
                "BOOL",
                "F4",
                "F6_E2M3",
                "F6_E3M2",
                "U8",
                "I8",
                "F8_E5M2",
                "F8_E4M3",
                "F8_E8M0",
                "F8_E4M3FNUZ",
                "F8_E5M2FNUZ",
                "I16",
                "U16",
                "F16",
                "BF16",
                "I32",
                "U32",
                "F32",
                "C64",
                "F64",
                "I64",
                "U64",
            ]
        );
        for pair in Dtype::ALL.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        for dtype in Dtype::ALL {
            assert_eq!(Dtype::parse(dtype.name()), Some(dtype));
        }
    }

    #[test]
    fn reference_fixture_parses_and_rebuilds_byte_identical() {
        for fixture in [MIXED, SINGLE_U8] {
            let layout = parse_object(fixture, None).unwrap_or_else(|e| panic!("{e}"));
            let header_end = layout.data_start() as usize;
            let built = build_header(&specs_of(&layout), layout.metadata.as_deref())
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                built.bytes,
                &fixture[..header_end],
                "header bytes differ from the reference"
            );
            assert_eq!(built.layout, layout);
        }
    }

    #[test]
    fn mixed_fixture_layout() {
        let layout = parse_object(MIXED, None).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(layout.metadata_get("format"), Some("pt"));
        let first = layout
            .metadata
            .as_ref()
            .and_then(|m| m.first())
            .map(|(k, _)| k.as_str());
        assert_eq!(first, Some("note"), "metadata keeps the writer's order");
        let order: Vec<&str> = layout.in_data_order().into_iter().map(|(n, _)| n).collect();
        // I64 > F32 > BF16 > F16 > BOOL in rank; ties would be by name
        assert_eq!(order, ["a/bias", "b/weight", "a/half", "empty", "c/flag"]);
        assert_eq!(layout.absolute_range("a/half"), Some([408 + 48, 408 + 52]));
        assert_eq!(layout.object_len() as usize, MIXED.len());
    }

    #[test]
    fn truncations_are_named() {
        assert!(matches!(
            parse_object(&MIXED[..4], None),
            Err(FormatError::TruncatedLength)
        ));
        assert!(matches!(
            parse_object(&MIXED[..40], None),
            Err(FormatError::TruncatedHeader {
                declared: 400,
                available: 40
            })
        ));
        assert!(matches!(
            parse_object(MIXED, Some(MIXED.len() as u64 - 1)),
            Err(FormatError::TruncatedData {
                promised: 54,
                available: 53
            })
        ));
    }

    #[test]
    fn header_grammar_is_strict() {
        let bad = |json: &str| parse_header(json.as_bytes());
        assert!(matches!(bad("[]"), Err(FormatError::NotAnObject)));
        assert!(matches!(
            bad(r#"{"__metadata__":{"a":1}}"#),
            Err(FormatError::MetadataNotStrings)
        ));
        assert!(matches!(
            bad(r#"{"t":{"dtype":"Q7","shape":[1],"data_offsets":[0,1]}}"#),
            Err(FormatError::UnknownDtype { .. })
        ));
        assert!(matches!(
            bad(r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#),
            Err(FormatError::SizeMismatch {
                expected: 8,
                span: 4,
                ..
            })
        ));
        assert!(matches!(
            bad(r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#),
            Err(FormatError::OffsetGap {
                start: 4,
                expected: 0,
                ..
            })
        ));
        assert!(matches!(
            bad(r#"{"t":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}"#),
            Err(FormatError::PartialByte {
                numel: 3,
                bits: 4,
                ..
            })
        ));
    }

    #[test]
    fn build_refuses_bad_specs() {
        let spec = |name: &str, nbytes| TensorSpec {
            name: name.into(),
            dtype: Dtype::F32,
            shape: vec![2],
            nbytes,
        };
        assert!(matches!(
            build_header(&[spec("__metadata__", 8)], None),
            Err(FormatError::ReservedName)
        ));
        assert!(matches!(
            build_header(&[spec("a", 8), spec("a", 8)], None),
            Err(FormatError::DuplicateName { .. })
        ));
        assert!(matches!(
            build_header(&[spec("a", 7)], None),
            Err(FormatError::BufferMismatch {
                expected: 8,
                actual: 7,
                ..
            })
        ));
    }

    fn arb_spec() -> impl Strategy<Value = TensorSpec> {
        (
            "[a-z][a-z0-9_./]{0,12}",
            proptest::sample::select(Dtype::ALL.to_vec()),
            proptest::collection::vec(0u64..5, 0..4),
        )
            .prop_filter_map("whole bytes", |(name, dtype, shape)| {
                let nbytes = dtype.nbytes(numel(&shape)).ok()?;
                Some(TensorSpec {
                    name,
                    dtype,
                    shape,
                    nbytes,
                })
            })
    }

    proptest! {
        #[test]
        fn build_then_parse_is_identity(
            specs in proptest::collection::vec(arb_spec(), 0..12)
                .prop_map(|mut v| { v.sort_by(|a, b| a.name.cmp(&b.name)); v.dedup_by(|a, b| a.name == b.name); v }),
            metadata in proptest::option::of(
                proptest::collection::btree_map("[a-z]{1,6}", "[ -~]{0,10}", 0..3)
                    .prop_map(|m| m.into_iter().collect::<Vec<_>>())
            ),
        ) {
            let built = build_header(&specs, metadata.as_deref()).unwrap_or_else(|e| panic!("{e}"));
            // the prefix, the JSON, eight-byte alignment
            prop_assert_eq!(built.bytes.len() % 8, 0);
            let layout = parse_object(&built.bytes, Some(built.layout.object_len())).unwrap_or_else(|e| panic!("{e}"));
            prop_assert_eq!(&layout, &built.layout);
            // data order is descending dtype then ascending name, and tiles from zero
            let mut offset = 0;
            for window in built.order.windows(2) {
                let (a, b) = (&specs[window[0]], &specs[window[1]]);
                prop_assert!(a.dtype > b.dtype || (a.dtype == b.dtype && a.name < b.name));
            }
            for &i in &built.order {
                let info = &layout.tensors[&specs[i].name];
                prop_assert_eq!(info.data_offsets, [offset, offset + specs[i].nbytes]);
                offset += specs[i].nbytes;
            }
            prop_assert_eq!(layout.data_len(), offset);
        }
    }
}
