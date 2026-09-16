//! One error type per domain. Nothing here is a bare string; every variant
//! names what was expected and what was found, so a caller — or the Python
//! layer above — can map it to a typed exception.

use std::path::PathBuf;

use crate::format::Dtype;

/// The container grammar was violated, on read or on build.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    /// Fewer than the eight length-prefix bytes were available.
    #[error("not a safetensors object: fewer than 8 bytes of header length")]
    TruncatedLength,
    /// The header length prefix exceeds the reference library's limit.
    #[error("header length {len} exceeds the {max}-byte limit")]
    HeaderTooLarge {
        /// The declared header length.
        len: u64,
        /// The limit it exceeded.
        max: u64,
    },
    /// The object ends inside its declared header.
    #[error("object of {available} bytes ends inside its {declared}-byte header")]
    TruncatedHeader {
        /// The header length the prefix declares.
        declared: u64,
        /// How many bytes were actually there.
        available: u64,
    },
    /// The header is not valid JSON, or not the JSON the grammar expects.
    #[error("header is not valid safetensors JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// The header's top level is not a JSON object.
    #[error("header must be a JSON object")]
    NotAnObject,
    /// A tensor entry lacks a field or has one of the wrong shape.
    #[error("tensor {name:?}: {reason}")]
    InvalidEntry {
        /// The tensor's key in the header.
        name: String,
        /// What was wrong with it.
        reason: String,
    },
    /// The dtype string is not one the format defines.
    #[error("tensor {name:?}: unknown dtype {dtype:?}")]
    UnknownDtype {
        /// The tensor's key in the header.
        name: String,
        /// The dtype string as written.
        dtype: String,
    },
    /// A tensor's byte range disagrees with its dtype and shape.
    #[error(
        "tensor {name:?}: data_offsets span {span} bytes but {dtype:?} x {numel} elements need {expected}"
    )]
    SizeMismatch {
        /// The tensor's key in the header.
        name: String,
        /// Its declared dtype.
        dtype: Dtype,
        /// The number of elements its shape implies.
        numel: u64,
        /// The bytes its dtype and shape need.
        expected: u64,
        /// The bytes its offsets span.
        span: u64,
    },
    /// The tensors' byte ranges do not tile the data section from zero.
    #[error(
        "tensor {name:?} starts at data offset {start} but the previous tensor ended at {expected}"
    )]
    OffsetGap {
        /// The tensor whose start is out of place.
        name: String,
        /// Where it starts.
        start: u64,
        /// Where it should have started.
        expected: u64,
    },
    /// A sub-byte dtype's element count does not fill whole bytes.
    #[error("tensor {name:?}: {numel} elements of {dtype:?} ({bits} bits) do not fill whole bytes")]
    PartialByte {
        /// The tensor's key in the header.
        name: String,
        /// Its dtype.
        dtype: Dtype,
        /// Its element count.
        numel: u64,
        /// The dtype's width.
        bits: u32,
    },
    /// `__metadata__` is the one reserved key.
    #[error("'__metadata__' is reserved and cannot name a tensor")]
    ReservedName,
    /// Metadata must map strings to strings.
    #[error("__metadata__ must map strings to strings")]
    MetadataNotStrings,
    /// The data section is shorter than the header promises.
    #[error("object holds {available} data bytes but the header promises {promised}")]
    TruncatedData {
        /// Bytes the header's offsets cover.
        promised: u64,
        /// Bytes present after the header.
        available: u64,
    },
    /// A caller-supplied buffer's length does not match its declared tensor.
    #[error(
        "tensor {name:?}: buffer holds {actual} bytes but {dtype:?} x shape {shape:?} needs {expected}"
    )]
    BufferMismatch {
        /// The tensor's key.
        name: String,
        /// Its dtype.
        dtype: Dtype,
        /// Its shape.
        shape: Vec<u64>,
        /// Bytes the dtype and shape need.
        expected: u64,
        /// Bytes the buffer holds.
        actual: u64,
    },
    /// The same tensor name was given twice.
    #[error("tensor {name:?} given twice")]
    DuplicateName {
        /// The repeated name.
        name: String,
    },
}

/// Moving bytes failed, or was asked of a backend that cannot.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// The operating system refused.
    #[error("{op} on {path}: {source}")]
    Io {
        /// What was being done.
        op: &'static str,
        /// To which file.
        path: PathBuf,
        /// The OS's answer.
        #[source]
        source: std::io::Error,
    },
    /// A read returned fewer bytes than the range asked for.
    #[error("{path}: read at offset {offset} returned {got} of {wanted} bytes")]
    ShortRead {
        /// The file.
        path: PathBuf,
        /// Where the read started.
        offset: u64,
        /// How much was asked for.
        wanted: usize,
        /// How much came back.
        got: usize,
    },
    /// A write accepted fewer bytes than offered and the backend cannot resume.
    #[error("{path}: write accepted {got} of {wanted} bytes")]
    ShortWrite {
        /// The file.
        path: PathBuf,
        /// How much was offered.
        wanted: usize,
        /// How much was taken.
        got: usize,
    },
    /// A range lies outside the object.
    #[error("{path}: range {start}..{end} lies outside an object of {len} bytes")]
    OutOfRange {
        /// The file.
        path: PathBuf,
        /// Range start.
        start: u64,
        /// Range end.
        end: u64,
        /// The object's length.
        len: u64,
    },
    /// This backend does not do that.
    #[error("{backend} cannot {op}")]
    Unsupported {
        /// The backend's name.
        backend: &'static str,
        /// The operation it lacks.
        op: &'static str,
    },
    /// The simulated backend fired a scheduled fault.
    #[error("simulated fault: {0}")]
    Simulated(String),
}

/// A request the planner cannot honour.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// Nothing to plan.
    #[error("the request names no files")]
    Empty,
    /// The destination cannot hold what the request reads.
    #[error("destination has {available} bytes free but the request needs {needed}")]
    DoesNotFit {
        /// Bytes the destination reports free.
        available: u64,
        /// Bytes the request would place there.
        needed: u64,
    },
}

/// A calibration profile that cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// The document is not JSON, or not the JSON the schema expects (an
    /// unknown field, a missing required one, a wrong type).
    #[error("profile is not valid JSON for the schema: {0}")]
    Json(#[from] serde_json::Error),
    /// The file could not be read.
    #[error("reading profile {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The OS's answer.
        #[source]
        source: std::io::Error,
    },
    /// The document follows a schema this build does not read.
    #[error("profile schema_version {found} is not supported (this build reads {supported})")]
    UnsupportedSchemaVersion {
        /// What the document declares.
        found: u32,
        /// What this build reads.
        supported: u32,
    },
    /// A `storage` key is not a storage class name.
    #[error(
        "unknown storage class {key:?} (expected one of LocalBlock, Nfs, Fuse, Ram, OtherNetwork, Other)"
    )]
    UnknownStorageClass {
        /// The key as written.
        key: String,
    },
    /// A throughput or cost is zero, negative, or not a number.
    #[error("profile entry {entry}: {field} must be positive, got {value}")]
    NonPositiveRate {
        /// Which entry (`Nfs`, `mount "/mnt/home"`, `device`).
        entry: String,
        /// Which field.
        field: &'static str,
        /// What was given.
        value: f64,
    },
    /// An aggregate table's rows are not strictly increasing in files in flight.
    #[error(
        "profile entry {entry}: aggregate_gbps row {index} (files_in_flight {files_in_flight}) is not strictly after the previous row"
    )]
    AggregateNotSorted {
        /// Which entry.
        entry: String,
        /// The offending row.
        index: usize,
        /// Its files-in-flight count.
        files_in_flight: u32,
    },
    /// A table that must have at least one row has none.
    #[error("profile entry {entry}: {table} is empty")]
    EmptyTable {
        /// Which entry.
        entry: String,
        /// Which table.
        table: &'static str,
    },
}

/// A selection that does not fit its tensor, or cannot be read as bytes.
#[derive(Debug, thiserror::Error)]
pub enum SelectError {
    /// The number of ranges differs from the number of dimensions.
    #[error("selection has {ranges} ranges for a {ndim}-d shape")]
    NdimMismatch {
        /// Dimensions of the shape.
        ndim: usize,
        /// Ranges given.
        ranges: usize,
    },
    /// A range is inverted or runs past its dimension.
    #[error("selection {start}..{end} on dim {dim} lies outside its extent {extent}")]
    OutOfBounds {
        /// Which dimension.
        dim: usize,
        /// Range start.
        start: u64,
        /// Range end (exclusive).
        end: u64,
        /// The dimension's extent.
        extent: u64,
    },
    /// The shard dimension does not exist.
    #[error("shard dim {dim} is out of range for a {ndim}-d shape")]
    DimOutOfRange {
        /// The dimension asked for.
        dim: usize,
        /// Dimensions of the shape.
        ndim: usize,
    },
    /// The rank is not below the world size, or the world is empty.
    #[error("shard rank {rank} is not within a world of {world}")]
    BadRank {
        /// The rank asked for.
        rank: u64,
        /// The world size.
        world: u64,
    },
    /// The dimension cannot be split into equal shards.
    #[error("dim {dim} of extent {extent} does not divide into {world} equal shards")]
    DoesNotDivide {
        /// Which dimension.
        dim: usize,
        /// Its extent.
        extent: u64,
        /// The world size.
        world: u64,
    },
    /// The shape's element count, in bits, does not fit a `u64`.
    #[error("shape {shape:?} is too large to address")]
    TooLarge {
        /// The shape.
        shape: Vec<u64>,
    },
    /// A run of a sub-byte dtype does not start and end on byte boundaries.
    #[error(
        "{dtype:?} ({bits} bits): a run of {elems} elements from element {start_elems} does not start and end on a byte boundary"
    )]
    SubByteMisaligned {
        /// The dtype.
        dtype: Dtype,
        /// Its width.
        bits: u32,
        /// The run's first element.
        start_elems: u64,
        /// Elements in the run.
        elems: u64,
    },
}
