//! N-d selections over a tensor, the byte runs they read, and how those runs
//! are coalesced into I/O.
//!
//! Tensor-parallel loading reads a slice of a tensor per rank. A slice along
//! the outermost dimension is one contiguous byte range; a slice along an
//! inner dimension is many short runs with gaps — a row-parallel `Linear`
//! weight `[out, in]` sharded on `in` over 8 ranks in bf16 is `out` runs of
//! `in / 8 * 2` bytes each. On NFS every read call costs a round trip
//! however small it is, so those runs are merged into larger reads and the
//! wanted bytes are then placed into the contiguous destination.
//!
//! Three steps, each pure and tested on its own:
//!
//! * [`Selection`]: a box over the tensor's shape, one half-open range per
//!   dimension, validated against the shape. [`Selection::full`] is the
//!   whole tensor; [`Selection::shard`] is one of `world` equal parts along
//!   a dimension, the slice `torch.narrow` (or `torch.chunk`) would give.
//! * [`Selection::runs`]: the contiguous byte runs of the box in row-major
//!   order. Dimensions fully selected from the inside out merge into the
//!   run, so a box that is full on every inner dimension is one run per
//!   outer index and a box full on every dimension is a single run. The
//!   runs concatenated in order are exactly the bytes of the sliced tensor.
//! * [`coalesce`]: neighbouring runs whose gap is at most
//!   [`CoalescePolicy::max_gap_bytes`] are merged into one [`Read`], up to
//!   [`CoalescePolicy::max_read_bytes`] per read; each read carries the
//!   [`Placement`]s that put its wanted bytes into the destination. The
//!   policy comes from the calibration profile through
//!   [`crate::plan::coalesce_policy`]: a gap is worth reading through when
//!   it moves in less time than a read call's fixed cost.
//!
//! Sub-byte dtypes (`F4`, `F6_*`) pack several elements per byte, so a run
//! must start and end on a byte boundary; a selection that does not is
//! refused with [`SelectError::SubByteMisaligned`] rather than rounded.

use std::ops::Range;

use crate::error::SelectError;
use crate::format::Dtype;
use crate::storage::ReadRange;

/// A box over a tensor: one half-open range of indices per dimension.
///
/// Built with [`Selection::new`], [`Selection::full`] or
/// [`Selection::shard`]; every constructor validates against the shape, so
/// a `Selection` always lies inside the tensor it was made for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Selection {
    shape: Vec<u64>,
    ranges: Vec<Range<u64>>,
}

impl Selection {
    /// A box of `ranges` over `shape`: one range per dimension, each within
    /// `0..extent`. An empty shape is a scalar and takes an empty list.
    pub fn new(shape: &[u64], ranges: Vec<Range<u64>>) -> Result<Selection, SelectError> {
        if ranges.len() != shape.len() {
            return Err(SelectError::NdimMismatch {
                ndim: shape.len(),
                ranges: ranges.len(),
            });
        }
        // every later product — element offsets, then bits — must fit a u64
        let numel = shape
            .iter()
            .try_fold(1u64, |acc, &extent| acc.checked_mul(extent))
            .and_then(|numel| numel.checked_mul(64));
        if numel.is_none() {
            return Err(SelectError::TooLarge {
                shape: shape.to_vec(),
            });
        }
        for (dim, (range, &extent)) in ranges.iter().zip(shape).enumerate() {
            if range.start > range.end || range.end > extent {
                return Err(SelectError::OutOfBounds {
                    dim,
                    start: range.start,
                    end: range.end,
                    extent,
                });
            }
        }
        Ok(Selection {
            shape: shape.to_vec(),
            ranges,
        })
    }

    /// The whole tensor.
    pub fn full(shape: &[u64]) -> Result<Selection, SelectError> {
        Selection::new(shape, shape.iter().map(|&extent| 0..extent).collect())
    }

    /// Part `rank` of `world` equal parts along `dim`, the other dimensions
    /// whole: what `torch.chunk(t, world, dim)[rank]` selects when the
    /// dimension divides evenly. A dimension that does not divide is
    /// [`SelectError::DoesNotDivide`]; the caller decides how to pad.
    pub fn shard(
        shape: &[u64],
        dim: usize,
        rank: u64,
        world: u64,
    ) -> Result<Selection, SelectError> {
        let Some(&extent) = shape.get(dim) else {
            return Err(SelectError::DimOutOfRange {
                dim,
                ndim: shape.len(),
            });
        };
        if world == 0 || rank >= world {
            return Err(SelectError::BadRank { rank, world });
        }
        if !extent.is_multiple_of(world) {
            return Err(SelectError::DoesNotDivide { dim, extent, world });
        }
        let per = extent / world;
        let ranges = shape
            .iter()
            .enumerate()
            .map(|(d, &e)| {
                if d == dim {
                    rank * per..(rank + 1) * per
                } else {
                    0..e
                }
            })
            .collect();
        Selection::new(shape, ranges)
    }

    /// The tensor's shape this box lies in.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// The index range per dimension.
    pub fn ranges(&self) -> &[Range<u64>] {
        &self.ranges
    }

    /// The box's own shape: its extent per dimension.
    pub fn box_shape(&self) -> Vec<u64> {
        self.ranges.iter().map(|r| r.end - r.start).collect()
    }

    /// Elements in the box.
    pub fn numel(&self) -> u64 {
        self.ranges.iter().map(|r| r.end - r.start).product()
    }

    /// Whether the box is the whole tensor.
    pub fn is_full(&self) -> bool {
        self.ranges
            .iter()
            .zip(&self.shape)
            .all(|(r, &extent)| r.start == 0 && r.end == extent)
    }

    /// Bytes the box's elements occupy when packed contiguously — the
    /// destination's size. A sub-byte dtype whose count does not fill whole
    /// bytes is [`SelectError::SubByteMisaligned`].
    pub fn nbytes(&self, dtype: Dtype) -> Result<u64, SelectError> {
        let numel = self.numel();
        dtype
            .nbytes(numel)
            .map_err(|elems| SelectError::SubByteMisaligned {
                dtype,
                bits: dtype.bits(),
                start_elems: 0,
                elems,
            })
    }

    /// The contiguous byte runs of the box in row-major order, as offsets
    /// from the tensor's first byte. Dimensions fully selected from the
    /// innermost outward are merged into each run, so the runs are as few
    /// and as long as the box allows: a full box is one run, a box full on
    /// every dimension but the outermost is one run per selected outer
    /// index. Concatenated, the runs are the sliced tensor's bytes.
    ///
    /// Every run of a sub-byte dtype must start and end on a byte boundary;
    /// the first that does not is [`SelectError::SubByteMisaligned`].
    pub fn runs(&self, dtype: Dtype) -> Result<Vec<Run>, SelectError> {
        let bits = u64::from(dtype.bits());
        let numel = self.numel();
        if numel == 0 {
            return Ok(Vec::new());
        }
        let ndim = self.shape.len();
        // element stride per dimension: the product of the extents inside it
        let mut strides = vec![1u64; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * self.shape[d + 1];
        }
        // the innermost dimension the box does not span whole; everything
        // inside it is contiguous per index of it
        let pivot = (0..ndim)
            .rev()
            .find(|&d| self.ranges[d].start != 0 || self.ranges[d].end != self.shape[d]);
        let Some(pivot) = pivot else {
            return Ok(vec![byte_run(dtype, bits, 0, numel)?]);
        };
        let run_elems = (self.ranges[pivot].end - self.ranges[pivot].start) * strides[pivot];
        let base = self.ranges[pivot].start * strides[pivot];
        let outer = &self.ranges[..pivot];
        let count: u64 = outer.iter().map(|r| r.end - r.start).product();
        let mut runs = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        let mut index: Vec<u64> = outer.iter().map(|r| r.start).collect();
        loop {
            let start = base
                + index
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &stride)| i * stride)
                    .sum::<u64>();
            runs.push(byte_run(dtype, bits, start, run_elems)?);
            // odometer over the outer dimensions, innermost fastest
            let mut d = pivot;
            loop {
                if d == 0 {
                    return Ok(runs);
                }
                d -= 1;
                index[d] += 1;
                if index[d] < outer[d].end {
                    break;
                }
                index[d] = outer[d].start;
            }
        }
    }
}

/// `elems` elements from element `start` as a byte run, or the misalignment
/// that stops a sub-byte dtype from being one.
fn byte_run(dtype: Dtype, bits: u64, start: u64, elems: u64) -> Result<Run, SelectError> {
    let start_bits = start * bits;
    let len_bits = elems * bits;
    if !start_bits.is_multiple_of(8) || !len_bits.is_multiple_of(8) {
        return Err(SelectError::SubByteMisaligned {
            dtype,
            bits: dtype.bits(),
            start_elems: start,
            elems,
        });
    }
    Ok(Run {
        offset: start_bits / 8,
        len: len_bits / 8,
    })
}

/// One contiguous byte run of a selection, relative to the tensor's first
/// byte in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Run {
    /// Bytes from the tensor's first byte.
    pub offset: u64,
    /// Bytes in the run.
    pub len: u64,
}

impl Run {
    /// The run as a file range, given the tensor's absolute start.
    pub fn in_file(&self, tensor_start: u64) -> ReadRange {
        ReadRange {
            offset: tensor_start + self.offset,
            len: self.len,
        }
    }
}

/// When two neighbouring runs become one read.
///
/// Derived from the calibration profile by [`crate::plan::coalesce_policy`]:
/// `max_gap_bytes` is what the mount moves in one read call's fixed cost,
/// so reading through a gap that size costs no more than a second call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoalescePolicy {
    /// Largest gap between two runs that is read through rather than
    /// skipped. Zero disables coalescing: one read per run.
    pub max_gap_bytes: u64,
    /// Largest read one coalesced range may grow to — the staging buffer,
    /// since a device piece must fit one. A single run longer than this is
    /// still one read; the engine splits it.
    pub max_read_bytes: u64,
}

/// One copy out of a read: `len` bytes from `src` within the read's range to
/// `dst` within the destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Placement {
    /// Offset within the read (or piece) the bytes come from.
    pub src: u64,
    /// Offset within the destination they land at.
    pub dst: u64,
    /// Bytes.
    pub len: u64,
}

impl Placement {
    /// One past the last source byte.
    pub fn src_end(&self) -> u64 {
        self.src + self.len
    }

    /// One past the last destination byte.
    pub fn dst_end(&self) -> u64 {
        self.dst + self.len
    }
}

/// One coalesced read: a byte range of the tensor and how its wanted bytes
/// land. The placements are sorted by `src`, do not overlap, and land back
/// to back — `placements[0].dst == 0` and each starts where the previous
/// ended — so a read's wanted bytes form one contiguous piece of the
/// destination beginning at [`Read::dst`]. Bytes of the range no placement
/// names are the gaps read through and dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Read {
    /// Bytes from the tensor's first byte.
    pub offset: u64,
    /// Bytes read, gaps included.
    pub len: u64,
    /// Where the read's first wanted byte lands in the selection's
    /// contiguous destination.
    pub dst: u64,
    /// The wanted bytes, relative to `offset` and to `dst`.
    pub placements: Vec<Placement>,
}

impl Read {
    /// Bytes placed: the sum of the placements' lengths.
    pub fn wanted(&self) -> u64 {
        self.placements.iter().map(|p| p.len).sum()
    }

    /// Whether the read is one run landing whole — no gaps, so it needs no
    /// placements at all.
    pub fn is_contiguous(&self) -> bool {
        self.wanted() == self.len
    }

    /// The read as a file range, given the tensor's absolute start.
    pub fn in_file(&self, tensor_start: u64) -> ReadRange {
        ReadRange {
            offset: tensor_start + self.offset,
            len: self.len,
        }
    }
}

/// Merge neighbouring runs into reads under `policy`.
///
/// `runs` are taken in order and their bytes land in that order, back to
/// back, so runs must be sorted by offset and non-overlapping (what
/// [`Selection::runs`] returns; an out-of-order run is not merged into its
/// predecessor). A run joins the read before it when the gap between them
/// is at most `max_gap_bytes` and the read would stay within
/// `max_read_bytes`; otherwise it starts a new read. Zero-length runs are
/// dropped. Every wanted byte is placed exactly once, reads are sorted and
/// disjoint, and the bytes read exceed the bytes wanted by at most one gap
/// per run.
pub fn coalesce(runs: &[Run], policy: &CoalescePolicy) -> Vec<Read> {
    let mut reads: Vec<Read> = Vec::new();
    let mut dst = 0u64;
    for run in runs {
        if run.len == 0 {
            continue;
        }
        let run_end = run.offset + run.len;
        let joined = reads.last_mut().filter(|read| {
            let read_end = read.offset + read.len;
            policy.max_gap_bytes > 0
                && run.offset >= read_end
                && run.offset - read_end <= policy.max_gap_bytes
                && run_end - read.offset <= policy.max_read_bytes
        });
        match joined {
            Some(read) => {
                read.placements.push(Placement {
                    src: run.offset - read.offset,
                    dst: dst - read.dst,
                    len: run.len,
                });
                read.len = run_end - read.offset;
            }
            None => reads.push(Read {
                offset: run.offset,
                len: run.len,
                dst,
                placements: vec![Placement {
                    src: 0,
                    dst: 0,
                    len: run.len,
                }],
            }),
        }
        dst += run.len;
    }
    reads
}

/// What coalescing did, in the numbers `explain()` prints: runs in, reads
/// out, bytes wanted against bytes read. Accumulated over every selection of
/// a request with [`CoalesceSummary::add`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoalesceSummary {
    /// Runs the selections produced.
    pub runs: usize,
    /// Reads they were coalesced into.
    pub reads: usize,
    /// Bytes the selections want.
    pub wanted_bytes: u64,
    /// Bytes the reads cover, gaps included.
    pub read_bytes: u64,
}

impl CoalesceSummary {
    /// Count `reads`, the output of one [`coalesce`] call.
    pub fn add(&mut self, reads: &[Read]) {
        for read in reads {
            self.runs += read.placements.len();
            self.reads += 1;
            self.wanted_bytes += read.wanted();
            self.read_bytes += read.len;
        }
    }

    /// Bytes read per byte wanted; 1.0 when nothing was wanted.
    pub fn amplification(&self) -> f64 {
        if self.wanted_bytes == 0 {
            1.0
        } else {
            self.read_bytes as f64 / self.wanted_bytes as f64
        }
    }

    /// Whether any read carries more than one run — the only case worth a
    /// line in `explain()`.
    pub fn coalesced_anything(&self) -> bool {
        self.reads < self.runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn sel(shape: &[u64], ranges: &[Range<u64>]) -> Selection {
        Selection::new(shape, ranges.to_vec()).unwrap_or_else(|e| panic!("{e}"))
    }

    fn runs_of(s: &Selection, dtype: Dtype) -> Vec<Run> {
        s.runs(dtype).unwrap_or_else(|e| panic!("{e}"))
    }

    fn run(offset: u64, len: u64) -> Run {
        Run { offset, len }
    }

    /// Bit membership of the box: for every element of the tensor in
    /// row-major order, whether it lies in the box — the reference every
    /// run-producing rule is checked against.
    fn member_bits(s: &Selection, bits: u64) -> Vec<bool> {
        let shape = s.shape();
        let numel: u64 = shape.iter().product();
        let mut out = Vec::with_capacity((numel * bits) as usize);
        for flat in 0..numel {
            let mut rest = flat;
            let mut inside = true;
            for d in (0..shape.len()).rev() {
                let i = rest % shape[d];
                rest /= shape[d];
                inside &= s.ranges()[d].contains(&i);
            }
            out.extend(std::iter::repeat_n(inside, bits as usize));
        }
        out
    }

    fn bits_of_runs(runs: &[Run], total_bits: u64) -> Vec<bool> {
        let mut out = vec![false; total_bits as usize];
        for r in runs {
            for bit in r.offset * 8..(r.offset + r.len) * 8 {
                assert!(!out[bit as usize], "bit {bit} covered twice");
                out[bit as usize] = true;
            }
        }
        out
    }

    /// The bytes `torch.narrow`-style slicing yields: the box's elements in
    /// row-major order, each as its bytes of a tensor whose byte `i` holds
    /// the value `i`.
    fn reference_bytes(s: &Selection, dtype: Dtype) -> Vec<u8> {
        let bytes = dtype.bits() as u64 / 8;
        let shape = s.shape();
        let numel: u64 = shape.iter().product();
        let mut out = Vec::new();
        for flat in 0..numel {
            let mut rest = flat;
            let mut inside = true;
            for d in (0..shape.len()).rev() {
                let i = rest % shape[d];
                rest /= shape[d];
                inside &= s.ranges()[d].contains(&i);
            }
            if inside {
                out.extend((flat * bytes..(flat + 1) * bytes).map(|b| b as u8));
            }
        }
        out
    }

    #[test]
    fn constructors_validate_against_the_shape() {
        assert!(matches!(
            Selection::new(&[4, 4], vec![0..4, 0..4, 0..4]),
            Err(SelectError::NdimMismatch { ndim: 2, ranges: 3 })
        ));
        assert!(matches!(
            Selection::new(&[4, 4], vec![0..4, 2..5]),
            Err(SelectError::OutOfBounds {
                dim: 1,
                start: 2,
                end: 5,
                extent: 4
            })
        ));
        assert!(matches!(
            Selection::new(&[4], Vec::from([Range { start: 3, end: 2 }])),
            Err(SelectError::OutOfBounds { dim: 0, .. })
        ));
        assert!(matches!(
            Selection::new(&[u64::MAX, 2], vec![0..1, 0..1]),
            Err(SelectError::TooLarge { .. })
        ));
        assert!(matches!(
            Selection::shard(&[8, 8], 2, 0, 2),
            Err(SelectError::DimOutOfRange { dim: 2, ndim: 2 })
        ));
        assert!(matches!(
            Selection::shard(&[8, 8], 1, 2, 2),
            Err(SelectError::BadRank { rank: 2, world: 2 })
        ));
        assert!(matches!(
            Selection::shard(&[8, 8], 1, 0, 0),
            Err(SelectError::BadRank { rank: 0, world: 0 })
        ));
        assert!(matches!(
            Selection::shard(&[8, 6], 1, 0, 4),
            Err(SelectError::DoesNotDivide {
                dim: 1,
                extent: 6,
                world: 4
            })
        ));
        let shard = Selection::shard(&[8, 8], 1, 3, 4).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(shard.ranges(), &[0..8, 6..8]);
        assert_eq!(shard.box_shape(), vec![8, 2]);
        assert_eq!(shard.numel(), 16);
        assert!(!shard.is_full());
        let full = Selection::full(&[8, 8]).unwrap_or_else(|e| panic!("{e}"));
        assert!(full.is_full());
        // a scalar
        let scalar = Selection::full(&[]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(scalar.numel(), 1);
        assert_eq!(runs_of(&scalar, Dtype::F32), vec![run(0, 4)]);
    }

    #[test]
    fn runs_merge_full_inner_dimensions() {
        // the motivating case: [out, in] bf16, row-parallel over 8 ranks
        let s = Selection::shard(&[7168, 18432], 1, 2, 8).unwrap_or_else(|e| panic!("{e}"));
        let runs = runs_of(&s, Dtype::BF16);
        assert_eq!(runs.len(), 7168);
        assert!(runs.iter().all(|r| r.len == 18432 / 8 * 2));
        assert_eq!(runs[0].offset, 2 * 2304 * 2);
        assert_eq!(runs[1].offset - runs[0].offset, 18432 * 2);
        // column-parallel: one contiguous range
        let s = Selection::shard(&[7168, 18432], 0, 5, 8).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            runs_of(&s, Dtype::BF16),
            vec![run(5 * 896 * 18432 * 2, 896 * 18432 * 2)]
        );
        // a full box is one run whatever the rank
        let s = Selection::full(&[3, 4, 5]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(runs_of(&s, Dtype::U8), vec![run(0, 60)]);
        // a partial middle dimension: one run per outer index, spanning the inner ones
        let s = sel(&[3, 4, 5], &[0..3, 1..3, 0..5]);
        assert_eq!(
            runs_of(&s, Dtype::U8),
            vec![run(5, 10), run(25, 10), run(45, 10)]
        );
        // partial on two dimensions: runs per (outer, middle) index
        let s = sel(&[2, 3, 4], &[0..2, 1..3, 2..4]);
        assert_eq!(
            runs_of(&s, Dtype::I16),
            vec![run(12, 4), run(20, 4), run(36, 4), run(44, 4)]
        );
        // an empty box has no runs
        assert!(runs_of(&sel(&[4, 4], &[1..1, 0..4]), Dtype::F32).is_empty());
        assert!(runs_of(&sel(&[0, 4], &[0..0, 0..4]), Dtype::F32).is_empty());
    }

    #[test]
    fn sub_byte_runs_must_be_byte_aligned() {
        // F4: two elements per byte. An even inner extent at an even start is fine
        let s = sel(&[4, 8], &[0..4, 2..6]);
        assert_eq!(
            runs_of(&s, Dtype::F4),
            vec![run(1, 2), run(5, 2), run(9, 2), run(13, 2)]
        );
        assert_eq!(s.nbytes(Dtype::F4).ok(), Some(8));
        // an odd extent is refused, naming the run
        let odd = sel(&[4, 8], &[0..4, 2..5]);
        assert!(matches!(
            odd.runs(Dtype::F4),
            Err(SelectError::SubByteMisaligned {
                dtype: Dtype::F4,
                bits: 4,
                start_elems: 2,
                elems: 3
            })
        ));
        // twelve nibbles are six whole bytes: the packed size is fine, only
        // the runs are not
        assert_eq!(odd.nbytes(Dtype::F4).ok(), Some(6));
        assert!(matches!(
            sel(&[3, 3], &[0..1, 0..3]).nbytes(Dtype::F4),
            Err(SelectError::SubByteMisaligned { elems: 3, .. })
        ));
        // an odd start is refused even with an even extent
        assert!(matches!(
            sel(&[4, 8], &[0..4, 1..5]).runs(Dtype::F4),
            Err(SelectError::SubByteMisaligned { start_elems: 1, .. })
        ));
        // an odd row stride misaligns every second row
        assert!(matches!(
            sel(&[4, 3], &[0..4, 0..2]).runs(Dtype::F4),
            Err(SelectError::SubByteMisaligned { start_elems: 3, .. })
        ));
        // F6: four elements fill three bytes
        assert_eq!(
            runs_of(&sel(&[2, 8], &[0..2, 4..8]), Dtype::F6_E2M3),
            vec![run(3, 3), run(9, 3)]
        );
        // whole tensors of a sub-byte dtype are a single run, as always
        assert_eq!(
            runs_of(
                &Selection::full(&[3, 4]).unwrap_or_else(|e| panic!("{e}")),
                Dtype::F4
            ),
            vec![run(0, 6)]
        );
    }

    fn policy(max_gap_bytes: u64, max_read_bytes: u64) -> CoalescePolicy {
        CoalescePolicy {
            max_gap_bytes,
            max_read_bytes,
        }
    }

    #[test]
    fn coalesce_merges_across_small_gaps_up_to_the_cap() {
        let runs = [run(0, 4), run(8, 4), run(16, 4), run(40, 4), run(44, 4)];
        let reads = coalesce(&runs, &policy(4, 64));
        assert_eq!(
            reads,
            vec![
                Read {
                    offset: 0,
                    len: 20,
                    dst: 0,
                    placements: vec![
                        Placement {
                            src: 0,
                            dst: 0,
                            len: 4
                        },
                        Placement {
                            src: 8,
                            dst: 4,
                            len: 4
                        },
                        Placement {
                            src: 16,
                            dst: 8,
                            len: 4
                        },
                    ],
                },
                Read {
                    offset: 40,
                    len: 8,
                    dst: 12,
                    placements: vec![
                        Placement {
                            src: 0,
                            dst: 0,
                            len: 4
                        },
                        Placement {
                            src: 4,
                            dst: 4,
                            len: 4
                        },
                    ],
                },
            ]
        );
        assert!(!reads[0].is_contiguous());
        assert!(reads[1].is_contiguous());
        assert_eq!(
            reads[0].in_file(100),
            ReadRange {
                offset: 100,
                len: 20
            }
        );
        // the read cap splits what the gap rule would merge
        let capped = coalesce(&runs, &policy(4, 12));
        assert_eq!(
            capped.iter().map(|r| (r.offset, r.len)).collect::<Vec<_>>(),
            vec![(0, 12), (16, 4), (40, 8)]
        );
        // a run longer than the cap is still one read
        assert_eq!(
            coalesce(&[run(0, 100)], &policy(4, 12)),
            vec![Read {
                offset: 0,
                len: 100,
                dst: 0,
                placements: vec![Placement {
                    src: 0,
                    dst: 0,
                    len: 100
                }],
            }]
        );
        // zero gap disables coalescing, even for adjacent runs
        let off = coalesce(&runs, &policy(0, 64));
        assert_eq!(off.len(), runs.len());
        assert!(off.iter().all(Read::is_contiguous));
        // an out-of-order run is never merged into its predecessor
        let back = coalesce(&[run(8, 4), run(0, 4)], &policy(64, 64));
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].dst, 4);
        // zero-length runs vanish
        assert!(coalesce(&[run(3, 0)], &policy(64, 64)).is_empty());
        let mut summary = CoalesceSummary::default();
        summary.add(&reads);
        assert_eq!(
            summary,
            CoalesceSummary {
                runs: 5,
                reads: 2,
                wanted_bytes: 20,
                read_bytes: 28
            }
        );
        assert!((summary.amplification() - 1.4).abs() < 1e-12);
        assert!(summary.coalesced_anything());
        assert_eq!(CoalesceSummary::default().amplification(), 1.0);
    }

    /// Shapes small enough to enumerate, with every dtype.
    fn arb_shape() -> impl Strategy<Value = Vec<u64>> {
        prop::collection::vec(1u64..6, 0..4)
    }

    fn arb_selection() -> impl Strategy<Value = Selection> {
        arb_shape().prop_flat_map(|shape| {
            let ranges: Vec<_> = shape
                .iter()
                .map(|&extent| {
                    (0..=extent).prop_flat_map(move |start| (Just(start), start..=extent))
                })
                .collect();
            ranges.prop_map(move |bounds| {
                Selection::new(&shape, bounds.into_iter().map(|(s, e)| s..e).collect())
                    .unwrap_or_else(|e| panic!("{e}"))
            })
        })
    }

    fn arb_byte_dtype() -> impl Strategy<Value = Dtype> {
        prop::sample::select(
            Dtype::ALL
                .into_iter()
                .filter(|d| d.bits() >= 8)
                .collect::<Vec<_>>(),
        )
    }

    /// Sorted, non-overlapping runs with positive lengths.
    fn arb_runs() -> impl Strategy<Value = Vec<Run>> {
        prop::collection::vec((0u64..16, 1u64..24), 0..12).prop_map(|steps| {
            let mut offset = 0;
            steps
                .into_iter()
                .map(|(gap, len)| {
                    offset += gap;
                    let r = run(offset, len);
                    offset += len;
                    r
                })
                .collect()
        })
    }

    proptest! {
        #[test]
        fn runs_tile_the_box_exactly(s in arb_selection(), dtype in prop::sample::select(Dtype::ALL.to_vec())) {
            let bits = u64::from(dtype.bits());
            let runs = match s.runs(dtype) {
                Ok(runs) => runs,
                Err(SelectError::SubByteMisaligned { .. }) => {
                    prop_assume!(bits >= 8, "byte dtypes never misalign");
                    return Ok(());
                }
                Err(e) => panic!("{e}"),
            };
            let numel: u64 = s.shape().iter().product();
            // sorted, non-overlapping, and exactly the box's bits
            for pair in runs.windows(2) {
                prop_assert!(pair[0].offset + pair[0].len < pair[1].offset, "{:?}", pair);
            }
            prop_assert!(runs.iter().all(|r| r.len > 0));
            prop_assert_eq!(bits_of_runs(&runs, numel * bits), member_bits(&s, bits));
            prop_assert_eq!(runs.iter().map(|r| r.len).sum::<u64>() * 8, s.numel() * bits);
            if s.is_full() && numel > 0 {
                prop_assert_eq!(runs.len(), 1);
            }
            // the box's element count in whole bytes agrees with nbytes
            prop_assert_eq!(s.nbytes(dtype).ok(), Some(runs.iter().map(|r| r.len).sum::<u64>()));
        }

        #[test]
        fn shards_are_what_narrow_gives_and_partition_the_tensor(
            shape in prop::collection::vec(1u64..7, 1..4),
            dim_seed in 0usize..4,
            world_seed in 1u64..4,
            dtype in arb_byte_dtype(),
        ) {
            let dim = dim_seed % shape.len();
            // pick a world that divides the dimension
            let world = (1..=shape[dim]).filter(|w| shape[dim] % w == 0).nth(world_seed as usize % 3).unwrap_or(1);
            let bytes = u64::from(dtype.bits()) / 8;
            let mut seen = vec![false; (shape.iter().product::<u64>() * bytes) as usize];
            for rank in 0..world {
                let s = Selection::shard(&shape, dim, rank, world).unwrap_or_else(|e| panic!("{e}"));
                let runs = runs_of(&s, dtype);
                // concatenated runs are the narrow'd tensor's bytes
                let got: Vec<u8> = runs.iter().flat_map(|r| (r.offset..r.offset + r.len).map(|b| b as u8)).collect();
                prop_assert_eq!(got, reference_bytes(&s, dtype));
                // the outermost dim shards as one run; every shard is one run per outer index at most
                if dim == 0 {
                    prop_assert_eq!(runs.len(), 1);
                }
                let outer: u64 = shape[..dim].iter().product();
                prop_assert!(runs.len() as u64 <= outer.max(1));
                for r in &runs {
                    for b in r.offset..r.offset + r.len {
                        prop_assert!(!seen[b as usize]);
                        seen[b as usize] = true;
                    }
                }
            }
            prop_assert!(seen.iter().all(|&b| b), "shards do not cover the tensor");
        }

        #[test]
        fn coalesced_reads_place_every_byte_once_within_bounds(
            runs in arb_runs(),
            max_gap in 0u64..32,
            max_read in 1u64..128,
        ) {
            let policy = policy(max_gap, max_read);
            let reads = coalesce(&runs, &policy);
            let wanted: u64 = runs.iter().map(|r| r.len).sum();
            // every wanted byte lands exactly once, in run order
            let mut landed = vec![None; wanted as usize];
            for read in &reads {
                prop_assert!(!read.placements.is_empty());
                let mut expect_dst = 0;
                for p in &read.placements {
                    prop_assert_eq!(p.dst, expect_dst, "placements land back to back");
                    expect_dst = p.dst_end();
                    prop_assert!(p.src_end() <= read.len);
                    for i in 0..p.len {
                        let slot = &mut landed[(read.dst + p.dst + i) as usize];
                        prop_assert!(slot.is_none(), "byte placed twice");
                        *slot = Some(read.offset + p.src + i);
                    }
                }
            }
            let expected: Vec<Option<u64>> = runs.iter().flat_map(|r| (r.offset..r.offset + r.len).map(Some)).collect();
            prop_assert_eq!(landed, expected);
            // reads are sorted and disjoint, and within the cap unless a single run exceeds it
            for pair in reads.windows(2) {
                prop_assert!(pair[0].offset + pair[0].len <= pair[1].offset);
            }
            for read in &reads {
                prop_assert!(read.len <= max_read || read.placements.len() == 1, "{read:?}");
            }
            // amplification is bounded by one gap per run
            let read_bytes: u64 = reads.iter().map(|r| r.len).sum();
            if let Some(min_run) = runs.iter().map(|r| r.len).min() {
                let bound = 1.0 + max_gap as f64 / min_run as f64;
                prop_assert!(read_bytes as f64 <= wanted as f64 * bound + 1e-9, "{read_bytes} / {wanted} > {bound}");
            }
            // no coalescing: one read per run
            if max_gap == 0 {
                prop_assert_eq!(reads.len(), runs.len());
                prop_assert!(reads.iter().all(Read::is_contiguous));
            }
            let mut summary = CoalesceSummary::default();
            summary.add(&reads);
            prop_assert_eq!(summary.runs, runs.len());
            prop_assert_eq!(summary.reads, reads.len());
            prop_assert_eq!(summary.wanted_bytes, wanted);
            prop_assert_eq!(summary.read_bytes, read_bytes);
        }
    }
}
