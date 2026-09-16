//! Byte movement into and out of caller-owned memory, through the engines.
//!
//! [`read_job`] builds a [`ReadJob`] and runs `fst_core::read::execute`;
//! [`write_object`] builds a [`Payload`] and runs
//! `fst_core::write::write_object`. Both take addresses of memory the Python
//! layer owns and keeps alive, and both run with the GIL released.
//!
//! # Trust boundary
//!
//! These are the only functions in the package that dereference addresses
//! they are handed, and the `unsafe` blocks in [`host_slice`] and
//! [`host_bytes`] are the crate's only ones. Their callers are `causalab.io.fastersafetensors._files` and
//! `causalab.io.fastersafetensors._serialize`, which allocate every destination and
//! source as a **contiguous** tensor with torch, pass `tensor.data_ptr()`
//! with that tensor's exact byte length and device, and keep the tensor
//! alive across the call. Torch's stream is synchronized by the caller
//! before a device pointer comes down: the runtime's copies run on a
//! non-blocking stream that does not order against torch's default stream.
//! Checked here: a length fits the address space, no range wraps, no
//! non-empty destination is null or overlaps another, a device destination
//! names its device. Everything about the file — the range lies inside it,
//! it exists — is checked by the engine and surfaced as a typed error.
//!
//! # Placements
//!
//! A transfer may carry `placements`: the range is a coalesced read
//! (`fst_core::select::coalesce`, reached through [`crate::select`]) and
//! only the pieces the placements name land, back to back from the
//! destination's origin, the gaps read and dropped. They go onto the
//! engine's [`Transfer`] as they are; the engine checks they are a gather
//! of the range (sorted, disjoint, packed) before any byte moves. What is
//! checked here is the landing: a host destination is exactly the
//! placements' total, a device buffer holds it from its offset.
//!
//! The CUDA runtime is loaded once per device and kept for the life of the
//! process ([`runtime`]): opening `libcudart` and creating a stream is not
//! free, and a load is many calls.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use fst_core::plan::{ReadPlan, Staging, Transport};
use fst_core::read::{self, Dest, FileJob, Placement, ReadJob, Transfer};
use fst_core::storage::ReadRange;
use fst_core::storage::mmap::MmapStorage;
use fst_core::storage::posix::PosixStorage;
use fst_core::write::{self, Part, Payload, StagingRing, WriteOptions};
use fst_cuda::{CudaRuntime, DeviceCopier, DevicePtr};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::{IntoPyErr, IoError};
use crate::header::{SpecRow, tensor_specs};

/// `(offset within the range, offset within the destination, nbytes)`: one
/// piece of a coalesced read and where it lands — a
/// `fst_core::select::Placement` as `select_reads` returned it.
type PyPlacement = (u64, u64, u64);
/// `(offset in the file, nbytes, destination address, destination offset,
/// placements)`. The destination offset is `None` for a host destination
/// (the address is where the bytes land) and `Some(offset)` for a device
/// buffer (the address is the buffer's base; the bytes land `offset` into
/// it). `placements` `None` (or empty) lands the whole range contiguously
/// there; otherwise only the pieces land, packed from that origin, and the
/// destination is their total.
type PyTransfer = (u64, u64, usize, Option<u64>, Option<Vec<PyPlacement>>);
/// One file and its transfers.
type PyFileJob = (PathBuf, Vec<PyTransfer>);

/// Loaded runtimes by device ordinal. A `Vec` because `Vec::new` is `const`;
/// a process talks to a handful of devices.
static RUNTIMES: Mutex<Vec<(u32, Arc<CudaRuntime>)>> = Mutex::new(Vec::new());

/// The runtime for `device`, loaded on first use and shared thereafter.
fn runtime(device: u32) -> Result<Arc<CudaRuntime>, IoError> {
    let mut runtimes = RUNTIMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, rt)) = runtimes.iter().find(|(d, _)| *d == device) {
        return Ok(Arc::clone(rt));
    }
    let rt = Arc::new(CudaRuntime::load(device)?);
    runtimes.push((device, Arc::clone(&rt)));
    Ok(rt)
}

/// A checked `(address, nbytes)` as a mutable byte slice.
///
/// # Safety contract (see the module docs)
///
/// The caller established that the range is non-null, does not wrap, and
/// overlaps no other destination of the job; the Python layer guarantees it
/// is a live, writable, `nbytes`-long, contiguous host allocation for the
/// duration of the call.
fn host_slice<'a>(address: usize, nbytes: usize) -> &'a mut [u8] {
    // SAFETY: as documented above; `nbytes <= isize::MAX` was checked.
    unsafe { std::slice::from_raw_parts_mut(address as *mut u8, nbytes) }
}

/// The read-only counterpart of [`host_slice`], for parts to write: the
/// Python layer keeps a live, unmodified, `nbytes`-long, contiguous host
/// tensor at `address` for the duration of the call.
fn host_bytes<'a>(address: usize, nbytes: usize) -> &'a [u8] {
    // SAFETY: as documented above; `nbytes <= isize::MAX` was checked.
    unsafe { std::slice::from_raw_parts(address as *const u8, nbytes) }
}

fn checked_len(nbytes: u64, what: &str) -> Result<usize, IoError> {
    usize::try_from(nbytes)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            IoError::Invalid(format!(
                "{what} of {nbytes} bytes exceeds the address space"
            ))
        })
}

/// Refuse a null or wrapping `[address, address + nbytes)`; return its end.
fn checked_span(address: usize, nbytes: u64, what: &str) -> Result<(usize, usize), IoError> {
    let len = checked_len(nbytes, what)?;
    if address == 0 {
        return Err(IoError::Invalid(format!("null {what} of {nbytes} bytes")));
    }
    let end = address.checked_add(len).ok_or_else(|| {
        IoError::Invalid(format!(
            "{what} {address:#x} + {nbytes} wraps the address space"
        ))
    })?;
    Ok((address, end))
}

/// Refuse two spans that share a byte (two workers would race on it).
fn check_disjoint(mut spans: Vec<(usize, usize)>) -> Result<(), IoError> {
    spans.sort_unstable();
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(IoError::Invalid(format!(
                "destinations {:#x}..{:#x} and {:#x}..{:#x} overlap",
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            )));
        }
    }
    Ok(())
}

fn transport(name: &str) -> Result<Transport, IoError> {
    match name {
        "pread" => Ok(Transport::Pread),
        "mmap" => Ok(Transport::Mmap),
        "cufile" => Ok(Transport::CuFile),
        other => Err(IoError::Invalid(format!(
            "transport must be 'pread', 'mmap' or 'cufile', got {other:?}"
        ))),
    }
}

/// The engine's placements for a row, and the bytes they land: the range
/// when there are none.
fn placements(raw: Option<&[PyPlacement]>, nbytes: u64) -> Result<(Vec<Placement>, u64), IoError> {
    let placements: Vec<Placement> = raw
        .unwrap_or_default()
        .iter()
        .map(|&(src, dst, len)| Placement { src, dst, len })
        .collect();
    if placements.is_empty() {
        return Ok((placements, nbytes));
    }
    let landed = placements
        .iter()
        .try_fold(0u64, |acc, p| acc.checked_add(p.len))
        .ok_or_else(|| IoError::Invalid("placements land more bytes than fit a u64".to_owned()))?;
    Ok((placements, landed))
}

/// Build the engine's job from the Python tuples. Empty transfers are
/// dropped (nothing to land, whatever their address); a file with none left
/// is still opened so a missing file is reported.
fn build_job<'a>(files: &[PyFileJob], device: Option<u32>) -> Result<ReadJob<'a>, IoError> {
    let mut spans = Vec::new();
    let mut job = ReadJob::default();
    for (path, transfers) in files {
        let mut out = Vec::with_capacity(transfers.len());
        for (offset, nbytes, address, dst_offset, raw) in transfers {
            let (offset, nbytes, address) = (*offset, *nbytes, *address);
            if nbytes == 0 {
                continue;
            }
            offset
                .checked_add(nbytes)
                .ok_or_else(|| IoError::Invalid(format!("offset {offset} + {nbytes} overflows")))?;
            let range = ReadRange {
                offset,
                len: nbytes,
            };
            let (placements, landed) = placements(raw.as_deref(), nbytes)?;
            let dest = match *dst_offset {
                None => {
                    let (start, end) = checked_span(address, landed, "destination")?;
                    spans.push((start, end));
                    Dest::Host(host_slice(start, end - start))
                }
                Some(dst_offset) => {
                    let device = device.ok_or_else(|| {
                        IoError::Invalid(
                            "a device destination was given without a device".to_owned(),
                        )
                    })?;
                    let total = dst_offset.checked_add(landed).ok_or_else(|| {
                        IoError::Invalid(format!(
                            "destination offset {dst_offset} + {landed} overflows"
                        ))
                    })?;
                    let skip = checked_len(dst_offset, "destination offset")?;
                    let base = address.checked_add(skip).ok_or_else(|| {
                        IoError::Invalid(format!(
                            "destination {address:#x} + {dst_offset} wraps the address space"
                        ))
                    })?;
                    spans.push(checked_span(base, landed, "device destination")?);
                    // the buffer is described as exactly the bytes this
                    // transfer lands; the caller guarantees it holds them
                    Dest::Device {
                        ptr: DevicePtr {
                            address,
                            nbytes: total,
                            device,
                        },
                        offset: dst_offset,
                    }
                }
            };
            out.push(Transfer {
                range,
                dest,
                placements,
            });
        }
        job.files.push(FileJob {
            path: path.clone(),
            transfers: out,
        });
    }
    check_disjoint(spans)?;
    Ok(job)
}

fn run_read(
    files: &[PyFileJob],
    plan: &ReadPlan,
    device: Option<u32>,
) -> Result<read::ReadReport, IoError> {
    let job = build_job(files, device)?;
    let has_device_dest = job
        .files
        .iter()
        .flat_map(|f| &f.transfers)
        .any(|t| matches!(t.dest, Dest::Device { .. }));
    let copier = match device {
        Some(d) if has_device_dest => Some(runtime(d)?),
        _ => None,
    };
    let copier: Option<&dyn DeviceCopier> = copier.as_deref().map(|c| c as &dyn DeviceCopier);
    // GPUDirect Storage is opened only when the plan asks for it; a library
    // that will not load is not an error here — the engine records the
    // fall-back to pread in the report and the load still completes
    let direct = match plan.transport {
        Transport::CuFile if has_device_dest => fst_cuda::cufile::CuFile::load().ok(),
        _ => None,
    };
    let direct: Option<&dyn fst_cuda::DirectStorage> =
        direct.as_ref().map(|d| d as &dyn fst_cuda::DirectStorage);
    let report = match plan.transport {
        Transport::Mmap => read::execute(plan, &MmapStorage, copier, direct, job),
        Transport::Pread | Transport::CuFile => {
            read::execute(plan, &PosixStorage, copier, direct, job)
        }
    }?;
    Ok(report)
}

/// Run one read job as the plan says, with the GIL released.
///
/// `files` is `[(path, [(offset, nbytes, address, dst_offset,
/// placements)])]`; see [`PyTransfer`] for the destination encoding. The
/// plan fields are those `plan_read` returned (`transport` by name,
/// `staging` as `(count, bytes)` or `None`). `device` is the CUDA ordinal
/// every device destination is on; it is required when there is one and the
/// runtime for it is loaded on first use. Returns `{bytes_read, pieces,
/// files_opened, gap_bytes, placed_pieces, scatter_copies, elapsed}`
/// (`elapsed` in seconds; `gap_bytes` the bytes read through and dropped
/// between placements; `scatter_copies` the per-placement device copies
/// issued for pieces whose placements were not one regular 2-D pattern)
/// once every byte has landed — device copies included, the engine
/// synchronizes before returning.
///
/// Trusts the addresses: see the module docs for what the caller guarantees.
#[pyfunction]
#[pyo3(signature = (files, files_in_flight, split_bytes, readers_per_file, transport, staging, device=None))]
#[allow(clippy::too_many_arguments)]
pub fn read_job<'py>(
    py: Python<'py>,
    files: Vec<PyFileJob>,
    files_in_flight: usize,
    split_bytes: u64,
    readers_per_file: usize,
    transport: &str,
    staging: Option<(usize, u64)>,
    device: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = ReadPlan {
        files_in_flight,
        split_bytes,
        readers_per_file,
        transport: self::transport(transport).map_err(|e| e.into_py_err(py))?,
        staging: match staging {
            None => Staging::None,
            Some((count, bytes)) => Staging::Pinned { count, bytes },
        },
        reasons: Vec::new(),
        // the plan was decided in `planning::plan_read` and re-encoded by
        // the Python layer; the provenance travels with the explanation there
        profile_source: String::new(),
    };
    let report = py
        .detach(|| run_read(&files, &plan, device))
        .map_err(|e| e.into_py_err(py))?;
    let dict = PyDict::new(py);
    dict.set_item("bytes_read", report.bytes_read)?;
    dict.set_item("pieces", report.pieces)?;
    dict.set_item("files_opened", report.files_opened)?;
    dict.set_item("gap_bytes", report.gap_bytes)?;
    dict.set_item("placed_pieces", report.placed_pieces)?;
    dict.set_item("scatter_copies", report.scatter_copies)?;
    dict.set_item("elapsed", report.elapsed.as_secs_f64())?;
    Ok(dict)
}

/// `(address, nbytes, on_device)` of one part to write.
type PyPart = (usize, u64, bool);

fn build_parts<'a>(parts: &[PyPart], device: Option<u32>) -> Result<Vec<Part<'a>>, IoError> {
    parts
        .iter()
        .map(|&(address, nbytes, on_device)| {
            if nbytes == 0 {
                return Ok(Part::Host(&[]));
            }
            let (start, end) = checked_span(address, nbytes, "source")?;
            if on_device {
                let device = device.ok_or_else(|| {
                    IoError::Invalid("a device part was given without a device".to_owned())
                })?;
                Ok(Part::Device {
                    ptr: DevicePtr {
                        address,
                        nbytes,
                        device,
                    },
                    nbytes,
                })
            } else {
                Ok(Part::Host(host_bytes(start, end - start)))
            }
        })
        .collect()
}

fn run_write(
    path: &Path,
    specs: &[SpecRow],
    parts: &[PyPart],
    metadata: Option<&[(String, String)]>,
    options: &WriteOptions,
    device: Option<u32>,
) -> Result<write::WriteReport, IoError> {
    let specs = tensor_specs(specs).map_err(write::WriteError::from)?;
    let parts = build_parts(parts, device)?;
    let payload = Payload::new(&specs, &parts, metadata)?;
    let copier = match (payload.has_device_parts(), device) {
        (true, Some(d)) => Some(runtime(d)?),
        _ => None,
    };
    let copier: Option<&dyn DeviceCopier> = copier.as_deref().map(|c| c as &dyn DeviceCopier);
    Ok(write::write_object(
        &PosixStorage,
        path,
        &payload,
        options,
        copier,
    )?)
}

/// Write the object described by `specs` — `(name, dtype, shape, nbytes)`
/// each — and `parts` — `(address, nbytes, on_device)`, `parts[i]` being
/// `specs[i]`'s bytes — to `path`, with the GIL released. The header is
/// built here and the parts follow it in data-section order, each written
/// straight from its memory; device parts are drained through the engine's
/// pinned staging ring on the runtime for `device`. `durable` fsyncs the
/// file and its directory; `atomic` writes a sibling temp file and renames
/// it into place. Returns the bytes written.
///
/// Trusts the addresses: see the module docs for what the caller guarantees.
#[pyfunction]
#[pyo3(signature = (path, specs, parts, metadata=None, durable=false, atomic=true, device=None))]
#[allow(clippy::too_many_arguments)]
pub fn write_object(
    py: Python<'_>,
    path: PathBuf,
    specs: Vec<SpecRow>,
    parts: Vec<PyPart>,
    metadata: Option<Vec<(String, String)>>,
    durable: bool,
    atomic: bool,
    device: Option<u32>,
) -> PyResult<u64> {
    let options = WriteOptions {
        durable,
        atomic,
        staging: StagingRing::default(),
    };
    let report = py
        .detach(|| run_write(&path, &specs, &parts, metadata.as_deref(), &options, device))
        .map_err(|e| e.into_py_err(py))?;
    Ok(report.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_plan() -> ReadPlan {
        ReadPlan {
            files_in_flight: 2,
            split_bytes: 1024,
            readers_per_file: 2,
            transport: Transport::Pread,
            staging: Staging::None,
            reasons: Vec::new(),
            profile_source: String::new(),
        }
    }

    #[test]
    fn overlapping_or_null_destinations_are_refused() {
        let buf = [0u8; 16];
        let base = buf.as_ptr() as usize;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obj");
        assert!(
            build_job(
                &[(
                    path.clone(),
                    vec![(0, 8, base, None, None), (8, 8, base + 8, None, None)]
                )],
                None
            )
            .is_ok()
        );
        assert!(matches!(
            build_job(
                &[(
                    path.clone(),
                    vec![(0, 8, base, None, None), (8, 8, base + 4, None, None)]
                )],
                None
            ),
            Err(IoError::Invalid(_))
        ));
        assert!(matches!(
            build_job(&[(path.clone(), vec![(0, 8, 0, None, None)])], None),
            Err(IoError::Invalid(_))
        ));
        // a device destination needs a device
        assert!(matches!(
            build_job(&[(path.clone(), vec![(0, 8, base, Some(0), None)])], None),
            Err(IoError::Invalid(_))
        ));
        // a placed transfer's landing is its placements' total: 8 bytes here,
        // so it fits beside a transfer landing at base + 8 ...
        let placed = Some(vec![(0, 0, 4), (12, 4, 4)]);
        assert!(
            build_job(
                &[(
                    path.clone(),
                    vec![
                        (0, 16, base, None, placed.clone()),
                        (16, 8, base + 8, None, None)
                    ]
                )],
                None
            )
            .is_ok()
        );
        // ... and not beside one landing at base + 4
        assert!(matches!(
            build_job(
                &[(
                    path.clone(),
                    vec![(0, 16, base, None, placed), (16, 8, base + 4, None, None)]
                )],
                None
            ),
            Err(IoError::Invalid(_))
        ));
        // empty transfers are dropped, whatever their pointer
        let job = build_job(&[(path, vec![(0, 0, 0, None, None)])], None).unwrap();
        assert_eq!(job.files.len(), 1);
        assert!(job.files[0].transfers.is_empty());
    }

    #[test]
    fn reads_land_once_and_errors_name_the_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obj");
        let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        std::fs::write(&path, &data).unwrap();
        let mut out = vec![0u8; 4096];
        let base = out.as_mut_ptr() as usize;
        let transfers: Vec<PyTransfer> = (0..4)
            .map(|i| (i * 1024, 1024, base + (i as usize) * 1024, None, None))
            .collect();
        let report = run_read(&[(path.clone(), transfers)], &host_plan(), None).unwrap();
        assert_eq!(report.bytes_read, 4096);
        assert_eq!(report.pieces, 4);
        assert_eq!(out, data);
        match run_read(
            &[(path.clone(), vec![(4000, 256, base, None, None)])],
            &host_plan(),
            None,
        ) {
            Err(IoError::Read(err)) => {
                let text = err.to_string();
                assert!(text.contains("4000..4256"), "{text}");
            }
            other => panic!("expected an out-of-range read error, got {other:?}"),
        }
        assert!(matches!(
            run_read(&[(dir.path().join("missing"), vec![])], &host_plan(), None),
            Err(IoError::Read(fst_core::ReadError::Open { .. }))
        ));
    }

    #[test]
    fn placed_host_transfer_lands_only_its_pieces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obj");
        let data: Vec<u8> = (0..=255u8).collect();
        std::fs::write(&path, &data).unwrap();
        // a 4x8 byte matrix, columns 2..4 wanted: four runs of 2 in a span of 26
        let mut out = vec![0xffu8; 8];
        let base = out.as_mut_ptr() as usize;
        let placements: Vec<PyPlacement> = (0..4).map(|r| (r * 8, r * 2, 2)).collect();
        let files = vec![(
            path.clone(),
            vec![(2, 26, base, None, Some(placements.clone()))],
        )];
        let report = run_read(&files, &host_plan(), None).unwrap();
        assert_eq!(report.bytes_read, 26); // the span, gaps included
        assert_eq!(report.gap_bytes, 18);
        assert_eq!(report.placed_pieces, 1);
        assert_eq!(out, vec![2, 3, 10, 11, 18, 19, 26, 27]);
        // placements that are not a gather of the range: the engine refuses
        // before any byte moves
        assert!(matches!(
            run_read(
                &[(
                    path.clone(),
                    vec![(2, 26, base, None, Some(vec![(25, 0, 2)]))]
                )],
                &host_plan(),
                None
            ),
            Err(IoError::Read(fst_core::ReadError::InvalidPlacement { .. }))
        ));
        assert!(matches!(
            run_read(
                &[(
                    path,
                    vec![(2, 26, base, None, Some(vec![(0, 0, 2), (8, 4, 2)]))]
                )],
                &host_plan(),
                None
            ),
            Err(IoError::Read(fst_core::ReadError::InvalidPlacement { .. }))
        ));
    }

    #[test]
    fn writes_are_header_then_parts_in_layout_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obj");
        let a = [1u8, 2, 3, 4];
        let b = [9u8, 8];
        let specs = vec![
            ("b".to_owned(), "U8".to_owned(), vec![2], 2),
            ("a".to_owned(), "U8".to_owned(), vec![4], 4),
            ("e".to_owned(), "U8".to_owned(), vec![0], 0),
        ];
        let parts = vec![
            (b.as_ptr() as usize, 2, false),
            (a.as_ptr() as usize, 4, false),
            (0, 0, false),
        ];
        let options = WriteOptions {
            durable: true,
            atomic: true,
            staging: StagingRing::default(),
        };
        let report = run_write(&path, &specs, &parts, None, &options, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(report.bytes, bytes.len() as u64);
        assert!(bytes.ends_with(&[1, 2, 3, 4, 9, 8]));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        assert!(matches!(
            build_parts(&[(0, 4, false)], None),
            Err(IoError::Invalid(_))
        ));
        assert!(matches!(
            build_parts(&[(a.as_ptr() as usize, 4, true)], None),
            Err(IoError::Invalid(_))
        ));
        assert!(matches!(
            run_write(&path, &specs, &parts[..2], None, &options, None),
            Err(IoError::Write(fst_core::WriteError::PartCount { .. }))
        ));
    }
}
