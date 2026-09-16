//! A simulated device: "device memory" is a byte vector per pointer, copies
//! are `memcpy`, and every call is logged. What the staging orchestration is
//! tested against without a GPU.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::{Copy2D, CudaError, DeviceCopier, DevicePtr, Fence, PinnedBuffer};

/// One recorded call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Call {
    /// `alloc_pinned(nbytes)`.
    AllocPinned(u64),
    /// `copy_to_device(nbytes, dst address, offset)`.
    ToDevice(u64, usize, u64),
    /// `copy_to_device_2d(dst address, shape)`.
    ToDevice2D(usize, Copy2D),
    /// `copy_to_host(src address, offset, nbytes)`.
    ToHost(usize, u64, u64),
    /// `synchronize()`.
    Synchronize,
    /// `record()`.
    Record,
    /// `Fence::wait()` on a recorded fence.
    Wait,
}

#[derive(Default)]
struct State {
    /// Device buffers by address.
    memory: BTreeMap<usize, Vec<u8>>,
    log: Vec<Call>,
    next_address: usize,
}

/// The simulated runtime. Clones share state.
#[derive(Clone, Default)]
pub struct SimCuda {
    state: Arc<Mutex<State>>,
    free_bytes: u64,
}

impl SimCuda {
    /// A device reporting `free_bytes` free.
    pub fn new(free_bytes: u64) -> Self {
        SimCuda {
            state: Arc::default(),
            free_bytes,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// "Allocate" a device buffer: a fresh address backed by zeroed bytes.
    pub fn alloc_device(&self, nbytes: u64, device: u32) -> DevicePtr {
        let mut state = self.lock();
        state.next_address += 4096;
        let address = state.next_address;
        state.memory.insert(address, vec![0; nbytes as usize]);
        DevicePtr {
            address,
            nbytes,
            device,
        }
    }

    /// The bytes behind a device pointer.
    pub fn device_bytes(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.lock().memory.get(&ptr.address).cloned()
    }

    /// Write `bytes` into `dst` at `offset` without a copy call: what a
    /// simulated cuFile read does, and how a test seeds a device buffer.
    /// Not logged.
    pub fn write_device(&self, dst: DevicePtr, offset: u64, bytes: &[u8]) -> Result<(), CudaError> {
        let nbytes = bytes.len() as u64;
        check(nbytes, offset, dst.nbytes)?;
        let mut state = self.lock();
        let memory = state
            .memory
            .get_mut(&dst.address)
            .ok_or(CudaError::Runtime {
                call: "cudaMemcpyAsync",
                code: 1, // cudaErrorInvalidValue
            })?;
        memory[offset as usize..(offset + nbytes) as usize].copy_from_slice(bytes);
        Ok(())
    }

    /// The calls so far.
    pub fn log(&self) -> Vec<Call> {
        self.lock().log.clone()
    }
}

struct SimPinned(Vec<u8>);

impl PinnedBuffer for SimPinned {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.0
    }
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
    fn capacity(&self) -> u64 {
        self.0.len() as u64
    }
}

fn check(nbytes: u64, offset: u64, capacity: u64) -> Result<(), CudaError> {
    if offset + nbytes > capacity {
        return Err(CudaError::Overrun {
            nbytes,
            offset,
            capacity,
        });
    }
    Ok(())
}

impl DeviceCopier for SimCuda {
    fn alloc_pinned(&self, nbytes: u64) -> Result<Box<dyn PinnedBuffer>, CudaError> {
        self.lock().log.push(Call::AllocPinned(nbytes));
        Ok(Box::new(SimPinned(vec![0; nbytes as usize])))
    }

    fn copy_to_device(
        &self,
        src: &dyn PinnedBuffer,
        nbytes: u64,
        dst: DevicePtr,
        offset: u64,
    ) -> Result<(), CudaError> {
        check(nbytes, 0, src.capacity())?;
        check(nbytes, offset, dst.nbytes)?;
        let mut state = self.lock();
        state.log.push(Call::ToDevice(nbytes, dst.address, offset));
        let memory = state
            .memory
            .get_mut(&dst.address)
            .ok_or(CudaError::Runtime {
                call: "cudaMemcpyAsync",
                code: 1, // cudaErrorInvalidValue
            })?;
        memory[offset as usize..(offset + nbytes) as usize]
            .copy_from_slice(&src.as_slice()[..nbytes as usize]);
        Ok(())
    }

    fn copy_to_device_2d(
        &self,
        src: &dyn PinnedBuffer,
        dst: DevicePtr,
        shape: Copy2D,
    ) -> Result<(), CudaError> {
        shape.check(src.capacity(), dst.nbytes)?;
        let mut state = self.lock();
        state.log.push(Call::ToDevice2D(dst.address, shape));
        let memory = state
            .memory
            .get_mut(&dst.address)
            .ok_or(CudaError::Runtime {
                call: "cudaMemcpy2DAsync",
                code: 1, // cudaErrorInvalidValue
            })?;
        let bytes = src.as_slice();
        for row in 0..shape.height {
            let from = (shape.src_offset + row * shape.src_pitch) as usize;
            let to = (shape.dst_offset + row * shape.dst_pitch) as usize;
            let width = shape.width as usize;
            memory[to..to + width].copy_from_slice(&bytes[from..from + width]);
        }
        Ok(())
    }

    fn copy_to_host(
        &self,
        src: DevicePtr,
        offset: u64,
        nbytes: u64,
        dst: &mut dyn PinnedBuffer,
    ) -> Result<(), CudaError> {
        check(nbytes, offset, src.nbytes)?;
        check(nbytes, 0, dst.capacity())?;
        let mut state = self.lock();
        state.log.push(Call::ToHost(src.address, offset, nbytes));
        let memory = state.memory.get(&src.address).ok_or(CudaError::Runtime {
            call: "cudaMemcpyAsync",
            code: 1,
        })?;
        dst.as_mut_slice()[..nbytes as usize]
            .copy_from_slice(&memory[offset as usize..(offset + nbytes) as usize]);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        self.lock().log.push(Call::Synchronize);
        Ok(())
    }

    fn record(&self) -> Result<Box<dyn Fence>, CudaError> {
        self.lock().log.push(Call::Record);
        Ok(Box::new(SimFence(self.clone())))
    }

    fn free_bytes(&self, _device: u32) -> Result<u64, CudaError> {
        Ok(self.free_bytes)
    }
}

/// Copies are synchronous in the simulator, so waiting only logs.
struct SimFence(SimCuda);

impl Fence for SimFence {
    fn wait(&self) -> Result<(), CudaError> {
        self.0.lock().log.push(Call::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_copies_land_row_by_row_and_bad_shapes_are_refused() {
        let cuda = SimCuda::new(1 << 30);
        let dst = cuda.alloc_device(8, 0);
        let mut pinned = cuda.alloc_pinned(12).unwrap_or_else(|e| panic!("{e}"));
        pinned.as_mut_slice().copy_from_slice(b"ab__cd__ef__");
        let shape = Copy2D {
            src_offset: 0,
            src_pitch: 4,
            width: 2,
            height: 3,
            dst_offset: 1,
            dst_pitch: 2,
        };
        cuda.copy_to_device_2d(pinned.as_ref(), dst, shape)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cuda.device_bytes(dst), Some(b"\0abcdef\0".to_vec()));
        assert_eq!(
            cuda.log().last(),
            Some(&Call::ToDevice2D(dst.address, shape))
        );
        // rows past the source: overrun, nothing logged
        let before = cuda.log().len();
        assert!(matches!(
            cuda.copy_to_device_2d(pinned.as_ref(), dst, Copy2D { height: 4, ..shape }),
            Err(CudaError::Overrun { capacity: 12, .. })
        ));
        // rows past the destination
        assert!(matches!(
            cuda.copy_to_device_2d(
                pinned.as_ref(),
                dst,
                Copy2D {
                    dst_offset: 3,
                    ..shape
                }
            ),
            Err(CudaError::Overrun { capacity: 8, .. })
        ));
        // rows wider than a pitch
        assert!(matches!(
            cuda.copy_to_device_2d(
                pinned.as_ref(),
                dst,
                Copy2D {
                    dst_pitch: 1,
                    ..shape
                }
            ),
            Err(CudaError::BadGeometry {
                width: 2,
                dst_pitch: 1,
                ..
            })
        ));
        assert_eq!(cuda.log().len(), before);
        // an empty copy is a no-op that is still logged
        cuda.copy_to_device_2d(pinned.as_ref(), dst, Copy2D { height: 0, ..shape })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cuda.log().len(), before + 1);
    }

    #[test]
    fn copies_land_and_overruns_are_refused() {
        let cuda = SimCuda::new(1 << 30);
        let dst = cuda.alloc_device(8, 0);
        let mut pinned = cuda.alloc_pinned(4).unwrap_or_else(|e| panic!("{e}"));
        pinned.as_mut_slice().copy_from_slice(b"abcd");
        cuda.copy_to_device(pinned.as_ref(), 4, dst, 4)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(cuda.device_bytes(dst), Some(b"\0\0\0\0abcd".to_vec()));
        assert!(matches!(
            cuda.copy_to_device(pinned.as_ref(), 4, dst, 6),
            Err(CudaError::Overrun {
                nbytes: 4,
                offset: 6,
                capacity: 8
            })
        ));
        let mut back = cuda.alloc_pinned(2).unwrap_or_else(|e| panic!("{e}"));
        cuda.copy_to_host(dst, 5, 2, back.as_mut())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back.as_slice(), b"bc");
        assert_eq!(cuda.log().len(), 4);
        let fence = cuda.record().unwrap_or_else(|e| panic!("{e}"));
        fence.wait().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&cuda.log()[4..], &[Call::Record, Call::Wait]);
    }
}
