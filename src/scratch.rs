//! Pre-allocated 128-byte aligned bump-reset workspace for zero-allocation decode steps.
//!
//! Aligns every returned activation slice to [`SCRATCH_ALIGNMENT`] (`128` bytes, matching
//! `SpacemiT` K3 `A100` `VLEN = 1024`, `vlenb = 128`) so recipes can stage `norm`, `q`, `k`, `v`,
//! `attn_out`, `gate`, `up`, and `scores` buffers without per-token heap allocations.

use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use core::slice;
use std::alloc::{Layout, alloc_zeroed, dealloc};

use crate::error::K3Error;

/// Hardware vector alignment in bytes for temporary decode activations (`A100` `vlenb = 128`).
pub const SCRATCH_ALIGNMENT: usize = 128;

#[repr(C, align(128))]
struct Align128;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
}

/// Marker trait for scalar element types safely allocatable from zero-initialized bytes.
///
/// # Safety
/// Implementors must permit any initialized all-bits pattern (including all-zero bytes)
/// and have alignment strictly at most [`SCRATCH_ALIGNMENT`].
pub unsafe trait ScratchElement: Copy + sealed::Sealed {}

// SAFETY: `f32` allows every 32-bit pattern (including `0.0f32`) and has alignment 4 <= 128.
unsafe impl ScratchElement for f32 {}
// SAFETY: `i8` allows every 8-bit pattern (including `0i8`) and has alignment 1 <= 128.
unsafe impl ScratchElement for i8 {}
// SAFETY: `u8` allows every 8-bit pattern (including `0u8`) and has alignment 1 <= 128.
unsafe impl ScratchElement for u8 {}

/// Single pre-allocated 128-byte aligned bump-reset arena for temporary layer activations.
#[derive(Debug)]
pub struct ScratchWorkspace {
    ptr: NonNull<u8>,
    capacity_bytes: usize,
    cursor: usize,
    peak_used_bytes: usize,
}

// SAFETY: `ScratchWorkspace` uniquely owns its heap allocation, exposes mutable slices only via
// `&mut self`, and shares no interior mutability across threads.
unsafe impl Send for ScratchWorkspace {}
// SAFETY: Shared `&ScratchWorkspace` references read immutable geometry and counters without
// dereferencing `ptr` or mutating state.
unsafe impl Sync for ScratchWorkspace {}

impl Drop for ScratchWorkspace {
    fn drop(&mut self) {
        if self.capacity_bytes == 0 {
            return;
        }
        if let Ok(layout) = Layout::from_size_align(self.capacity_bytes, SCRATCH_ALIGNMENT) {
            // SAFETY: `self.ptr` was allocated in `ScratchWorkspace::new` with this exact
            // non-zero `layout` and remains uniquely owned until `Drop`.
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

impl ScratchWorkspace {
    /// Allocates a zero-initialized, 128-byte aligned scratch arena of `capacity_bytes` bytes.
    ///
    /// A zero `capacity_bytes` creates an empty sentinel workspace without performing an OS heap
    /// allocation (allowing empty `0`-length slices while rejecting non-zero allocations).
    ///
    /// # Errors
    /// Returns [`K3Error::InsufficientLockableMemory`] if `capacity_bytes` overflows [`Layout`]
    /// alignment bounds or if the underlying allocator returns null.
    pub fn new(capacity_bytes: usize) -> Result<Self, K3Error> {
        if capacity_bytes == 0 {
            return Ok(Self {
                ptr: NonNull::<Align128>::dangling().cast::<u8>(),
                capacity_bytes: 0,
                cursor: 0,
                peak_used_bytes: 0,
            });
        }

        let required_u64 = u64::try_from(capacity_bytes).unwrap_or(u64::MAX);
        let layout = Layout::from_size_align(capacity_bytes, SCRATCH_ALIGNMENT).map_err(|_| {
            K3Error::InsufficientLockableMemory {
                required_bytes: required_u64,
                weight_bytes: 0,
                kv_bytes: 0,
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            }
        })?;

        // SAFETY: `layout` has non-zero `size` (`capacity_bytes > 0`) and valid power-of-two
        // alignment (`SCRATCH_ALIGNMENT == 128`).
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            return Err(K3Error::InsufficientLockableMemory {
                required_bytes: required_u64,
                weight_bytes: 0,
                kv_bytes: 0,
                available_bytes: 0,
                rlimit_memlock_bytes: 0,
            });
        };

        Ok(Self {
            ptr,
            capacity_bytes,
            cursor: 0,
            peak_used_bytes: 0,
        })
    }

    /// Returns the total pre-allocated capacity in bytes.
    #[must_use]
    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Returns the current bump cursor offset in bytes.
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.cursor
    }

    /// Returns the unallocated byte headroom from the current cursor to capacity.
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.cursor)
    }

    /// Returns the peak watermark in bytes observed across the lifetime of this workspace.
    #[must_use]
    pub const fn peak_used_bytes(&self) -> usize {
        self.peak_used_bytes
    }

    /// Captures a checkpoint offset for later partial reclamation via [`Self::rewind`].
    #[must_use]
    pub const fn mark(&self) -> usize {
        self.cursor
    }

    /// Resets the bump cursor to offset `0` in O(1) without releasing the backing allocation.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Rewinds the bump cursor to a previously captured [`Self::mark`] offset.
    ///
    /// Monotonically clamps `mark` against the active cursor so stale marks captured before a
    /// [`Self::reset`] cannot advance the cursor past allocated bytes.
    pub fn rewind(&mut self, mark: usize) {
        self.cursor = mark.min(self.cursor);
    }

    /// Rewinds the bump cursor to `mark`, failing closed if `mark` exceeds the active cursor.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if `mark > self.used_bytes()`.
    pub fn try_rewind(&mut self, mark: usize) -> Result<(), K3Error> {
        if mark > self.cursor {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "scratch_workspace".to_owned(),
                offset: mark,
                byte_len: 0,
                buffer_len: self.cursor,
                reason: format!(
                    "scratch rewind mark {mark} exceeds active cursor {}",
                    self.cursor
                ),
            });
        }
        self.cursor = mark;
        Ok(())
    }

    /// Allocates a 128-byte aligned mutable `f32` slice of length `len` from the arena.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or byte size overflows `usize`, or if
    /// the allocation exceeds [`Self::capacity_bytes`].
    pub fn alloc_f32(&mut self, len: usize) -> Result<&mut [f32], K3Error> {
        self.alloc_slice_mut::<f32>(len)
    }

    /// Allocates a 128-byte aligned mutable `i8` slice of length `len` from the arena.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or byte size overflows `usize`, or if
    /// the allocation exceeds [`Self::capacity_bytes`].
    pub fn alloc_i8(&mut self, len: usize) -> Result<&mut [i8], K3Error> {
        self.alloc_slice_mut::<i8>(len)
    }

    /// Allocates a 128-byte aligned mutable `u8` slice of length `len` from the arena.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or byte size overflows `usize`, or if
    /// the allocation exceeds [`Self::capacity_bytes`].
    pub fn alloc_u8(&mut self, len: usize) -> Result<&mut [u8], K3Error> {
        self.alloc_slice_mut::<u8>(len)
    }

    /// Allocates a 128-byte aligned, zero-filled mutable `f32` slice of length `len`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if capacity is exhausted or length overflows.
    pub fn alloc_f32_zeroed(&mut self, len: usize) -> Result<&mut [f32], K3Error> {
        let slice = self.alloc_f32(len)?;
        slice.fill(0.0);
        Ok(slice)
    }

    /// Allocates a 128-byte aligned, zero-filled mutable `i8` slice of length `len`.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if capacity is exhausted or length overflows.
    pub fn alloc_i8_zeroed(&mut self, len: usize) -> Result<&mut [i8], K3Error> {
        let slice = self.alloc_i8(len)?;
        slice.fill(0);
        Ok(slice)
    }

    /// Captures a RAII checkpoint guard that automatically rewinds to the mark on drop.
    #[must_use]
    pub fn checkpoint(&mut self) -> ScratchCheckpoint<'_> {
        let mark = self.mark();
        ScratchCheckpoint {
            workspace: self,
            mark,
            _marker: PhantomData,
        }
    }

    /// Opens an interior-bump [`ScratchScope`] permitting simultaneous disjoint mutable slices
    /// (such as `norm`, `q`, `k`, `v`, `attn_out`, `gate`, `up`, and `scores`) and automatically
    /// rewinding the workspace mark when dropped.
    #[must_use]
    pub fn scope(&mut self) -> ScratchScope<'_> {
        ScratchScope {
            ptr: self.ptr,
            capacity_bytes: self.capacity_bytes,
            cursor: Cell::new(self.cursor),
            peak_used_bytes: Cell::new(self.peak_used_bytes),
            owner_peak_used_bytes: &mut self.peak_used_bytes,
            _borrow: PhantomData,
        }
    }

    fn alloc_slice_mut<T: ScratchElement>(&mut self, len: usize) -> Result<&mut [T], K3Error> {
        // SAFETY: `self.ptr` uniquely owns `self.capacity_bytes` initialized bytes for the
        // duration of `&mut self`, and `bump_alloc_slice` strictly advances `self.cursor`.
        unsafe {
            bump_alloc_slice::<T>(
                self.ptr,
                self.capacity_bytes,
                &mut self.cursor,
                &mut self.peak_used_bytes,
                len,
            )
        }
    }
}

/// Shared bump allocation helper used by both [`ScratchWorkspace`] and [`ScratchScope`].
///
/// # Safety
/// - `base` must point to a uniquely accessible zero-initialized allocation of `capacity_bytes`
///   bytes (unless `capacity_bytes == 0`), aligned to [`SCRATCH_ALIGNMENT`], valid for `'a`.
/// - No live reference may alias `[cursor..capacity_bytes]` during this call, and `*cursor` must
///   not decrease while any returned `'a` slice remains live.
unsafe fn bump_alloc_slice<'a, T: ScratchElement>(
    base: NonNull<u8>,
    capacity_bytes: usize,
    cursor: &mut usize,
    peak_used_bytes: &mut usize,
    len: usize,
) -> Result<&'a mut [T], K3Error> {
    const {
        assert!(align_of::<T>() <= SCRATCH_ALIGNMENT);
        assert!(SCRATCH_ALIGNMENT.is_power_of_two());
    }

    let aligned_start =
        align_up_checked(*cursor, SCRATCH_ALIGNMENT).ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "scratch_workspace".to_owned(),
            offset: *cursor,
            byte_len: 0,
            buffer_len: capacity_bytes,
            reason: "scratch cursor alignment overflowed usize".to_owned(),
        })?;

    if len == 0 {
        let zero_ptr = if aligned_start <= capacity_bytes {
            // SAFETY: `aligned_start <= capacity_bytes` stays within or one-past
            // the backing allocation, aligned to `SCRATCH_ALIGNMENT`.
            unsafe { base.as_ptr().add(aligned_start) }
        } else {
            NonNull::<Align128>::dangling().cast::<u8>().as_ptr()
        };
        // SAFETY: `len == 0`, `zero_ptr` is non-null and aligned to `128 >= align_of::<T>()`.
        return Ok(unsafe { slice::from_raw_parts_mut(zero_ptr.cast::<T>(), 0) });
    }

    let byte_len = len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "scratch_workspace".to_owned(),
            offset: aligned_start,
            byte_len: usize::MAX,
            buffer_len: capacity_bytes,
            reason: "scratch allocation byte length overflowed usize".to_owned(),
        })?;

    let end = aligned_start
        .checked_add(byte_len)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "scratch_workspace".to_owned(),
            offset: aligned_start,
            byte_len,
            buffer_len: capacity_bytes,
            reason: "scratch allocation end offset overflowed usize".to_owned(),
        })?;

    if end > capacity_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "scratch_workspace".to_owned(),
            offset: aligned_start,
            byte_len,
            buffer_len: capacity_bytes,
            reason: format!(
                "scratch allocation of {byte_len} bytes at aligned offset {aligned_start} exceeds capacity {capacity_bytes}"
            ),
        });
    }

    *cursor = end;
    if end > *peak_used_bytes {
        *peak_used_bytes = end;
    }

    // SAFETY: Disjoint `[aligned_start .. aligned_start + byte_len]` within initialized
    // `capacity_bytes`, 128-byte aligned (`>= align_of::<T>()`), `T: ScratchElement`.
    let slice_ptr = unsafe { base.as_ptr().add(aligned_start).cast::<T>() };
    // SAFETY: Exclusive mutable slice over `len` contiguous `T` elements for caller lifetime `'a`.
    Ok(unsafe { slice::from_raw_parts_mut(slice_ptr, len) })
}

/// Interior-cursor bump lease over a [`ScratchWorkspace`] permitting multiple simultaneous mutable
/// activation slices and automatically rewinding the workspace mark when dropped.
#[derive(Debug)]
pub struct ScratchScope<'a> {
    ptr: NonNull<u8>,
    capacity_bytes: usize,
    cursor: Cell<usize>,
    peak_used_bytes: Cell<usize>,
    owner_peak_used_bytes: &'a mut usize,
    _borrow: PhantomData<&'a mut [u8]>,
}

// Interior Cell<usize> bump cursor yields strictly disjoint mutable byte ranges per allocation call.
#[allow(clippy::mut_from_ref)]
impl ScratchScope<'_> {
    /// Returns the total workspace capacity in bytes.
    #[must_use]
    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Returns the active bump cursor offset in bytes inside this scope.
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        self.cursor.get()
    }

    /// Returns the unallocated byte headroom remaining inside this scope.
    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.cursor.get())
    }

    /// Captures a local checkpoint mark inside this scope.
    #[must_use]
    pub fn mark(&self) -> usize {
        self.cursor.get()
    }

    /// Allocates a 128-byte aligned mutable `f32` slice of length `len` within this scope.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or size overflows, or capacity is exceeded.
    pub fn alloc_f32(&self, len: usize) -> Result<&mut [f32], K3Error> {
        self.alloc_slice_mut::<f32>(len)
    }

    /// Allocates a 128-byte aligned, zero-filled mutable `f32` slice within this scope.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if capacity is exhausted or length overflows.
    pub fn alloc_f32_zeroed(&self, len: usize) -> Result<&mut [f32], K3Error> {
        let slice = self.alloc_f32(len)?;
        slice.fill(0.0);
        Ok(slice)
    }

    /// Allocates a 128-byte aligned mutable `i8` slice of length `len` within this scope.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or size overflows, or capacity is exceeded.
    pub fn alloc_i8(&self, len: usize) -> Result<&mut [i8], K3Error> {
        self.alloc_slice_mut::<i8>(len)
    }

    /// Allocates a 128-byte aligned, zero-filled mutable `i8` slice within this scope.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if capacity is exhausted or length overflows.
    pub fn alloc_i8_zeroed(&self, len: usize) -> Result<&mut [i8], K3Error> {
        let slice = self.alloc_i8(len)?;
        slice.fill(0);
        Ok(slice)
    }

    /// Allocates a 128-byte aligned mutable `u8` slice of length `len` within this scope.
    ///
    /// # Errors
    /// Returns [`K3Error::TensorBoundsError`] if alignment or size overflows, or capacity is exceeded.
    pub fn alloc_u8(&self, len: usize) -> Result<&mut [u8], K3Error> {
        self.alloc_slice_mut::<u8>(len)
    }

    fn alloc_slice_mut<T: ScratchElement>(&self, len: usize) -> Result<&mut [T], K3Error> {
        let mut cur = self.cursor.get();
        let mut peak = self.peak_used_bytes.get();
        // SAFETY: `ScratchScope` holds exclusive access over `[cursor..capacity_bytes]` for `'a`,
        // `cur` only increases monotonically while slices from `&self` remain alive, so every call
        // yields a strictly non-overlapping byte range within `self.ptr`.
        let res = unsafe {
            bump_alloc_slice::<T>(self.ptr, self.capacity_bytes, &mut cur, &mut peak, len)
        };
        self.cursor.set(cur);
        self.peak_used_bytes.set(peak);
        res
    }
}

impl Drop for ScratchScope<'_> {
    fn drop(&mut self) {
        let scope_peak = self.peak_used_bytes.get();
        if scope_peak > *self.owner_peak_used_bytes {
            *self.owner_peak_used_bytes = scope_peak;
        }
    }
}

/// Scoped RAII checkpoint that restores a [`ScratchWorkspace`] mark when dropped.
#[derive(Debug)]
pub struct ScratchCheckpoint<'a> {
    workspace: &'a mut ScratchWorkspace,
    mark: usize,
    _marker: PhantomData<&'a mut ScratchWorkspace>,
}

impl ScratchCheckpoint<'_> {
    /// Returns mutable access to the underlying [`ScratchWorkspace`] within this checkpoint scope.
    pub fn workspace(&mut self) -> &mut ScratchWorkspace {
        self.workspace
    }
}

impl Drop for ScratchCheckpoint<'_> {
    fn drop(&mut self) {
        self.workspace.rewind(self.mark);
    }
}

const fn align_up_checked(offset: usize, alignment: usize) -> Option<usize> {
    let rem = offset % alignment;
    if rem == 0 {
        Some(offset)
    } else {
        offset.checked_add(alignment - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_slices_are_128_byte_aligned_and_reusable() {
        let mut ws = ScratchWorkspace::new(512).expect("512-byte workspace");
        assert_eq!(ws.capacity_bytes(), 512);
        assert_eq!(ws.used_bytes(), 0);
        assert_eq!(ws.remaining_bytes(), 512);

        // Odd scalar count (3 * 4 = 12 bytes) advances cursor, forcing next alloc to 128B boundary.
        {
            let a = ws.alloc_f32(3).expect("alloc 3 f32");
            assert_eq!(a.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
            a.copy_from_slice(&[1.5, -2.25, 3.75]);
            assert_eq!(a, &[1.5, -2.25, 3.75]);
        }
        assert_eq!(ws.used_bytes(), 12);

        let mark = ws.mark();
        {
            let b = ws.alloc_i8(17).expect("alloc 17 i8");
            assert_eq!(b.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
            b.fill(-42);
            assert_eq!(b[0], -42);
            assert_eq!(b[16], -42);
        }
        assert_eq!(ws.used_bytes(), 128 + 17);

        {
            let c = ws.alloc_u8(64).expect("alloc 64 u8");
            assert_eq!(c.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
            c.fill(0xA5);
            assert_eq!(c[63], 0xA5);
        }
        assert_eq!(ws.used_bytes(), 256 + 64);
        assert_eq!(ws.peak_used_bytes(), 320);

        ws.try_rewind(mark).expect("valid mark");
        assert_eq!(ws.used_bytes(), mark);
        // Re-allocating over previously used offsets stays 128B aligned and Miri-clean.
        let reused = ws.alloc_f32_zeroed(8).expect("alloc zeroed after rewind");
        assert_eq!(reused.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
        assert!(reused.iter().all(|&v| v == 0.0));
        reused[7] = 9.0;

        ws.reset();
        assert_eq!(ws.used_bytes(), 0);
        assert_eq!(ws.peak_used_bytes(), 320);

        let fresh = ws.alloc_i8_zeroed(32).expect("alloc zeroed after reset");
        assert_eq!(fresh.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
        assert!(fresh.iter().all(|&v| v == 0));
    }

    #[test]
    fn capacity_exhaustion_and_invalid_bounds_return_k3_error() {
        let mut ws = ScratchWorkspace::new(256).expect("256-byte workspace");
        let _ = ws.alloc_f32(1).expect("4 bytes at offset 0");
        assert_eq!(ws.used_bytes(), 4);

        // Next alloc aligns to 128; 128 + 33 * 4 = 260 > 256 -> error, cursor untouched.
        let err = ws
            .alloc_f32(33)
            .expect_err("over-capacity allocation must fail");
        assert!(matches!(
            err,
            K3Error::TensorBoundsError {
                offset: 128,
                byte_len: 132,
                buffer_len: 256,
                ..
            }
        ));
        assert_eq!(ws.used_bytes(), 4);

        let rewind_err = ws.try_rewind(12).expect_err("future mark must fail");
        assert!(matches!(rewind_err, K3Error::TensorBoundsError { .. }));

        let overflow_err = ws.alloc_f32(usize::MAX).expect_err("overflow must fail");
        assert!(matches!(overflow_err, K3Error::TensorBoundsError { .. }));

        let huge_err = ScratchWorkspace::new(usize::MAX).expect_err("usize::MAX layout must fail");
        assert!(matches!(
            huge_err,
            K3Error::InsufficientLockableMemory { .. }
        ));
    }

    #[test]
    fn zero_capacity_empty_slices_and_raii_checkpoint() {
        let mut empty = ScratchWorkspace::new(0).expect("0-byte sentinel workspace");
        {
            let s = empty.alloc_u8(0).expect("0-length slice allowed");
            assert_eq!(s, [0u8; 0]);
            assert_eq!(s.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
        }
        assert!(matches!(
            empty.alloc_i8(1),
            Err(K3Error::TensorBoundsError { .. })
        ));

        let mut ws = ScratchWorkspace::new(384).expect("384-byte workspace");
        let _ = ws.alloc_f32(4).expect("outer allocation");
        let outer_mark = ws.mark();
        {
            let mut scope = ws.checkpoint();
            let inner = scope
                .workspace()
                .alloc_f32(16)
                .expect("inner checkpoint allocation");
            assert_eq!(inner.as_ptr() as usize % SCRATCH_ALIGNMENT, 0);
            inner[0] = 42.0;
        }
        assert_eq!(ws.used_bytes(), outer_mark);
    }

    #[test]
    fn scope_allows_simultaneous_disjoint_activations_under_miri() {
        let mut ws = ScratchWorkspace::new(768).expect("768-byte workspace");
        let _ = ws.alloc_u8(16).expect("prefix bytes");
        let saved_mark = ws.mark();

        {
            let scope = ws.scope();
            let norm = scope.alloc_f32(8).expect("norm");
            let q = scope.alloc_f32(8).expect("q");
            let quant = scope.alloc_i8(16).expect("quant");
            let gate = scope.alloc_f32_zeroed(8).expect("gate");
            let up = scope.alloc_f32_zeroed(8).expect("up");

            for addr in [
                norm.as_ptr() as usize,
                q.as_ptr() as usize,
                quant.as_ptr() as usize,
                gate.as_ptr() as usize,
                up.as_ptr() as usize,
            ] {
                assert_eq!(addr % SCRATCH_ALIGNMENT, 0);
            }

            norm.fill(1.0);
            q[0] = norm[0] + 2.0;
            quant[0] = -3;
            gate[7] = q[0];
            up[7] = gate[7] * 2.0;
            assert_eq!(up[7], 6.0);
            assert!(scope.used_bytes() > saved_mark);
        }

        // Scope drop automatically restores workspace mark and preserves peak accounting.
        assert_eq!(ws.used_bytes(), saved_mark);
        assert!(ws.peak_used_bytes() >= 640);
    }
}
