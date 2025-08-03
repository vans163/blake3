#[macro_use]
extern crate rustler;
extern crate rustler_codegen;

extern crate blake3;

use rustler::{types, Binary, Env, Error, NifResult, ResourceArc, OwnedBinary, Term};
use std::io::Write;
use std::sync::Mutex;

pub struct HasherResource(Mutex<blake3::Hasher>);

rustler::init!("Elixir.Blake3.Native", load = on_load);
fn on_load(env: Env, _info: Term) -> bool {
    resource!(HasherResource, env);
    true
}

mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

#[rustler::nif]
fn hash<'a>(env: Env<'a>, buf: Binary) -> NifResult<Binary<'a>> {
    let hash = blake3::hash(&buf);
    let hash_bytes = hash.as_bytes();

    let mut bin = OwnedBinary::new(hash_bytes.len()).ok_or(Error::Term(Box::new("no mem")))?;
    let _ = bin.as_mut_slice().write(hash_bytes);

    Ok(bin.release(env))
}

#[rustler::nif]
fn new() -> ResourceArc<HasherResource> {
    ResourceArc::new(HasherResource(Mutex::new(blake3::Hasher::new())))
}

#[rustler::nif]
fn update(resource: ResourceArc<HasherResource>, buf: Binary) -> ResourceArc<HasherResource> {
    {
        let mut hasher = resource.0.try_lock().unwrap();
        hasher.update(&buf);
    }

    resource
}

#[rustler::nif]
fn finalize<'a>(env: Env<'a>, resource: ResourceArc<HasherResource>) -> NifResult<Binary<'a>> {
    let hasher = resource.0.try_lock().unwrap();
    let hash_: blake3::Hash = hasher.finalize();
    let hash_bytes = hash_.as_bytes();

    let mut bin =
        types::OwnedBinary::new(hash_bytes.len()).ok_or(Error::Term(Box::new("no mem")))?;
    let _ = bin.as_mut_slice().write(hash_bytes);

    Ok(bin.release(env))
}

#[rustler::nif]
fn finalize_xof<'a>(env: Env<'a>, resource: ResourceArc<HasherResource>, output_size: usize) -> NifResult<Binary<'a>> {
    let hasher = resource.0.try_lock().unwrap();
    let mut output = vec![0u8; output_size];
    let mut output_reader = hasher.finalize_xof();
    output_reader.fill(&mut output);

    let mut bin =
        types::OwnedBinary::new(output.len()).ok_or(Error::Term(Box::new("no mem")))?;
    let _ = bin.as_mut_slice().write(&output);

    Ok(bin.release(env))
}

#[rustler::nif]
fn derive_key<'a>(env: Env<'a>, context: &str, input_key: Binary) -> NifResult<Binary<'a>> {
    let key = blake3::derive_key(context, &input_key);

    let mut bin = types::OwnedBinary::new(key.len()).ok_or(Error::Term(Box::new("no mem")))?;
    let _ = bin.as_mut_slice().write(&key);

    Ok(bin.release(env))
}

#[rustler::nif]
fn keyed_hash<'a>(env: Env<'a>, key: Binary, buf: Binary) -> NifResult<Binary<'a>> {
    let mut key_bytes: [u8; 32] = [0; 32];
    key_bytes.copy_from_slice(key.as_slice());

    let hash_ = blake3::keyed_hash(&key_bytes, &buf);
    let hash_bytes = hash_.as_bytes();

    let mut bin = types::OwnedBinary::new(hash_bytes.len()).unwrap();
    let _ = bin.as_mut_slice().write(hash_bytes);

    Ok(bin.release(env))
}

#[rustler::nif]
fn new_keyed<'a>(key: Binary) -> ResourceArc<HasherResource> {
    let mut key_bytes = [0; 32];
    key_bytes.copy_from_slice(key.as_slice());

    ResourceArc::new(HasherResource(Mutex::new(blake3::Hasher::new_keyed(
        &key_bytes,
    ))))
}

#[rustler::nif]
fn reset<'a>(resource: ResourceArc<HasherResource>) -> ResourceArc<HasherResource> {
    {
        let mut hasher = resource.0.try_lock().unwrap();
        let _ = hasher.reset();
    }

    resource
}

use std::{cell::RefCell, mem::{size_of, MaybeUninit}, ptr};
use std::arch::x86_64::*;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
compile_error!("freivalds requires AVX2; build with -C target-feature=+avx2");

#[repr(C, align(4096))]
struct AMAMatMul {
    pub A: [[u8; 50240]; 16],
    pub B: [[i8; 16]; 50240],
    pub B2: [[i8; 64]; 16],
    pub Rs: [[i8; 16]; 3],
    pub C: [[i32; 16]; 16],
}

thread_local! {
    static SCRATCH: RefCell<Option<Box<AMAMatMul>>> = RefCell::new(None);
}

struct ScratchGuard {
    buf: Option<Box<AMAMatMul>>,
}

impl std::ops::Deref for ScratchGuard {
    type Target = AMAMatMul;
    fn deref(&self) -> &Self::Target {
        self.buf.as_ref().expect("buffer disappeared")
    }
}
impl std::ops::DerefMut for ScratchGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.as_mut().expect("buffer disappeared")
    }
}
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            SCRATCH.with(|tls| *tls.borrow_mut() = Some(buf));
        }
    }
}

/// Obtain the per‑thread scratch buffer, allocating it the first time.
fn borrow_scratch() -> ScratchGuard {
    SCRATCH.with(|tls| {
        let mut slot = tls.borrow_mut();
        let buf = slot.take().unwrap_or_else(|| {
            // First time on this thread: allocate **uninitialised** memory.
            let boxed_uninit: Box<MaybeUninit<AMAMatMul>> =
                Box::new_uninit(); // ≈ zero cost for the OS here
            // SAFETY: we promise to fully overwrite every byte before reading.
            unsafe { boxed_uninit.assume_init() }
        });
        ScratchGuard { buf: Some(buf) }
    })
}

#[rustler::nif]
fn freivalds<'a>(env: Env<'a>, tensor: Binary) -> bool {
    let mut scratch = borrow_scratch();

    let mut hasher = blake3::Hasher::new();
    hasher.update(&tensor.as_slice()[..240]);
    let mut xof = hasher.finalize_xof();

    let head_bytes = 16 * 50_240         // A
                   + 50_240 * 16         // B
                   + 16 * 64             // B2
                   + 3 * 16;             // Rs

    unsafe {
        let dest = ptr::slice_from_raw_parts_mut(
            (&mut scratch.A) as *mut _ as *mut u8,
            head_bytes,
        ) as *mut [u8];
        xof.fill(&mut *dest);
    }

    let data = tensor.as_slice();
    let tail = &data[data.len() - 1024..];
    unsafe {
        let dst = &mut scratch.C as *mut _ as *mut u8;
        ptr::copy_nonoverlapping(tail.as_ptr(), dst, 1024);
    }

    unsafe {
        freivalds_inner(&scratch.Rs, &scratch.A, &scratch.B, &scratch.C)
    }
}

pub fn freivalds_inner(
    Rs: &[[i8; 16]; 3],
    A:  &[[u8; 50_240]; 16],
    B:  &[[i8; 16]; 50_240],
    C:  &[[i32; 16]; 16],
) -> bool {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { freivalds_inner_avx2(Rs, A, B, C) }
    } else {
        freivalds_inner_scalar(Rs, A, B, C)
    }
}

#[inline(always)]
unsafe fn hsum256_epi32(v: __m256i) -> i32 {
    // Reduce 8 × i32 → scalar
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);               // 4 lanes
    let sum64  = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32  = _mm_add_epi32(sum64 , _mm_srli_si128(sum64 , 4));
    _mm_cvtsi128_si32(sum32)
}

#[repr(C)]
struct I32x16 {
    lo: __m256i,
    hi: __m256i,
}

/// Load 16 × i8 and sign‑extend to 16 × i32 (as two 256‑bit halves)
#[inline(always)]
unsafe fn load_i8x16_as_i32(ptr: *const i8) -> I32x16 {
    // load 16 bytes
    let v = _mm_loadu_si128(ptr as *const __m128i);
    let lo = _mm256_cvtepi8_epi32(v);                // first 8
    let hi = _mm256_cvtepi8_epi32(_mm_srli_si128(v, 8));
    I32x16 { lo, hi }
}

pub unsafe fn freivalds_inner_avx2(
    Rs: &[[i8; 16]; 3],
    A: &[[u8; 50_240]; 16],
    B: &[[i8; 16]; 50_240],
    C: &[[i32; 16]; 16],
) -> bool {
    // the *body* is exactly what we previously had in `freivalds_inner_avx2`
    // (helpers like `hsum256_epi32` go below, unchanged)
    // ------------------------------------------------------------------ //
    const N: usize = 50_240;
    let mut U = [[0i32; 16]; 3];

    // --- Stage 1: U = C × R --------------------------------------------------
    let r0_i32 = load_i8x16_as_i32(Rs[0].as_ptr());
    let r1_i32 = load_i8x16_as_i32(Rs[1].as_ptr());
    let r2_i32 = load_i8x16_as_i32(Rs[2].as_ptr());

    for i in 0..16 {
        let c_lo = _mm256_loadu_si256(C[i].as_ptr() as *const __m256i);
        let c_hi = _mm256_loadu_si256(C[i].as_ptr().add(8) as *const __m256i);

        let u0 = _mm256_add_epi32(
            _mm256_mullo_epi32(c_lo, r0_i32.lo),
            _mm256_mullo_epi32(c_hi, r0_i32.hi),
        );
        let u1 = _mm256_add_epi32(
            _mm256_mullo_epi32(c_lo, r1_i32.lo),
            _mm256_mullo_epi32(c_hi, r1_i32.hi),
        );
        let u2 = _mm256_add_epi32(
            _mm256_mullo_epi32(c_lo, r2_i32.lo),
            _mm256_mullo_epi32(c_hi, r2_i32.hi),
        );

        U[0][i] = hsum256_epi32(u0);
        U[1][i] = hsum256_epi32(u1);
        U[2][i] = hsum256_epi32(u2);
    }

    // --- Stage 2: P(k) = B[k] · R -------------------------------------------
    let mut P0 = vec![0i32; N];
    let mut P1 = vec![0i32; N];
    let mut P2 = vec![0i32; N];

    let r0_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(Rs[0].as_ptr() as *const _));
    let r1_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(Rs[1].as_ptr() as *const _));
    let r2_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(Rs[2].as_ptr() as *const _));

    for k in 0..N {
        let row_i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(B[k].as_ptr() as *const _));

        P0[k] = hsum256_epi32(_mm256_madd_epi16(row_i16, r0_i16));
        P1[k] = hsum256_epi32(_mm256_madd_epi16(row_i16, r1_i16));
        P2[k] = hsum256_epi32(_mm256_madd_epi16(row_i16, r2_i16));
    }

    // --- Stage 3: dot( A[i], P ) --------------------------------------------
    for i in 0..16 {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();

        for k in (0..N).step_by(8) {
            let a_i32 = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64(A[i].as_ptr().add(k) as *const _));
            let p0 = _mm256_loadu_si256(P0.as_ptr().add(k) as *const _);
            let p1 = _mm256_loadu_si256(P1.as_ptr().add(k) as *const _);
            let p2 = _mm256_loadu_si256(P2.as_ptr().add(k) as *const _);

            acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(a_i32, p0));
            acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(a_i32, p1));
            acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(a_i32, p2));
        }

        if hsum256_epi32(acc0) != U[0][i]
        || hsum256_epi32(acc1) != U[1][i]
        || hsum256_epi32(acc2) != U[2][i] {
            return false;
        }
    }
    true
}

fn freivalds_inner_scalar(Rs: &[[i8; 16]; 3], A: &[[u8; 50_240]; 16], B: &[[i8; 16]; 50_240], C: &[[i32; 16]; 16]) -> bool {
    let mut U = [[0i32; 16]; 3];
    for r in 0..3 {
        for i in 0..16 {
            let mut sum = 0;
            for j in 0..16 {
                sum += C[i][j] * Rs[r][j] as i32;
            }
            U[r][i] = sum;
        }
    }

    let mut P = [[0i32; 3]; 50_240];
    for k in 0..50_240 {
        let row = &B[k];
        let mut s0 = 0;
        let mut s1 = 0;
        let mut s2 = 0;
        for j in 0..16 {
            let b = row[j] as i32;
            s0 += b * Rs[0][j] as i32;
            s1 += b * Rs[1][j] as i32;
            s2 += b * Rs[2][j] as i32;
        }
        P[k][0] = s0;
        P[k][1] = s1;
        P[k][2] = s2;
    }

    for i in 0..16 {
        let rowA = &A[i];
        let mut v0 = 0;
        let mut v1 = 0;
        let mut v2 = 0;
        for k in 0..50_240 {
            let a = rowA[k] as i32;
            let p = P[k];
            v0 += a * p[0];
            v1 += a * p[1];
            v2 += a * p[2];
        }
        if v0 != U[0][i] || v1 != U[1][i] || v2 != U[2][i] {
            return false;
        }
    }

    true
}

#[cfg(feature = "rayon")]
#[rustler::nif]
fn update_rayon<'a>(
    resource: ResourceArc<HasherResource>,
    buf: Binary,
) -> ResourceArc<HasherResource> {
    {
        let mut hasher = resource.0.try_lock().unwrap();
        hasher.update_rayon(&buf);
    }
    resource
}

#[cfg(not(feature = "rayon"))]
#[rustler::nif]
fn update_rayon<'a>(
    _resource: ResourceArc<HasherResource>,
    _buf: Binary,
) -> ResourceArc<HasherResource> {
    panic!("Blake3.update_rayon() called without rayon feature enabled");
}
