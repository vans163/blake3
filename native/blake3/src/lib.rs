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

#[repr(C, align(4096))]
struct AMAMatMul {
    pub A: [[i8; 50240]; 16],
    pub B: [[i8; 16]; 50240],
    pub B2: [[i8; 64]; 16],
    pub R: [i8; 16],
    pub C: [[i32; 16]; 16],
}

/*
use std::time::Instant;

#[rustler::nif]
fn freivalds<'a>(env: Env<'a>, tensor: Binary) {
    let mut uninit_array: Box<std::mem::MaybeUninit<[AMAMatMul; 1]>> = Box::new_uninit();
    let ptr0: *mut AMAMatMul = uninit_array.as_mut_ptr() as *mut AMAMatMul;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&tensor.as_slice()[..240]);
    let mut xof = hasher.finalize_xof();
    unsafe {
        let buf = std::slice::from_raw_parts_mut(ptr0 as *mut u8, 16*50240 + 50240*16 + 16*64 + 16);
        xof.fill(buf);
    };

    let data = tensor.as_slice();
    let tail = &data[data.len() - 1024 ..];
    unsafe {
        let c_ptr = &mut (*ptr0).C as *mut [[i32;16];16] as *mut u8;
        std::ptr::copy_nonoverlapping(tail.as_ptr(), c_ptr, 1024);
    }
    let struct_ama_matmul: Box<[AMAMatMul; 1]> = unsafe { uninit_array.assume_init() };


    let mat = &struct_ama_matmul[0];
    freivalds_inner(&mat.R, &mat.A, &mat.B, &mat.C);

    //Ok(atoms::ok())
}
*/

#[rustler::nif]
fn freivalds<'a>(env: Env<'a>, tensor: Binary) -> bool {
    //1.5ms fix the page faults
    let mut struct_ama_matmul = Box::new([AMAMatMul {
        A: [[0; 50240]; 16],
        B: [[0; 16]; 50240],
        B2: [[0; 64]; 16],
        R: [0; 16],
        C: [[0; 16]; 16],
    }; 1]);

    let mut hasher = blake3::Hasher::new();
    hasher.update(&tensor.as_slice()[..240]);
    let mut xof = hasher.finalize_xof();
    unsafe {
        let buf = std::slice::from_raw_parts_mut(&mut struct_ama_matmul[0] as *mut _ as *mut u8, 16*50240 + 50240*16 + 16*64 + 16);
        xof.fill(buf);
    };

    let data = tensor.as_slice();
    let tail = &data[data.len() - 1024 ..];
    unsafe {
        let dst = &mut struct_ama_matmul[0].C as *mut [[i32;16];16] as *mut u8;
        std::ptr::copy_nonoverlapping(tail.as_ptr(), dst, 1024);
    }

    let mat = &struct_ama_matmul[0];
    freivalds_inner(&mat.R, &mat.A, &mat.B, &mat.C)
}

fn freivalds_inner(R: &[i8; 16], A: &[[i8; 50_240]; 16], B: &[[i8; 16]; 50_240], C: &[[i32; 16]; 16]) -> bool {
    let mut P = [0i32; 50_240];
    for i in 0..50_240 {
        let mut sum: i32 = 0;
        for j in 0..16 {
            sum += B[i][j] as i32 * R[j] as i32;
        }
        P[i] = sum;
    }

    let mut V = [0i32; 16];
    for i in 0..16 {
        let mut sum: i32 = 0;
        for k in 0..50_240 {
            sum += A[i][k] as i32 * P[k] as i32;
        }
        V[i] = sum;
    }

    let mut U = [0i32; 16];
    for i in 0..16 {
        let mut sum: i32 = 0;
        for j in 0..16 {
            sum += C[i][j] * R[j] as i32;
        }
        U[i] = sum;
    }

    V == U
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
