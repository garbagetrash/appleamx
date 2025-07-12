use std::arch::asm;

#[inline(always)]
pub unsafe fn set() {
    unsafe {
        asm!(
            "nop", "nop", "nop",
            ".word 0x00201000 + ({op} << 5) + {operand}",
            op = const 17,
            operand = const 0,
        );
    }
}

#[inline(always)]
pub unsafe fn clr() {
    unsafe {
        asm!(
            "nop", "nop", "nop",
            ".word 0x00201000 + ({op} << 5) + {operand}",
            op = const 17,
            operand = const 1,
        );
    }
}

pub unsafe fn op<const OP: u8>(operand: u64) {
    unsafe {
        asm!(
            ".word 0x00201000 + ({op} << 5) + (0{operand} & 0x0f) + (0{operand} >> 4) * 10",
            op = const OP,
            operand = in(reg) operand,
        );
    }
}

pub unsafe fn ldx<T>(regidx: u8, ptr: *const T) {
    let gpr = (((regidx & 0x7) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<0>(gpr) };
}

pub unsafe fn ldy<T>(regidx: u8, ptr: *const T) {
    let gpr = (((regidx & 0x7) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<1>(gpr) };
}

pub unsafe fn stx<T>(regidx: u8, ptr: *mut T) {
    let gpr = (((regidx & 0x7) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<2>(gpr) };
}

pub unsafe fn sty<T>(regidx: u8, ptr: *mut T) {
    let gpr = (((regidx & 0x7) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<3>(gpr) };
}

pub unsafe fn ldz<T>(zrow: u8, ptr: *const T) {
    let gpr = (((zrow & 0x3f) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<4>(gpr) };
}

pub unsafe fn stz<T>(zrow: u8, ptr: *mut T) {
    let gpr = (((zrow & 0x3f) as u64) << 56) | (ptr as u64 & 0x00ff_ffff_ffff_ffff);
    unsafe { op::<5>(gpr) };
}

/// Performs a 16 element outer product into 16x16 matrix.
///
/// Output is strided every 4 rows of Z. So if `zrow` = 1, then output will be accessed from stz by
/// looking at rows: {1, 5, 9, 13, ect...}
///
/// `zrow`    - Chooses which of the 4 output matrixes to FMA into.
/// `xoffset` - Byte offset into X registers. 64 bytes per register (512 bits). Wraps.
/// `yoffset` - Byte offset into Y registers. 64 bytes per register (512 bits). Wraps.
pub unsafe fn fma32(zrow: u8, xoffset: u16, yoffset: u16) {
    let gpr = (((zrow & 0x3f) as u64) << 20) | (((xoffset & 0x01ff) as u64) << 10) | (yoffset & 0x01ff) as u64;
    unsafe { op::<12>(gpr) };
}

pub unsafe fn printX() {
    for ridx in 0..8 {
        let mut x: [f32; 16] = [0.0; 16];
        unsafe { stx(ridx, x.as_mut_ptr().into()) };
        println!("X[{}]: {:?}", ridx, x);
    }
}

pub unsafe fn printY() {
    for ridx in 0..8 {
        let mut y: [f32; 16] = [0.0; 16];
        unsafe { sty(ridx, y.as_mut_ptr().into()) };
        println!("Y[{}]: {:?}", ridx, y);
    }
}

pub unsafe fn printZ() {
    for ridx in 0..64 {
        let mut z: [f32; 16] = [0.0; 16];
        unsafe { stz(ridx, z.as_mut_ptr().into()) };
        println!("{:?}", z);
    }
}

/// WIP: Full matrix multiply.
pub unsafe fn matmul(a: &[[f32; 16]; 16], b: &[[f32; 16]; 16]) -> [[f32; 16]; 16] {
    //unsafe { set() };
    for ridx in 0..8 {
        unsafe { ldx(ridx, a[ridx as usize].as_ptr()) };
        unsafe { ldy(ridx, b[ridx as usize].as_ptr()) };
        //unsafe { printX() };
        //unsafe { printY() };
        unsafe { fma32(0, 64*ridx as u16, 64*ridx as u16) };
        //println!("ridx: {}", ridx);
        //unsafe { printZ() };
    }
    for ridx in 0..8 {
        unsafe { ldx(ridx, a[8 + ridx as usize].as_ptr().into()) };
        unsafe { ldy(ridx, b[8 + ridx as usize].as_ptr().into()) };
        unsafe { fma32(0, 64*ridx as u16, 64*ridx as u16) };
        //println!("ridx: {}", ridx);
        //unsafe { printZ() };
    }
    let mut output = [[0.0; 16]; 16];
    for i in 0..16 {
        unsafe { stz((4*i) as u8, output[i].as_mut_ptr().into()) };
    }
    //unsafe { clr() };
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn ldstx() {
        unsafe { set() };

        // Check each of the 8 512-bit registers
        for ridx in 0..8 {
            let x: [f32; 16] = (0..16).map(|i| (16 * ridx + i) as f32).collect::<Vec<_>>().try_into().unwrap();
            unsafe { ldx(ridx, x.as_ptr().into()) };
        }
        for ridx in 0..8 {
            let x: [f32; 16] = (0..16).map(|i| (16 * ridx + i) as f32).collect::<Vec<_>>().try_into().unwrap();
            let mut xout: [f32; 16] = [0.0; 16];
            unsafe { stx(ridx, xout.as_mut_ptr().into()) };
            for i in 0..16 {
                assert_eq!(xout[i], x[i]);
            }
        }
        unsafe { clr() };
    }

    #[test]
    fn ldsty() {
        unsafe { set() };
        let x: [f32; 16] = (0..16).map(|i| i as f32).collect::<Vec<_>>().try_into().unwrap();

        // Check each of the 8 512-bit registers
        for ridx in 0..8 {
            let mut xout: [f32; 16] = [0.0; 16];
            unsafe { ldy(ridx, x.as_ptr().into()) };
            unsafe { sty(ridx, xout.as_mut_ptr().into()) };
            for i in 0..16 {
                assert_eq!(xout[i], x[i]);
            }
        }
        unsafe { clr() };
    }

    #[test]
    fn ldstz() {
        unsafe { set() };

        // Check each of the 8 512-bit registers
        for ridx in 0..64 {
            let x: [f32; 16] = (0..16).map(|i| (16 * ridx as usize + i) as f32).collect::<Vec<_>>().try_into().unwrap();
            unsafe { ldz(ridx, x.as_ptr().into()) };
        }
        for ridx in 0..64 {
            let x: [f32; 16] = (0..16).map(|i| (16 * ridx as usize + i) as f32).collect::<Vec<_>>().try_into().unwrap();
            let mut xout: [f32; 16] = [0.0; 16];
            unsafe { stz(ridx, xout.as_mut_ptr().into()) };
            for i in 0..16 {
                assert_eq!(xout[i], x[i]);
            }
        }
        unsafe { clr() };
    }

    #[test]
    fn _fma32() {
        unsafe { set() };

        let x: [f32; 16] = (0..16).map(|i| i as f32).collect::<Vec<_>>().try_into().unwrap();
        unsafe { ldx(0, x.as_ptr().into()) };

        let y: [f32; 16] = (0..16).map(|i| i as f32).collect::<Vec<_>>().try_into().unwrap();
        unsafe { ldy(0, y.as_ptr().into()) };

        // Only 2 lsbs of first parameter (ZRow) matter. If you consider that output of a 16
        // element outer product is 16x16, and we have 64*16 values that can be stored in all of Z,
        // it's clear there are 4 times as many output spaces as required. ZRow effectively chooses
        // which to dump the matrix into. The access stride is 4 from stz. (so with ZRow = 1 in
        // fma32, stz can get first row from 1, 2nd from 5, 3rd from 9, ect....).
        unsafe { fma32(0, 0, 0) };
        for i in 0..16 {
            let mut z: [f32; 16] = [0.0; 16];
            unsafe { stz((4*i) as u8, z.as_mut_ptr().into()) };
            println!("{}: {:?}", i, z);
        }
        println!();
        unsafe { fma32(1, 4, 0) };
        for i in 0..16 {
            let mut z: [f32; 16] = [0.0; 16];
            unsafe { stz((4*i + 1) as u8, z.as_mut_ptr().into()) };
            println!("{}: {:?}", i, z);
        }
        println!();
        unsafe { fma32(2, 0, 8) };
        for i in 0..16 {
            let mut z: [f32; 16] = [0.0; 16];
            unsafe { stz((4*i + 2) as u8, z.as_mut_ptr().into()) };
            println!("{}: {:?}", i, z);
        }
        println!();
        unsafe { fma32(3, 4, 4) };
        for i in 0..16 {
            let mut z: [f32; 16] = [0.0; 16];
            unsafe { stz((4*i + 3) as u8, z.as_mut_ptr().into()) };
            println!("{}: {:?}", i, z);
        }
        println!();

        unsafe { clr() };
        assert_eq!(0, 0);
    }

    #[test]
    fn matrix_multiply() {
        let a: [[f32; 16]; 16] = (0..16).map(|i| {
            (0..16).map(|j| (i*j) as f32).collect::<Vec<_>>().try_into().unwrap()
        }).collect::<Vec<_>>().try_into().unwrap();
        let b: [[f32; 16]; 16] = (0..16).map(|i| {
            (0..16).map(|j| (i*j) as f32).collect::<Vec<_>>().try_into().unwrap()
        }).collect::<Vec<_>>().try_into().unwrap();
        for i in 0..16 {
            for j in 0..16 {
                print!("{}, ", a[j][i]);
            }
            println!();
        }
        println!();
        for i in 0..16 {
            for j in 0..16 {
                print!("{}, ", b[j][i]);
            }
            println!();
        }
        println!();
        let c = unsafe { matmul(&a, &b) };
        for i in 0..16 {
            println!("{:?}", c[i]);
        }
        assert_eq!(0, 0);
    }

    #[test]
    fn perf_matmul() {
        unsafe { set() };
        let a = [[0.0; 16]; 16];
        let mut cnt = 0;
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 2.0 {
            unsafe { matmul(&a, &a) };
            cnt += 1;
        }
        let matmuls_per_sec = cnt as f64 / 2.0;
        println!("{} 16x16 f32 matmuls per second", matmuls_per_sec);
        let flops = (2 * 16_u64.pow(3)) as f64 * matmuls_per_sec;
        println!("~{} GFLOPS", flops / 1e9);
        unsafe { clr() };
    }
}
