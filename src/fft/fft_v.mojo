from math import pi, cos, sin, log2
from algorithm.functional import vectorize, parallelize
from memory import memcpy, memset_zero
from sys.info import simd_width_of
from testing.testing import assert_true

comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2

@always_inline
fn bit_reverse(x: Int, num_bits: Int) -> Int:
    """Reverse the bits of x using num_bits bits."""
    rev = 0
    for i in range(num_bits):
        rev = (rev << 1) | ((x >> i) & 1)
    
    return rev


@always_inline
fn fft(
    N: Int,
    real: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external],
    imag: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
) raises:
    """Iterative Cooley-Tukey radix-2 FFT."""
    assert_true(N > 0 and (N & (N - 1)) == 0, msg="N must be a power of 2")

    # Bit-reversal permutation
    num_bits = Int(log2(Float64(N)))
    for i in range(N):
        j = bit_reverse(i, num_bits)
        if j > i:
            # Swaps are disjoint
            swap(real[i], real[j])
            swap(imag[i], imag[j])

    W_re = alloc[Scalar[DTYPE]](N // 2)
    W_im = alloc[Scalar[DTYPE]](N // 2)
    
    stage_size = 2
    while stage_size <= N:
        half_size = stage_size >> 1

        # Compute contiguous twiddles for this stage
        for k in range(half_size):
            theta = -2.0 * pi * Float64(k) / Float64(stage_size)
            W_re[k] = cos(theta)
            W_im[k] = sin(theta)

        # Process all groups at this stage
        for i in range(0, N, stage_size):
            # Perform butterfly operations within group
            @parameter
            fn compute_butterfly_segment[width: Int](offset: Int):
                even_idx = i + offset
                odd_idx = even_idx + half_size

                W_re_vec = W_re.load[width=width](offset)
                W_im_vec = W_im.load[width=width](offset)
                
                u_re = real.load[width=width](even_idx)
                u_im = imag.load[width=width](even_idx)
                odd_re = real.load[width=width](odd_idx)
                odd_im = imag.load[width=width](odd_idx)
                
                # Complex multiply
                v_re = odd_re * W_re_vec - odd_im * W_im_vec
                v_im = odd_re * W_im_vec + odd_im * W_re_vec
                
                # Butterfly operations
                real.store[width=width](even_idx, u_re + v_re)
                imag.store[width=width](even_idx, u_im + v_im)
                real.store[width=width](odd_idx, u_re - v_re)
                imag.store[width=width](odd_idx, u_im - v_im)

            vectorize[compute_butterfly_segment, NELTS](half_size)

        stage_size <<= 1  # Double the size for next stage

    W_re.free()
    W_im.free()
