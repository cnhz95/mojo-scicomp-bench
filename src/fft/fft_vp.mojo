from math import pi, cos, sin, log2
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from testing.testing import assert_true

alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2

@always_inline
fn bit_reverse(x: Int, num_bits: Int) -> Int:
    """Reverse the bits of x using num_bits bits."""
    rev = 0
    for i in range(num_bits):
        rev = (rev << 1) | ((x >> i) & 1)
    
    return rev


@always_inline
fn fft(N: Int, reals: UnsafePointer[Scalar[DTYPE]], imags: UnsafePointer[Scalar[DTYPE]]) raises:
    """Iterative Cooley-Tukey radix-2 FFT."""
    assert_true(N > 0 and (N & (N - 1)) == 0, msg="N must be a power of 2")

    # Bit-reversal permutation
    num_bits = Int(log2(Float64(N)))
    @parameter
    fn reverse_bits(i: Int):
        j = bit_reverse(i, num_bits)
        if j > i:
            # Swaps are disjoint
            swap(reals[i], reals[j])
            swap(imags[i], imags[j])
    
    parallelize[reverse_bits](N)

    w_re = UnsafePointer[Float64].alloc(N // 2)
    w_im = UnsafePointer[Float64].alloc(N // 2)

    @parameter
    fn precompute_twiddle_table(k: Int):
        theta = -2.0 * pi * Float64(k) / Float64(N)
        w_re[k] = cos(theta)
        w_im[k] = sin(theta)
    
    parallelize[precompute_twiddle_table](N // 2)
    
    len = 2
    while len <= N:
        half = len >> 1
        w_stride = N // len  # Stride in twiddle table

        @parameter
        fn process_group(i: Int):
            # Perform butterfly operations within group
            @parameter
            fn compute_butterfly_segment[width: Int](j_offset: Int):
                even_idx = i * len + j_offset
                odd_idx = even_idx + half
                w_idx = j_offset * w_stride

                w_re_vec = w_re.offset(w_idx).strided_load[width=width](w_stride)
                w_im_vec = w_im.offset(w_idx).strided_load[width=width](w_stride)
                
                u_re = reals.load[width=width](even_idx)
                u_im = imags.load[width=width](even_idx)
                odd_re = reals.load[width=width](odd_idx)
                odd_im = imags.load[width=width](odd_idx)
                
                # Complex multiply
                v_re = odd_re * w_re_vec - odd_im * w_im_vec
                v_im = odd_re * w_im_vec + odd_im * w_re_vec
                
                # Butterfly operations
                reals.store[width=width](even_idx, u_re + v_re)
                imags.store[width=width](even_idx, u_im + v_im)
                reals.store[width=width](odd_idx, u_re - v_re)
                imags.store[width=width](odd_idx, u_im - v_im)

            vectorize[compute_butterfly_segment, NELTS](half)

        num_groups = N // len
        parallelize[process_group](num_groups)  # Parallelize over groups
        len <<= 1  # Double the size for next stage

    w_re.free()
    w_im.free()
