from math import pi, cos, sin, log2
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from testing.testing import assert_true

comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2
comptime UNROLL_FACTOR = 4

@always_inline
fn bit_reverse(x: Int, num_bits: Int) -> Int:
    """Reverse the bits of x using num_bits bits."""
    reverse = 0
    for i in range(num_bits):
        reverse = (reverse << 1) | ((x >> i) & 1)
    
    return reverse


@always_inline
fn fft(
    N: Int,
    reals: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external],
    imags: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
) raises:
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

    w_re = alloc[Scalar[DTYPE]](N // 2)
    w_im = alloc[Scalar[DTYPE]](N // 2)
    
    len = 2
    while len <= N:
        half = len >> 1

        @parameter
        fn compute_twiddle_factor(k: Int):
            theta = -2.0 * pi * Float64(k) / Float64(len)
            w_re[k] = cos(theta)
            w_im[k] = sin(theta)

        parallelize[compute_twiddle_factor](half)

        @parameter
        fn process_group(i: Int):
            # Perform butterfly operations within group
            @parameter
            fn compute_butterfly_segment[width: Int](offset: Int):
                even_idx = i * len + offset
                odd_idx = even_idx + half

                w_re_vec = w_re.load[width=width](offset)
                w_im_vec = w_im.load[width=width](offset)
                
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

            vectorize[compute_butterfly_segment, NELTS, unroll_factor=UNROLL_FACTOR](half)

        num_groups = N // len
        parallelize[process_group](num_groups)  # Parallelize over groups
        len <<= 1  # Double the size for next stage

    w_re.free()
    w_im.free()
