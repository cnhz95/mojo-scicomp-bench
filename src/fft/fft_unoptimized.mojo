from math import pi, cos, sin, log2
from testing.testing import assert_true

comptime DTYPE = DType.float64

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
    for i in range(N):
        j = bit_reverse(i, num_bits)
        if j > i:
            # Swaps are disjoint
            swap(reals[i], reals[j])
            swap(imags[i], imags[j])

    w_re = alloc[Scalar[DTYPE]](N // 2)
    w_im = alloc[Scalar[DTYPE]](N // 2)

    len = 2
    while len <= N:
        half = len >> 1
        
        # Compute contiguous twiddles for this stage
        for k in range(half):
            theta = -2.0 * pi * Float64(k) / Float64(len)
            w_re[k] = cos(theta)
            w_im[k] = sin(theta)

        # Process all groups at this stage
        for i in range(0, N, len):
            # Perform butterfly operations within group
            for j in range(half):
                even_idx = i + j
                odd_idx = even_idx + half

                u_re = reals[even_idx]
                u_im = imags[even_idx]

                # Complex multiply
                v_re = reals[odd_idx] * w_re[j] - imags[odd_idx] * w_im[j]
                v_im = reals[odd_idx] * w_im[j] + imags[odd_idx] * w_re[j]
                
                # Butterfly operations
                reals[even_idx] = u_re + v_re
                imags[even_idx] = u_im + v_im
                reals[odd_idx] = u_re - v_re
                imags[odd_idx] = u_im - v_im

        len <<= 1  # Double the size for next stage

    w_re.free()
    w_im.free()
