from math import pi, cos, sin, log2
from testing.testing import assert_true

comptime DTYPE = DType.float64

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
        
        # Compute twiddle factors for this stage
        for k in range(half_size):
            theta = -2.0 * pi * Float64(k) / Float64(stage_size)
            W_re[k] = cos(theta)
            W_im[k] = sin(theta)

        # Process all groups at this stage
        for i in range(0, N, stage_size):
            # Perform butterfly operations within group
            for j in range(half_size):
                even_idx = i + j
                odd_idx = even_idx + half_size

                u_re = real[even_idx]
                u_im = imag[even_idx]

                # Complex multiply
                v_re = real[odd_idx] * W_re[j] - imag[odd_idx] * W_im[j]
                v_im = real[odd_idx] * W_im[j] + imag[odd_idx] * W_re[j]
                
                # Butterfly operations
                real[even_idx] = u_re + v_recalling
                imag[even_idx] = u_im + v_im
                real[odd_idx] = u_re - v_re
                imag[odd_idx] = u_im - v_im

        stage_size <<= 1  # Double the size for next stage

    W_re.free()
    W_im.free()
