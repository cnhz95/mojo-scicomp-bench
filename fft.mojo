from math import pi, cos, sin, log2
from memory import memset_zero
from time import perf_counter

alias N = 1 << 20
alias DTYPE = DType.float64

@always_inline
fn reverse_bits(x: Int, num_bits: Int) -> Int:
    """Reverse the bits of x in num_bits bits."""
    result = 0
    for i in range(num_bits):
        result = (result << 1) | ((x >> i) & 1)
    
    return result

@always_inline
fn fft(reals: UnsafePointer[Scalar[DTYPE]], imags: UnsafePointer[Scalar[DTYPE]], inverse: Bool):
    """Iterative Cooley-Tukey radix-2 FFT."""
    if not (N > 0 and (N & (N - 1)) == 0):
        print("Error: N must be power of 2")
        return

    sign = 1.0 if inverse else -1.0
    num_bits = Int(log2(Float64(N)))

    # Bit-reversal permutation
    for i in range(N):
        j = reverse_bits(i, num_bits)
        if j > i:
            swap(reals[i], reals[j])
            swap(imags[i], imags[j])

    w_re = UnsafePointer[Float64].alloc(N // 2)
    w_im = UnsafePointer[Float64].alloc(N // 2)
    
    # Precompute twiddle factors
    for k in range(N // 2):
        theta = sign * 2.0 * pi * Float64(k) / Float64(N)
        w_re[k] = cos(theta)
        w_im[k] = sin(theta)

    len = 2
    while len <= N:
        half = len >> 1
        w_stride = N // len

        for i in range(0, N, len):
            for j in range(half):
                even_idx = i + j
                odd_idx = even_idx + half
                w_idx = j * w_stride

                u_re = reals[even_idx]
                u_im = imags[even_idx]
                v_re = reals[odd_idx] * w_re[w_idx] - imags[odd_idx] * w_im[w_idx]
                v_im = reals[odd_idx] * w_im[w_idx] + imags[odd_idx] * w_re[w_idx]
                
                reals[even_idx] = u_re + v_re
                imags[even_idx] = u_im + v_im
                reals[odd_idx] = u_re - v_re
                imags[odd_idx] = u_im - v_im
                
        len <<= 1

    # Normalization for inverse FFT
    if inverse:
        scale = 1.0 / Float64(N)
        for i in range(N):
            reals[i] *= scale
            imags[i] *= scale

    w_re.free()
    w_im.free()


fn main():
    reals = UnsafePointer[Scalar[DTYPE]].alloc(N)
    imags = UnsafePointer[Scalar[DTYPE]].alloc(N)

    for i in range(N):
        reals[i] = sin(2.0 * pi * Float64(i) / Float64(N))

    memset_zero(imags, N)

    start_time = perf_counter()
    fft(reals, imags, inverse=False)
    end_time = perf_counter()

    print("Execution time:", end_time - start_time, "s")

    reals.free()
    imags.free()
    