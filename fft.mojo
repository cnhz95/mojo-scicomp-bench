from math import pi, cos, sin, log2
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from memory import memset_zero
from time import perf_counter

alias N = 1 << 20
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2
alias UNROLL_FACTOR = 4

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
    @parameter
    fn bit_reverse(i: Int):
        j = reverse_bits(i, num_bits)
        if j > i:
            swap(reals[i], reals[j])
            swap(imags[i], imags[j])
    
    parallelize[bit_reverse](N)  # The swaps are disjoint

    w_re = UnsafePointer[Float64].alloc(N // 2)
    w_im = UnsafePointer[Float64].alloc(N // 2)

    @parameter
    fn precompute_twiddle_table(k: Int):
        theta = sign * 2.0 * pi * Float64(k) / Float64(N)
        w_re[k] = cos(theta)
        w_im[k] = sin(theta)
    
    parallelize[precompute_twiddle_table](N // 2)
    
    len = 2
    while len <= N:
        half = len >> 1
        w_stride = N // len  # Stride in twiddle table

        @parameter
        fn process_block(i: Int):
            @parameter
            fn butterfly_computation[width: Int](j_offset: Int):
                even_base = (i * len) + j_offset 
                odd_base = even_base + half
                w_base = j_offset * w_stride

                var w_re = w_re.offset(w_base).strided_load[width=width](w_stride)
                var w_im = w_im.offset(w_base).strided_load[width=width](w_stride)
                
                u_re = reals.load[width=width](even_base)
                u_im = imags.load[width=width](even_base)
                odd_re = reals.load[width=width](odd_base)
                odd_im = imags.load[width=width](odd_base)
                
                v_re = odd_re * w_re - odd_im * w_im
                v_im = odd_re * w_im + odd_im * w_re
                
                reals.store[width=width](even_base, u_re + v_re)
                imags.store[width=width](even_base, u_im + v_im)
                reals.store[width=width](odd_base, u_re - v_re)
                imags.store[width=width](odd_base, u_im - v_im)

            vectorize[butterfly_computation, NELTS, unroll_factor=UNROLL_FACTOR](half)

        parallelize[process_block](N // len)
        len <<= 1

    # Normalization for inverse FFT
    if inverse:
        scale = 1.0 / Float64(N)
        @parameter
        fn normalize[width: Int](offset: Int):
            reals.store[width=width](offset, reals.load[width=width](offset) * scale)
            imags.store[width=width](offset, imags.load[width=width](offset) * scale)
        
        vectorize[normalize, NELTS, unroll_factor=UNROLL_FACTOR](N)

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
    