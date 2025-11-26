from fft_baseline import fft as fft_baseline
from fft_v import fft as fft_v
from fft_vp import fft as fft_vp
from fft_vpu import fft as fft_vpu
from memory import memset_zero
from algorithm import parallel_memcpy
from testing.testing import assert_true
from time import perf_counter
from python import Python

alias N = 1 << 20
alias WARMUP_RUNS = 3
alias BENCHMARK_RUNS = 10
alias DTYPE = DType.float64

fn main() raises:
    np = Python.import_module("numpy")

    ### NUMPY ###

    sine_wave = np.zeros(N)
    for i in range(N):
        sine_wave[i] = np.sin(2.0 * np.pi * i / N)

    # Warmup
    output_numpy = np.fft.fft(sine_wave)
    for _ in range(WARMUP_RUNS - 1):
        _ = np.fft.fft(sine_wave)

    # Benchmark NumPy
    times_numpy = np.zeros(BENCHMARK_RUNS)
    for i in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        _ = np.fft.fft(sine_wave)
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

    mean_numpy = np.mean(times_numpy)
    print("NumPy FFT")
    print("Mean time:\t", np.round(mean_numpy, 4), "s")
    print("Std dev:\t", np.round(np.std(times_numpy, ddof=1), 4), "s")


    ### MOJO

    funcs = [fft_baseline, fft_v, fft_vp, fft_vpu]

    reals_mojo_ref = alloc[Scalar[DTYPE]](N)
    reals_mojo = alloc[Scalar[DTYPE]](N)
    imags_mojo = alloc[Scalar[DTYPE]](N)

    # Access the raw pointer
    parallel_memcpy(dest=reals_mojo_ref, src=sine_wave.ctypes.data.unsafe_get_as_pointer[DTYPE](), count=N)

    # Benchmark
    for x, func in enumerate(funcs):
        # Warmup
        for _ in range(WARMUP_RUNS):
            parallel_memcpy(dest=reals_mojo, src=reals_mojo_ref, count=N)
            memset_zero(imags_mojo, N)

            func(N, reals_mojo, imags_mojo)

        # Benchmark Mojo implemenation
        times_mojo = np.zeros(BENCHMARK_RUNS)
        for i in range(BENCHMARK_RUNS):
            parallel_memcpy(dest=reals_mojo, src=reals_mojo_ref, count=N)
            memset_zero(imags_mojo, N)

            start_time = perf_counter()
            func(N, reals_mojo, imags_mojo)
            end_time = perf_counter()
            times_mojo[i] = end_time - start_time

            # Verify against NumPy baseline
            for i in range(N):
                err = abs(reals_mojo[i] - Float64(output_numpy.real[i])) + abs(imags_mojo[i] - Float64(output_numpy.imag[i]))
                assert_true(err < 1e-8, msg="Mismatch of " + String(err) + " at position " + String(i))

        if x == 0: print("\nBaseline", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo FFT")

        mean_mojo = np.mean(times_mojo)
        print("Mean time:\t", np.round(mean_mojo, 4), "s")
        print("Std dev:\t", np.round(np.std(times_mojo, ddof=1), 4), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 4), "x", sep="")

    reals_mojo_ref.free()
    reals_mojo.free()
    imags_mojo.free()
