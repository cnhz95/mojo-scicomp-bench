from fft_unoptimized import fft as fft_unoptimized
from fft_v import fft as fft_v
from fft_vp import fft as fft_vp
from fft_vpu import fft as fft_vpu
from memory import memset_zero
from algorithm import parallel_memcpy
from testing.testing import assert_true
from time import perf_counter
from python import Python

alias N = 1 << 20
alias WARMUP_RUNS = 10
alias BENCHMARK_RUNS = 10
alias DTYPE = DType.float64

fn main() raises:
    np = Python.import_module("numpy")
    stats = Python.import_module("scipy.stats")

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
    std_numpy = np.std(times_numpy, ddof=1)
    sem = std_numpy / np.sqrt(BENCHMARK_RUNS)  # Standard error of the mean
    ci = stats.t.interval(confidence=0.95, df=BENCHMARK_RUNS-1, loc=mean_numpy, scale=sem)
    
    print("NumPy FFT")
    print("Mean time:\t", np.round(mean_numpy, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")


    ### MOJO

    funcs = [fft_unoptimized, fft_v, fft_vp, fft_vpu]

    reals_mojo_ref = alloc[Scalar[DTYPE]](N)
    reals_mojo = alloc[Scalar[DTYPE]](N)
    imags_mojo = alloc[Scalar[DTYPE]](N)

    # Construct an UnsafePointer from the raw data pointer of the NumPy array
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
                ref_real = Float64(output_numpy.real[i])
                ref_imag = Float64(output_numpy.imag[i])

                err_real = reals_mojo[i] - ref_real
                err_imag = imags_mojo[i] - ref_imag
                err_mag = sqrt(err_real * err_real + err_imag * err_imag)
                ref_mag = sqrt(ref_real * ref_real + ref_imag * ref_imag)

                atol = 5e-10
                rtol = 1e-12
                assert_true(err_mag < atol + rtol * ref_mag, msg="Mismatch of " + String(err_mag) + " at position " + String(i))

        if x == 0: print("\nUnoptimized", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Unrolled", end=" ")
        print("Mojo FFT")

        mean_mojo = np.mean(times_mojo)
        std_mojo = np.std(times_mojo)
        sem = std_mojo / np.sqrt(BENCHMARK_RUNS)
        ci = stats.t.interval(confidence=0.95, df=BENCHMARK_RUNS-1, loc=mean_mojo, scale=sem)

        print("Mean time:\t", np.round(mean_mojo, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 6), "x", sep="")

    reals_mojo_ref.free()
    reals_mojo.free()
    imags_mojo.free()
