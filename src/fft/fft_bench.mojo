from fft_unoptimized import fft as fft_unoptimized
from fft_v import fft as fft_v
from fft_vp import fft as fft_vp
from fft_vpu import fft as fft_vpu
from memory import memset_zero
from algorithm import parallel_memcpy
from testing.testing import assert_true
from time import perf_counter
from python import Python

comptime N = 1 << 20
comptime WARMUP_RUNS = 10
comptime BENCHMARK_RUNS = 10
comptime DTYPE = DType.float64

fn main() raises:
    np = Python.import_module("numpy")
    stats = Python.import_module("scipy.stats")
    rng = np.random.default_rng(42)
    
    # Generate data for NumPy and Mojo
    input_signal = rng.normal(loc=0.0, scale=1.0, size=N)
    
    input_real_mojo = alloc[Scalar[DTYPE]](N)
    real_mojo = alloc[Scalar[DTYPE]](N)
    imag_mojo = alloc[Scalar[DTYPE]](N)

    # Construct an UnsafePointer from the raw data pointer of the NumPy array
    parallel_memcpy(dest=input_real_mojo, src=input_signal.ctypes.data.unsafe_get_as_pointer[DTYPE](), count=N)


    ### NUMPY ###

    # Warmup
    output_numpy = np.fft.fft(input_signal)
    for _ in range(WARMUP_RUNS - 1):
        _ = np.fft.fft(input_signal)

    # Benchmark NumPy
    times_numpy = np.zeros(BENCHMARK_RUNS)
    for i in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        _ = np.fft.fft(input_signal)
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

    # Benchmark
    for x, func in enumerate(funcs):
        # Warmup
        for _ in range(WARMUP_RUNS):
            parallel_memcpy(dest=real_mojo, src=input_real_mojo, count=N)
            memset_zero(imag_mojo, N)

            func(N, real_mojo, imag_mojo)

        # Benchmark Mojo implemenation
        times_mojo = np.zeros(BENCHMARK_RUNS)
        for i in range(BENCHMARK_RUNS):
            parallel_memcpy(dest=real_mojo, src=input_real_mojo, count=N)
            memset_zero(imag_mojo, N)

            start_time = perf_counter()
            func(N, real_mojo, imag_mojo)
            end_time = perf_counter()
            times_mojo[i] = end_time - start_time

            # Verify against NumPy baseline
            output_mojo = np.empty(Python.tuple(N), dtype=np.complex128)
            for i in range(N):
                output_mojo.real[i] = real_mojo[i]
                output_mojo.imag[i] = imag_mojo[i]
                
            rel_l2 = np.linalg.norm(output_mojo - output_numpy) / np.linalg.norm(output_numpy)
            assert_true(rel_l2 < 1e-12, msg="Relative L2 error is too large: " + String(rel_l2))

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

    input_real_mojo.free()
    real_mojo.free()
    imag_mojo.free()
