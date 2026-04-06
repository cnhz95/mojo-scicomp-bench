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
    
    # Generate data used by NumPy and Mojo
    random_signal = rng.normal(loc=0.0, scale=1.0, size=N)
    
    reals_mojo_ref = alloc[Scalar[DTYPE]](N)
    reals_mojo = alloc[Scalar[DTYPE]](N)
    imags_mojo = alloc[Scalar[DTYPE]](N)

    # Construct an UnsafePointer from the raw data pointer of the NumPy array
    parallel_memcpy(dest=reals_mojo_ref, src=random_signal.ctypes.data.unsafe_get_as_pointer[DTYPE](), count=N)


    ### NUMPY ###

    # Warmup
    output_numpy = np.fft.fft(random_signal)
    for _ in range(WARMUP_RUNS - 1):
        _ = np.fft.fft(random_signal)

    # Benchmark NumPy
    times_numpy = np.zeros(BENCHMARK_RUNS)
    for i in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        _ = np.fft.fft(random_signal)
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
            output_mojo = np.empty(Python.tuple(N), dtype=np.complex128)
            for i in range(N):
                output_mojo.real[i] = reals_mojo[i]
                output_mojo.imag[i] = imags_mojo[i]
                
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

    reals_mojo_ref.free()
    reals_mojo.free()
    imags_mojo.free()
