from algorithm.functional import vectorize, parallelize, Static2DTileUnitFunc
from memory import memset_zero
from sys.info import simd_width_of
from testing.testing import assert_true
from time import perf_counter
from python import Python

comptime M = 1 << 12
comptime N = 1 << 12
comptime K = 1 << 12
comptime WARMUP_RUNS = 3
comptime BENCHMARK_RUNS = 10
comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2
comptime TILE_SIZE = 4

@always_inline
fn matmul_baseline(
    C: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external], 
    A: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external], 
    B: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external]
):
    for m in range(M):
        for k in range(K):
            a_mk = A[m * K + k]
            for n in range(N):
                C[m * N + n] += a_mk * B[k * N + n]


@always_inline
fn matmul_v(
    C: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external], 
    A: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external], 
    B: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external]
):
    for m in range(M):
        for k in range(K):
            a_mk = A[m * K + k]
            @parameter  # Parametric closure
            fn dot[width: Int](n: Int):
                C.store[width=width](m * N + n, C.load[width=width](m * N + n) + a_mk * B.load[width=width](k * N + n))
            vectorize[dot, NELTS](M)


@always_inline
fn matmul_vp(
    C: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external], 
    A: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external], 
    B: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external]
):
    @parameter
    fn calc_row(m: Int):
        for k in range(K):
            a_mk = A[m * K + k]
            @parameter
            fn dot[width: Int](n: Int):
                C.store[width=width](m * N + n, C.load[width=width](m * N + n) + a_mk * B.load[width=width](k * N + n))
            vectorize[dot, NELTS](M)
    parallelize[calc_row](M, M)


@always_inline
fn tile[tiled_fn: Static2DTileUnitFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

@always_inline
fn matmul_vpt(
    C: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external], 
    A: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external], 
    B: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external]
):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                a_mk = A[m * K + k]
                @parameter
                fn dot[width: Int](n: Int):
                    C.store[width=width](m * N + n + x, C.load[width=width](m * N + n + x) + a_mk * B.load[width=width](k * N + n + x))
                vectorize[dot, NELTS](tile_x)
        tile[calc_tile, NELTS * TILE_SIZE, TILE_SIZE](K, M)
    parallelize[calc_row](M, M)


@always_inline
fn matmul_vptu(
    C: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external], 
    A: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external], 
    B: UnsafePointer[mut=False, Scalar[DTYPE], MutOrigin.external]
):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                a_mk = A[m * K + k]
                @parameter
                fn dot[width: Int](n: Int):
                    C.store[width=width](m * N + n + x, C.load[width=width](m * N + n + x) + a_mk * B.load[width=width](k * N + n + x))
                alias UNROLL_FACTOR = tile_x // NELTS
                vectorize[dot, NELTS, unroll_factor=UNROLL_FACTOR](tile_x)
        tile[calc_tile, NELTS * TILE_SIZE, TILE_SIZE](K, M)
    parallelize[calc_row](M, M)


fn main() raises:
    np = Python.import_module("numpy")

    ### NUMPY ###

    C_numpy = np.zeros(Python.tuple(M, N))
    A_numpy = np.full(Python.tuple(M, K), 3.14, dtype=np.float64)
    B_numpy = np.full(Python.tuple(K, N), 3.14, dtype=np.float64)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = np.matmul(A_numpy, B_numpy)

    # Benchmark NumPy
    times_numpy = np.zeros(BENCHMARK_RUNS)
    for i in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        C_numpy = np.matmul(A_numpy, B_numpy)
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

    numpy_mean = np.mean(times_numpy)
    print("NumPy Matrix Multiplication")
    print("Mean time:\t", np.round(numpy_mean, 4), "s")
    print("Std dev:\t", np.round(np.std(times_numpy, ddof=1), 4), "s")


    ### MOJO ###

    C_mojo = alloc[Scalar[DTYPE]](M * N)
    A_mojo = alloc[Scalar[DTYPE]](M * K)
    B_mojo = alloc[Scalar[DTYPE]](K * N)

    # Initialize matrices A and B
    for m in range(M):
        for k in range(K):
            A_mojo[m * K + k] = 3.14
    for k in range(K):
        for n in range(N):
            B_mojo[k * N + n] = 3.14

    funcs = [matmul_baseline, matmul_v, matmul_vp, matmul_vpt, matmul_vptu]

    # Benchmark
    for x, func in enumerate(funcs):
        # Warmup
        for _ in range(WARMUP_RUNS):
            memset_zero(C_mojo, M * N)

            func(C_mojo, A_mojo, B_mojo)

        # Benchmark Mojo implementation
        times_mojo = np.zeros(BENCHMARK_RUNS)
        for i in range(BENCHMARK_RUNS):
            memset_zero(C_mojo, M * N)

            start_time = perf_counter()
            func(C_mojo, A_mojo, B_mojo)
            end_time = perf_counter()
            times_mojo[i] = end_time - start_time

            # Verify against NumPy baseline
            for m in range(M):
                for n in range(N):
                    assert_true(abs(C_mojo[m * N + n] - Float64(C_numpy[m][n])) > 1e-8,
                        msg="Mismatch at (" + String(m) + "," + String(n) +
                        ") - Mojo=" + String(C_mojo[m * N + n]) + ", NumPy=" + String(C_numpy[m][n])
                    )

        if x == 0: print("\nBaseline", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo Matrix Multiplication")

        mojo_mean = np.mean(times_mojo)
        print("Mean time:\t", np.round(mojo_mean, 4), "s")
        print("Std dev:\t", np.round(np.std(times_mojo, ddof=1), 4), "s")
        print("Speedup:\t ", np.round(numpy_mean / mojo_mean, 4), "x", sep="")

    C_mojo.free()
    A_mojo.free()
    B_mojo.free()
