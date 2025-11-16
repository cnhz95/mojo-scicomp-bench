from algorithm.functional import vectorize, parallelize, Static2DTileUnitFunc
from memory import memset_zero
from sys.info import simd_width_of
from time import perf_counter
from python import Python

alias M = 1 << 12
alias N = 1 << 12
alias K = 1 << 12
alias WARMUP_ITERS = 3
alias BENCHMARK_ITERS = 10
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2
alias TILE_SIZE = 4

@always_inline
fn matmul_baseline(C: UnsafePointer[Scalar[DTYPE]], A: UnsafePointer[Scalar[DTYPE]], B: UnsafePointer[Scalar[DTYPE]]):
    for m in range(M):
        for k in range(K):
            a_mk = A[m * K + k]
            for n in range(N):
                C[m * N + n] += a_mk * B[k * N + n]


@always_inline
fn matmul_vectorized(C: UnsafePointer[Scalar[DTYPE]], A: UnsafePointer[Scalar[DTYPE]], B: UnsafePointer[Scalar[DTYPE]]):
    for m in range(M):
        for k in range(K):
            a_mk = A[m * K + k]
            @parameter  # Parametric closure
            fn dot[nelts: Int](n: Int):
                C.store[width=nelts](m * N + n, C.load[width=nelts](m * N + n) + a_mk * B.load[width=nelts](k * N + n))
            vectorize[dot, NELTS](M)


@always_inline
fn matmul_vectorized_parallelized(C: UnsafePointer[Scalar[DTYPE]], A: UnsafePointer[Scalar[DTYPE]], B: UnsafePointer[Scalar[DTYPE]]):
    @parameter
    fn calc_row(m: Int):
        for k in range(K):
            a_mk = A[m * K + k]
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[width=nelts](m * N + n, C.load[width=nelts](m * N + n) + a_mk * B.load[width=nelts](k * N + n))
            vectorize[dot, NELTS](M)
    parallelize[calc_row](M, M)


@always_inline
fn tile[tiled_fn: Static2DTileUnitFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

@always_inline
fn matmul_vectorized_parallelized_tiled(C: UnsafePointer[Scalar[DTYPE]], A: UnsafePointer[Scalar[DTYPE]], B: UnsafePointer[Scalar[DTYPE]]):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                a_mk = A[m * K + k]
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store[width=nelts](m * N + n + x, C.load[width=nelts](m * N + n + x) + a_mk * B.load[width=nelts](k * N + n + x))
                vectorize[dot, NELTS](tile_x)
        tile[calc_tile, NELTS * TILE_SIZE, TILE_SIZE](K, M)
    parallelize[calc_row](M, M)


@always_inline
fn matmul_vectorized_parallelized_tiled_unrolled(C: UnsafePointer[Scalar[DTYPE]], A: UnsafePointer[Scalar[DTYPE]], B: UnsafePointer[Scalar[DTYPE]]):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                a_mk = A[m * K + k]
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store[width=nelts](m * N + n + x, C.load[width=nelts](m * N + n + x) + a_mk * B.load[width=nelts](k * N + n + x))
                alias UNROLL_FACTOR = tile_x // NELTS
                vectorize[dot, NELTS, unroll_factor=UNROLL_FACTOR](tile_x)
        tile[calc_tile, NELTS * TILE_SIZE, TILE_SIZE](K, M)
    parallelize[calc_row](M, M)


fn main() raises:
    np = Python.import_module("numpy")

    C_numpy = np.zeros(Python.tuple(M, N))
    A_numpy = np.full(Python.tuple(M, K), 3.14, dtype=np.float64)
    B_numpy = np.full(Python.tuple(K, N), 3.14, dtype=np.float64)

    for _ in range(WARMUP_ITERS):
        _ = np.matmul(A_numpy, B_numpy)

    # Benchmark NumPy
    times_numpy = np.zeros(BENCHMARK_ITERS)
    for i in range(BENCHMARK_ITERS):
        start_time = perf_counter()
        C_numpy = np.matmul(A_numpy, B_numpy)
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

    numpy_mean = np.mean(times_numpy)
    print("NumPy Matrix Multiplication")
    print("Mean time:\t", np.round(numpy_mean, 4), "s")
    print("Std dev:\t", np.round(np.std(times_numpy, ddof=1), 4), "s")


    ### MOJO ###

    funcs = [
        matmul_baseline,
        matmul_vectorized,
        matmul_vectorized_parallelized,
        matmul_vectorized_parallelized_tiled,
        matmul_vectorized_parallelized_tiled_unrolled
    ]

    C_mojo = UnsafePointer[Scalar[DTYPE]].alloc(M * N)
    A_mojo = UnsafePointer[Scalar[DTYPE]].alloc(M * K)
    B_mojo = UnsafePointer[Scalar[DTYPE]].alloc(K * N)

    for m in range(M):
        for k in range(K):
            A_mojo[m * K + k] = 3.14
    for k in range(K):
        for n in range(N):
            B_mojo[k * N + n] = 3.14

    # Benchmark Mojo
    for x, func in enumerate(funcs):
        for _ in range(WARMUP_ITERS):
            memset_zero(C_mojo, M * N)

            func(C_mojo, A_mojo, B_mojo)

        times_mojo = np.zeros(BENCHMARK_ITERS)
        for i in range(BENCHMARK_ITERS):
            memset_zero(C_mojo, M * N)

            start_time = perf_counter()
            func(C_mojo, A_mojo, B_mojo)
            end_time = perf_counter()
            times_mojo[i] = end_time - start_time

            # Verify against NumPy baseline
            for m in range(M):
                for n in range(N):
                    if np.abs(C_mojo[m * N + n] - C_numpy[m][n]) > 1e-8:
                        print("Error: Mismatch at (", m, ",", n, ") - Mojo=", C_mojo[m * N + n], ", NumPy=", C_numpy[m][n], sep="")
                        return

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
