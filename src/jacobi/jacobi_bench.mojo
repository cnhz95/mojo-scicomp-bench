from jacobi_unoptimized import Heat2DJacobi as jacobi_unoptimized
from jacobi_v import Heat2DJacobi as jacobi_v
from jacobi_vp import Heat2DJacobi as jacobi_vp
from jacobi_vpt import Heat2DJacobi as jacobi_vpt
from jacobi_vptu import Heat2DJacobi as jacobi_vptu
from jacobi_trait import Jacobi
from time import perf_counter
from testing.testing import assert_true
from python import Python, PythonObject

comptime NX = 1 << 8
comptime NY = 1 << 8
comptime MAX_ITER = 500
comptime WARMUP_RUNS = 10
comptime BENCHMARK_RUNS = 10

@always_inline
fn benchmark[T: Jacobi](solver: T, grid_numpy: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    start_time = perf_counter()
    iters = solver.solve()
    end_time = perf_counter()

    # Verify against NumPy baseline
    for i in range(1, NX - 1):
        for j in range(1, NY - 1):
            assert_true(abs(solver[i, j] - Float64(grid_numpy[i][j])) < 1e-8,
                msg="Mismatch at (" + String(i) + "," + String(j) + ") - Mojo=" +
                String(solver[i, j]) + ", NumPy=" + String(grid_numpy[i][j])
            )
    
    return end_time - start_time, iters


fn main() raises:
    np = Python.import_module("numpy")
    stats = Python.import_module("scipy.stats")
    Python.add_to_path("src/jacobi")
    jacobi_numpy = Python.import_module("jacobi_numpy")

    ### NUMPY ###

    # Warmup
    warmup_solver_numpy = jacobi_numpy.Heat2DJacobi(NX, NY, MAX_ITER)
    _ = warmup_solver_numpy.solve()
    grid_numpy = warmup_solver_numpy.get_grid()
    for _ in range(WARMUP_RUNS - 1):
        _ = jacobi_numpy.Heat2DJacobi(NX, NY, MAX_ITER).solve()
    
    # Benchmark NumPy
    iters_numpy = np.zeros(BENCHMARK_RUNS, dtype=np.int32)
    times_numpy = np.zeros(BENCHMARK_RUNS, dtype=np.float64)
    for i in range(BENCHMARK_RUNS):
        solver_numpy = jacobi_numpy.Heat2DJacobi(NX, NY, MAX_ITER)

        start_time = perf_counter()
        iters_numpy[i] = Int(solver_numpy.solve())
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

        assert_true(solver_numpy.verify_solution(), msg="NumPy solver verification failed")
        
    assert_true(np.all(iters_numpy == iters_numpy[0]), msg="NumPy iteration count differs")
    mean_numpy = np.mean(times_numpy)
    std_numpy = np.std(times_numpy, ddof=1)
    sem = std_numpy / np.sqrt(BENCHMARK_RUNS)  # Standard error of the mean
    ci = stats.t.interval(confidence=0.95, df=BENCHMARK_RUNS-1, loc=mean_numpy, scale=sem)

    print("NumPy Jacobi 2D Heat Equation Solver")
    print("Execution time:\t", np.round(mean_numpy, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")


    ### MOJO ###
    
    iters_mojo = np.zeros(BENCHMARK_RUNS, dtype=np.int32)
    times_mojo = np.zeros(BENCHMARK_RUNS, dtype=np.float64)
    for x in range(5):
        # Warmup
        for _ in range(WARMUP_RUNS):
            if x == 0: _ = jacobi_unoptimized(NX, NY, MAX_ITER).solve()
            if x == 1: _ = jacobi_v(NX, NY, MAX_ITER).solve()
            if x == 2: _ = jacobi_vp(NX, NY, MAX_ITER).solve()
            if x == 3: _ = jacobi_vpt(NX, NY, MAX_ITER).solve()
            if x == 4: _ = jacobi_vptu(NX, NY, MAX_ITER).solve()

        # Benchmark Mojo implementations
        for i in range(BENCHMARK_RUNS):
            if x == 0: times_mojo[i], iters_mojo[i] = benchmark(jacobi_unoptimized(NX, NY, MAX_ITER), grid_numpy)
            if x == 1: times_mojo[i], iters_mojo[i] = benchmark(jacobi_v(NX, NY, MAX_ITER), grid_numpy)
            if x == 2: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vp(NX, NY, MAX_ITER), grid_numpy)
            if x == 3: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vpt(NX, NY, MAX_ITER), grid_numpy)
            if x == 4: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vptu(NX, NY, MAX_ITER), grid_numpy)

        if x == 0: print("\nUnoptimized", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo Jacobi 2D Heat Equation Solver")

        # Make sure that all solvers ran for the same number of iterations
        assert_true(np.all(iters_mojo == iters_numpy[0]), msg="Mojo and NumPy iteration count differs")

        mean_mojo = np.mean(times_mojo)
        std_mojo = np.std(times_mojo)
        sem = std_mojo / np.sqrt(BENCHMARK_RUNS)  # Standard error of the mean
        ci = stats.t.interval(confidence=0.95, df=BENCHMARK_RUNS-1, loc=mean_mojo, scale=sem)

        print("Mean time:\t", np.round(mean_mojo, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 2), "x", sep="")
