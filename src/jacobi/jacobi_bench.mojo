from jacobi_baseline import Heat2DJacobi as jacobi_baseline
from jacobi_v import Heat2DJacobi as jacobi_v
from jacobi_vp import Heat2DJacobi as jacobi_vp
from jacobi_vpt import Heat2DJacobi as jacobi_vpt
from jacobi_vptu import Heat2DJacobi as jacobi_vptu
from jacobi_trait import Jacobi
from time import perf_counter
from testing.testing import assert_true
from python import Python, PythonObject

alias NX = 1 << 8
alias NY = 1 << 8
alias BENCHMARK_RUNS = 10

@always_inline
fn benchmark[T: Jacobi](solver: T, grid_numpy: PythonObject) raises -> (PythonObject, PythonObject):
    start_time = perf_counter()
    iters = solver.solve()
    end_time = perf_counter()

    # Verify against NumPy baseline
    for i in range(1, NX - 1):
        for j in range(1, NY - 1):
            assert_true(abs(solver[i, j] - Float64(grid_numpy[i][j])) < 1e-6,
                msg="Mismatch at (" + String(i) + "," + String(j) + ") - Mojo=" +
                String(solver[i, j]) + ", NumPy=" + String(grid_numpy[i][j])
            )
    
    return end_time - start_time, iters


fn main() raises:
    np = Python.import_module("numpy")
    Python.add_to_path("src/jacobi")
    jacobi_numpy = Python.import_module("jacobi_numpy")

    ### NUMPY ###

    # Warmup
    warmup_solver_numpy = jacobi_numpy.Heat2DJacobi(NX, NY)
    _ = warmup_solver_numpy.solve()
    grid_numpy = warmup_solver_numpy.get_grid()
    
    # Benchmark NumPy
    iters_numpy = np.zeros(BENCHMARK_RUNS, dtype=np.int32)
    times_numpy = np.zeros(BENCHMARK_RUNS, dtype=np.float64)
    for i in range(BENCHMARK_RUNS):
        solver_numpy = jacobi_numpy.Heat2DJacobi(NX, NY)

        start_time = perf_counter()
        iters_numpy[i] = Int(solver_numpy.solve())
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

        assert_true(solver_numpy.verify_solution(), msg="NumPy solver verification failed")
        
    assert_true(np.all(iters_numpy == iters_numpy[0]), msg="NumPy iteration count differs")
    mean_numpy = np.mean(times_numpy)
    print("NumPy Jacobi 2D Heat Equation Solver")
    print("Execution time:\t", np.round(mean_numpy, 4), "s")
    print("Std dev:\t", np.round(np.std(times_numpy, ddof=1), 4), "s")


    ### MOJO ###
    
    iters_mojo = np.zeros(BENCHMARK_RUNS, dtype=np.int32)
    times_mojo = np.zeros(BENCHMARK_RUNS, dtype=np.float64)
    for x in range(5):
        # Warmup
        if x == 0: _ = jacobi_baseline(NX, NY).solve()
        if x == 1: _ = jacobi_v(NX, NY).solve()
        if x == 2: _ = jacobi_vp(NX, NY).solve()
        if x == 3: _ = jacobi_vpt(NX, NY).solve()
        if x == 4: _ = jacobi_vptu(NX, NY).solve()

        # Benchmark Mojo implementations
        for i in range(BENCHMARK_RUNS):
            if x == 0: times_mojo[i], iters_mojo[i] = benchmark(jacobi_baseline(NX, NY), grid_numpy)
            if x == 1: times_mojo[i], iters_mojo[i] = benchmark(jacobi_v(NX, NY), grid_numpy)
            if x == 2: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vp(NX, NY), grid_numpy)
            if x == 3: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vpt(NX, NY), grid_numpy)
            if x == 4: times_mojo[i], iters_mojo[i] = benchmark(jacobi_vptu(NX, NY), grid_numpy)

        if x == 0: print("\nBaseline", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo Jacobi 2D Heat Equation Solver")

        # Make sure that all solvers ran for the same number of iterations
        assert_true(np.all(iters_mojo == iters_numpy[0]), msg="Mojo and NumPy iteration count differs")

        mean_mojo = np.round(np.mean(times_mojo), 4)
        print("Mean time:\t", mean_mojo, "s")
        print("Std dev:\t", np.round(np.std(times_mojo, ddof=1), 4), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 4), "x", sep="")
