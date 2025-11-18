from nbody_baseline import NBodySystem as nbody_baseline
from nbody_v import NBodySystem as nbody_v
from nbody_vp import NBodySystem as nbody_vp
from nbody_vpt import NBodySystem as nbody_vpt
from nbody_vptu import NBodySystem as nbody_vptu
from nbody_trait import NBody
from time import perf_counter
from testing.testing import assert_true
from python import Python

alias N = 1 << 10
alias TOL = 0.005
alias NUM_RUNS = 10
alias WARMUP_STEPS = 10
alias BENCHMARK_STEPS = 100
alias DTYPE = DType.float64

@always_inline
fn benchmark[T: NBody](system: T) raises -> Float64:
    system.compute_acceleration()

    # Warmup
    for _ in range(WARMUP_STEPS):
        system.advance()

    initial_energy = system.compute_energy()

    # Benchmark Mojo implementation
    start_time = perf_counter()
    for _ in range(BENCHMARK_STEPS):
        system.advance()
    end_time = perf_counter()

    final_energy = system.compute_energy()
    
    relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
    assert_true(relative_error < TOL, "Energy drift too large: " + String(round(relative_error * 100, 4)) + "%")

    return end_time - start_time
    

fn main() raises:
    np = Python.import_module("numpy")
    Python.add_to_path("src/nbody")
    nbody_numpy = Python.import_module("nbody_numpy")
    np.random.seed(42)

    # Generate data used by both NumPy and Mojo
    pos_data = np.random.uniform(-1.0, 1.0, Python.tuple(N, 3))
    vel_data = np.random.uniform(-0.1, 0.1, Python.tuple(N, 3))
    acc_data = np.zeros(Python.tuple(N, 3))
    mass_data = np.random.uniform(0.1, 1.0, N)

    ### NUMPY ###

    times_numpy = np.zeros(NUM_RUNS)
    for i in range(NUM_RUNS):
        system_numpy = nbody_numpy.NBodySystem(N, pos_data, vel_data, acc_data, mass_data)
        system_numpy.compute_acceleration()

        # Warmup
        for _ in range(WARMUP_STEPS):
            system_numpy.advance()

        initial_energy = system_numpy.compute_energy()

        # Benchmark NumPy
        start_time = perf_counter()
        for _ in range(BENCHMARK_STEPS):
            system_numpy.advance()
        end_time = perf_counter()
        times_numpy[i] = end_time - start_time

        final_energy = system_numpy.compute_energy()

        relative_error = np.abs(final_energy - initial_energy) / np.abs(initial_energy)
        assert_true(relative_error < TOL, "Energy drift too large: " + String(np.round(relative_error * 100, 4)) + "%")
        
    mean_numpy = np.round(np.mean(times_numpy), 4)
    print("NumPy N-body Simulation")
    print("Mean time:\t", mean_numpy, "s")
    print("Std dev:\t", np.round(np.std(times_numpy, ddof=1), 4), "s")


    ### MOJO ###

    pos_mojo = UnsafePointer[Scalar[DTYPE]].alloc(3 * N)
    vel_mojo = UnsafePointer[Scalar[DTYPE]].alloc(3 * N)
    mass_mojo = UnsafePointer[Scalar[DTYPE]].alloc(3 * N)

    for i in range(N):
        pos_mojo[i] = Float64(pos_data[i][0])
        vel_mojo[i] = Float64(vel_data[i][0])
        pos_mojo[i + N] = Float64(pos_data[i][1])
        vel_mojo[i + N] = Float64(vel_data[i][1])
        pos_mojo[i + 2 * N] = Float64(pos_data[i][2])
        vel_mojo[i + 2 * N] = Float64(vel_data[i][2])
        mass_mojo[i] = Float64(mass_data[i])

    times_mojo = np.zeros(NUM_RUNS)
    for x in range(5):
        for i in range(NUM_RUNS):
            if x == 0: times_mojo[i] = benchmark(nbody_baseline(pos_mojo, vel_mojo, mass_mojo, N))
            if x == 1: times_mojo[i] = benchmark(nbody_v(pos_mojo, vel_mojo, mass_mojo, N))
            if x == 2: times_mojo[i] = benchmark(nbody_vp(pos_mojo, vel_mojo, mass_mojo, N))
            if x == 3: times_mojo[i] = benchmark(nbody_vpt(pos_mojo, vel_mojo, mass_mojo, N))
            if x == 4: times_mojo[i] = benchmark(nbody_vptu(pos_mojo, vel_mojo, mass_mojo, N))

        if x == 0: print("\nBaseline", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo N-body Simulation")

        mean_mojo = np.round(np.mean(times_mojo), 4)
        print("Mean time:\t", mean_mojo, "s")
        print("Std dev:\t", np.round(np.std(times_mojo, ddof=1), 4), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 4), "x", sep="")

    pos_mojo.free()
    vel_mojo.free()
    mass_mojo.free()
