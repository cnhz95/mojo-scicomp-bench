from nbody_unoptimized import NBodySystem as nbody_unoptimized
from nbody_v import NBodySystem as nbody_v
from nbody_vp import NBodySystem as nbody_vp
from nbody_vpt import NBodySystem as nbody_vpt
from nbody_vptu import NBodySystem as nbody_vptu
from nbody_trait import NBody
from time import perf_counter
from testing.testing import assert_true
from python import Python, PythonObject

comptime N = 1 << 10
comptime TOL = 1e-5
comptime NUM_RUNS = 10
comptime WARMUP_STEPS = 50
comptime BENCHMARK_STEPS = 50
comptime DTYPE = DType.float64

@always_inline
fn benchmark[T: NBody](system: T) raises -> Tuple[PythonObject, PythonObject, PythonObject]:
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
    assert_true(relative_error < TOL, msg="Energy drift too large: " + String(round(relative_error * 100, 4)) + "%")

    return end_time - start_time, initial_energy, final_energy
    

fn main() raises:
    np = Python.import_module("numpy")
    stats = Python.import_module("scipy.stats")
    Python.add_to_path("src/nbody")
    nbody_numpy = Python.import_module("nbody_numpy")
    rng = np.random.default_rng(42)

    # Generate data for NumPy and Mojo
    pos_data = rng.normal(loc=0.0, scale=0.5, size=Python.tuple(N, 3))
    vel_data = rng.normal(loc=0.0, scale=0.1, size=Python.tuple(N, 3))
    mass_data = np.ones(N) / N

    pos_mojo = alloc[Scalar[DTYPE]](3 * N)
    vel_mojo = alloc[Scalar[DTYPE]](3 * N)
    mass_mojo = alloc[Scalar[DTYPE]](N)

    for i in range(N):
        pos_mojo[i] = Float64(pos_data[i, 0])
        vel_mojo[i] = Float64(vel_data[i, 0])
        pos_mojo[i + N] = Float64(pos_data[i, 1])
        vel_mojo[i + N] = Float64(vel_data[i, 1])
        pos_mojo[i + 2 * N] = Float64(pos_data[i, 2])
        vel_mojo[i + 2 * N] = Float64(vel_data[i, 2])
        mass_mojo[i] = Float64(mass_data[i])
        

    ### NUMPY ###

    initial_energy_numpy = np.zeros(NUM_RUNS)
    final_energy_numpy = np.zeros(NUM_RUNS)
    times_numpy = np.zeros(NUM_RUNS)
    for i in range(NUM_RUNS):
        system_numpy = nbody_numpy.NBodySystem(N, pos_data, vel_data, mass_data)
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
        assert_true(relative_error < TOL, msg="Energy drift too large: " + String(np.round(relative_error * 100, 4)) + "%")

        initial_energy_numpy[i] = initial_energy
        final_energy_numpy[i] = final_energy

    assert_true(np.all(initial_energy_numpy == initial_energy_numpy[0]), msg="Initial energy differs between NumPy runs")
    assert_true(np.all(final_energy_numpy == final_energy_numpy[0]), msg="Final energy differs between NumPy runs")

    mean_numpy = np.mean(times_numpy)
    std_numpy = np.std(times_numpy, ddof=1)
    sem = std_numpy / np.sqrt(NUM_RUNS)  # Standard error of the mean
    ci = stats.t.interval(confidence=0.95, df=NUM_RUNS-1, loc=mean_numpy, scale=sem)

    print("NumPy N-body Simulation")
    print("Execution time:\t", np.round(mean_numpy, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")


    ### MOJO ###

    initial_energy_mojo = np.zeros(NUM_RUNS)
    final_energy_mojo = np.zeros(NUM_RUNS)
    times_mojo = np.zeros(NUM_RUNS)
    for x in range(5):
        for i in range(NUM_RUNS):
            if x == 0: times_mojo[i], initial_energy_mojo[i], final_energy_mojo[i] = benchmark(nbody_unoptimized(N, pos_mojo, vel_mojo, mass_mojo))
            if x == 1: times_mojo[i], initial_energy_mojo[i], final_energy_mojo[i] = benchmark(nbody_v(N, pos_mojo, vel_mojo, mass_mojo))
            if x == 2: times_mojo[i], initial_energy_mojo[i], final_energy_mojo[i] = benchmark(nbody_vp(N, pos_mojo, vel_mojo, mass_mojo))
            if x == 3: times_mojo[i], initial_energy_mojo[i], final_energy_mojo[i] = benchmark(nbody_vpt(N, pos_mojo, vel_mojo, mass_mojo))
            if x == 4: times_mojo[i], initial_energy_mojo[i], final_energy_mojo[i] = benchmark(nbody_vptu(N, pos_mojo, vel_mojo, mass_mojo))

        if x == 0: print("\nUnoptimized", end=" ")
        if x == 1: print("\nVectorized", end=" ")
        if x == 2: print("\nVectorized+Parallelized", end=" ")
        if x == 3: print("\nVectorized+Parallelized+Tiled", end=" ")
        if x == 4: print("\nVectorized+Parallelized+Tiled+Unrolled", end=" ")
        print("Mojo N-body Simulation")

        assert_true(np.allclose(initial_energy_mojo, initial_energy_numpy, rtol=1e-6, atol=1e-6), msg="Initial energy differs between Mojo and NumPy")
        assert_true(np.allclose(final_energy_mojo, final_energy_numpy, rtol=1e-6, atol=1e-6), msg="Final energy differs between Mojo and NumPy")

        mean_mojo = np.mean(times_mojo)
        std_mojo = np.std(times_mojo)
        sem = std_mojo / np.sqrt(NUM_RUNS)
        ci = stats.t.interval(confidence=0.95, df=NUM_RUNS-1, loc=mean_mojo, scale=sem)
        
        print("Mean time:\t", np.round(mean_mojo, 6), "s ±", np.round((ci[1] - ci[0]) / 2, 6), "s")
        print("Speedup:\t ", np.round(mean_numpy / mean_mojo, 2), "x", sep="")
    
    pos_mojo.free()
    vel_mojo.free()
    mass_mojo.free()
