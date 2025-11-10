from math import sqrt, iota
from memory import UnsafePointer, memset_zero
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from time import perf_counter
from random import rand
from testing import assert_true

alias N = 1 << 14
alias G = 1.0
alias DT = 0.0001
alias SOFTENING = 0.005
alias WARMUP_STEPS = 10
alias BENCHMARK_STEPS = 100
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2
alias UNROLL_FACTOR = 4

struct NBodySystem(Copyable, Movable):
    var pos_x: UnsafePointer[Scalar[DTYPE]]
    var pos_y: UnsafePointer[Scalar[DTYPE]]
    var pos_z: UnsafePointer[Scalar[DTYPE]]
    var vel_x: UnsafePointer[Scalar[DTYPE]]
    var vel_y: UnsafePointer[Scalar[DTYPE]]
    var vel_z: UnsafePointer[Scalar[DTYPE]]
    var acc_x: UnsafePointer[Scalar[DTYPE]]
    var acc_y: UnsafePointer[Scalar[DTYPE]]
    var acc_z: UnsafePointer[Scalar[DTYPE]]
    var mass: UnsafePointer[Scalar[DTYPE]]

    fn __init__(out self):
        self.pos_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.pos_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.pos_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.mass = UnsafePointer[Scalar[DTYPE]].alloc(N)

        rand(self.pos_x, N, min=-1.0, max=1.0)
        rand(self.pos_y, N, min=-1.0, max=1.0)
        rand(self.pos_z, N, min=-1.0, max=1.0)
        rand(self.vel_x, N, min=-0.1, max=0.1)
        rand(self.vel_y, N, min=-0.1, max=0.1)
        rand(self.vel_z, N, min=-0.1, max=0.1)
        rand(self.mass, N, min=0.1, max=2.0)
        self.reset_acceleration()

    fn __del__(deinit self):
        self.pos_x.free()
        self.pos_y.free()
        self.pos_z.free()
        self.vel_x.free()
        self.vel_y.free()
        self.vel_z.free()
        self.acc_x.free()
        self.acc_y.free()
        self.acc_z.free()
        self.mass.free()


    @always_inline
    fn calculate_forces(self):
        @parameter
        fn process_particle(i: Int):
            acc_x_local = 0.0
            acc_y_local = 0.0
            acc_z_local = 0.0

            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]

            @parameter
            fn calculate_force[width: Int](offset: Int):
                dx_vec = self.pos_x.load[width=width](offset) - pos_x_i
                dy_vec = self.pos_y.load[width=width](offset) - pos_y_i
                dz_vec = self.pos_z.load[width=width](offset) - pos_z_i
                mass_j_vec = self.mass.load[width=width](offset)
                
                # Avoid self-interaction
                indices = iota[DType.int32, width](offset)
                mask_vec = SIMD[DType.bool, width](fill=(indices != i)).select(1.0, 0.0)

                distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                distance_vec = sqrt(distance_squared_vec)
                force_vec = G * mass_j_vec / (distance_vec * distance_squared_vec) * mask_vec

                acc_x_local += (force_vec * dx_vec).reduce_add()
                acc_y_local += (force_vec * dy_vec).reduce_add()
                acc_z_local += (force_vec * dz_vec).reduce_add()

            vectorize[calculate_force, NELTS, unroll_factor=UNROLL_FACTOR](N)

            # Each particle writes to its own acceleration slots
            self.acc_x[i] = acc_x_local
            self.acc_y[i] = acc_y_local
            self.acc_z[i] = acc_z_local

        parallelize[process_particle](N)


    @always_inline
    fn leapfrog_integration(self):
        alias HALF_DT = 0.5 * DT

        @parameter
        fn kick_step[width: Int](offset: Int):
            self.vel_x.store[width=width](offset, self.vel_x.load[width=width](offset) + self.acc_x.load[width=width](offset) * HALF_DT)
            self.vel_y.store[width=width](offset, self.vel_y.load[width=width](offset) + self.acc_y.load[width=width](offset) * HALF_DT)
            self.vel_z.store[width=width](offset, self.vel_z.load[width=width](offset) + self.acc_z.load[width=width](offset) * HALF_DT)

        vectorize[kick_step, NELTS, unroll_factor=UNROLL_FACTOR](N)
        
        @parameter
        fn drift_step[width: Int](offset: Int):
            self.pos_x.store[width=width](offset, self.pos_x.load[width=width](offset) + self.vel_x.load[width=width](offset) * DT)
            self.pos_y.store[width=width](offset, self.pos_y.load[width=width](offset) + self.vel_y.load[width=width](offset) * DT)
            self.pos_z.store[width=width](offset, self.pos_z.load[width=width](offset) + self.vel_z.load[width=width](offset) * DT)

        vectorize[drift_step, NELTS, unroll_factor=UNROLL_FACTOR](N)

        self.reset_acceleration()
        self.calculate_forces()

        vectorize[kick_step, NELTS, unroll_factor=UNROLL_FACTOR](N)
        

    @always_inline
    fn compute_total_energy(self) -> Float64:
        energy = 0.0

        @parameter
        fn compute_kinetic_energy[width: Int](offset: Int):
            vel_x_vec = self.vel_x.load[width=width](offset)
            vel_y_vec = self.vel_y.load[width=width](offset)
            vel_z_vec = self.vel_z.load[width=width](offset)
            mass_vec = self.mass.load[width=width](offset)
            vel_squared_vec = vel_x_vec * vel_x_vec + vel_y_vec * vel_y_vec + vel_z_vec * vel_z_vec
            
            # Kinetic energy: 0.5 * m * v^2
            energy += 0.5 * (mass_vec * vel_squared_vec).reduce_add()

        vectorize[compute_kinetic_energy, NELTS, unroll_factor=UNROLL_FACTOR](N)

        potential_contributions = UnsafePointer[Scalar[DTYPE]].alloc(N)
        
        @parameter
        fn compute_contributions(i: Int):
            local_energy = 0.0
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]
            mass_i = self.mass[i]

            @parameter
            fn compute_potential_energy[width: Int](offset: Int):
                j = i + 1 + offset  # Only process j > i to avoid double counting
                dx_vec = self.pos_x.load[width=width](j) - pos_x_i
                dy_vec = self.pos_y.load[width=width](j) - pos_y_i
                dz_vec = self.pos_z.load[width=width](j) - pos_z_i
                mass_j_vec = self.mass.load[width=width](j)

                distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                distance_vec = sqrt(distance_squared_vec)
                
                # Potential energy: -G * m1 * m2 / r
                G_vec = SIMD[DTYPE, width](G)
                mass_i_vec = SIMD[DTYPE, width](mass_i)
                local_energy -= (G_vec * mass_i_vec * mass_j_vec / distance_vec).reduce_add()
                
            vectorize[compute_potential_energy, NELTS, unroll_factor=UNROLL_FACTOR](N - (i + 1))
            potential_contributions[i] = local_energy

        parallelize[compute_contributions](N)

        @parameter
        fn sum_chunk[width: Int](offset: Int):
            energy += potential_contributions.load[width=width](offset).reduce_add()

        vectorize[sum_chunk, NELTS](N)
        
        potential_contributions.free()
        return energy 


    @always_inline
    fn reset_acceleration(self):
        memset_zero(self.acc_x, N)
        memset_zero(self.acc_y, N)
        memset_zero(self.acc_z, N)


fn main() raises:
    warmup_system = NBodySystem()
    for _ in range(WARMUP_STEPS):
        warmup_system.leapfrog_integration()

    n_body_system = NBodySystem()    
    n_body_system.calculate_forces()
    initial_energy = n_body_system.compute_total_energy()

    start_time = perf_counter()
    for _ in range(BENCHMARK_STEPS):
        n_body_system.leapfrog_integration()
    end_time = perf_counter()
    
    final_energy = n_body_system.compute_total_energy()

    relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
    assert_true(relative_error < 0.01, "Energy drift too large: " + String(round(relative_error * 100, 4)) + "%")

    print("Initial system energy:\t", initial_energy)
    print("Final system energy:\t", final_energy)
    print("Execution time:\t\t",  end_time - start_time, "s")
    