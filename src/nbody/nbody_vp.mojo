from memory import memset_zero, memcpy
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from os.atomic import Atomic
from math import sqrt, iota
from nbody_trait import NBody

alias G = 1.0
alias DT = 0.01
alias SOFTENING = 0.001
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2

struct NBodySystem(NBody):
    var N: Int
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
        

    fn __init__(out self, N: Int, pos: UnsafePointer[Scalar[DTYPE]], vel: UnsafePointer[Scalar[DTYPE]], mass: UnsafePointer[Scalar[DTYPE]]):
        self.N = N
        self.pos_x = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.pos_y = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.pos_z = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.vel_x = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.vel_y = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.vel_z = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.acc_x = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.acc_y = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.acc_z = UnsafePointer[Scalar[DTYPE]].alloc(self.N)
        self.mass = UnsafePointer[Scalar[DTYPE]].alloc(self.N)

        memcpy(self.pos_x, pos, self.N)
        memcpy(self.pos_y, pos + self.N, self.N)
        memcpy(self.pos_z, pos + 2 * self.N, self.N)
        memcpy(self.vel_x, vel, self.N)
        memcpy(self.vel_y, vel + self.N, self.N)
        memcpy(self.vel_z, vel + 2 * self.N, self.N)
        memcpy(self.mass, mass, self.N)
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
    fn compute_acceleration(self):
        """Compute gravitational acceleration."""
        @parameter
        fn compute_acceleration_for_particle(i: Int):
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]
            acc_x_local = 0.0
            acc_y_local = 0.0
            acc_z_local = 0.0

            @parameter
            fn accumulate_pairwise_acceleration[width: Int](offset: Int):
                dx_vec = self.pos_x.load[width=width](offset) - pos_x_i  # Broadcast scalar value
                dy_vec = self.pos_y.load[width=width](offset) - pos_y_i
                dz_vec = self.pos_z.load[width=width](offset) - pos_z_i
                mass_j_vec = self.mass.load[width=width](offset)
                
                # Mask out self-interaction
                indices = iota[DType.int32, width](offset)  # [offset, offset+1, offset+2, ...]
                mask_vec = SIMD[DType.bool, width](fill=(indices != i)).select(1.0, 0.0)

                distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                distance_vec = sqrt(distance_squared_vec)
                force_vec = G * mass_j_vec / (distance_vec * distance_squared_vec) * mask_vec

                acc_x_local += (force_vec * dx_vec).reduce_add()
                acc_y_local += (force_vec * dy_vec).reduce_add()
                acc_z_local += (force_vec * dz_vec).reduce_add()

            vectorize[accumulate_pairwise_acceleration, NELTS](self.N)

            # Each particle writes to its own acceleration slots
            self.acc_x[i] = acc_x_local
            self.acc_y[i] = acc_y_local
            self.acc_z[i] = acc_z_local

        parallelize[compute_acceleration_for_particle](self.N)


    @always_inline
    fn advance(self):
        """Perform one leapfrog integration step, updating positions and velocities for all particles."""
        alias HALF_DT = 0.5 * DT
        
        @parameter
        fn kick_step[width: Int](offset: Int):
            self.vel_x.store[width=width](offset, self.vel_x.load[width=width](offset) + self.acc_x.load[width=width](offset) * HALF_DT)
            self.vel_y.store[width=width](offset, self.vel_y.load[width=width](offset) + self.acc_y.load[width=width](offset) * HALF_DT)
            self.vel_z.store[width=width](offset, self.vel_z.load[width=width](offset) + self.acc_z.load[width=width](offset) * HALF_DT)

        vectorize[kick_step, NELTS](self.N)
        
        @parameter
        fn drift_step[width: Int](offset: Int):
            self.pos_x.store[width=width](offset, self.pos_x.load[width=width](offset) + self.vel_x.load[width=width](offset) * DT)
            self.pos_y.store[width=width](offset, self.pos_y.load[width=width](offset) + self.vel_y.load[width=width](offset) * DT)
            self.pos_z.store[width=width](offset, self.pos_z.load[width=width](offset) + self.vel_z.load[width=width](offset) * DT)

        vectorize[drift_step, NELTS](self.N)

        self.reset_acceleration()
        self.compute_acceleration()

        vectorize[kick_step, NELTS](self.N)
        

    @always_inline
    fn compute_energy(self) -> Float64:
        """Compute total system energy (kinetic + potential)."""
        kinetic_energy = 0.0
        potential_energy = Atomic[DTYPE](0.0)

        @parameter
        fn accumulate_kinetic_energy[width: Int](offset: Int):
            vel_x_vec = self.vel_x.load[width=width](offset)
            vel_y_vec = self.vel_y.load[width=width](offset)
            vel_z_vec = self.vel_z.load[width=width](offset)
            mass_vec = self.mass.load[width=width](offset)
            vel_squared_vec = vel_x_vec * vel_x_vec + vel_y_vec * vel_y_vec + vel_z_vec * vel_z_vec
            
            # Kinetic energy: 0.5 * m * v^2
            kinetic_energy += 0.5 * (mass_vec * vel_squared_vec).reduce_add()

        vectorize[accumulate_kinetic_energy, NELTS](self.N)
        
        @parameter
        fn accumulate_potential_for_particle(i: Int):
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]
            mass_i = self.mass[i]
            local_potential_energy = 0.0

            @parameter
            fn accumulate_potential_energy[width: Int](offset: Int):
                j = i + 1 + offset  # Avoid double counting
                dx_vec = self.pos_x.load[width=width](j) - pos_x_i
                dy_vec = self.pos_y.load[width=width](j) - pos_y_i
                dz_vec = self.pos_z.load[width=width](j) - pos_z_i
                mass_j_vec = self.mass.load[width=width](j)

                distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                distance_vec = sqrt(distance_squared_vec)
                
                # Potential energy: -G * m1 * m2 / r
                local_potential_energy -= G * mass_i * (mass_j_vec / distance_vec).reduce_add()
                
            vectorize[accumulate_potential_energy, NELTS](self.N - (i + 1))
            _ = potential_energy.fetch_add(local_potential_energy)

        parallelize[accumulate_potential_for_particle](self.N)

        return kinetic_energy + potential_energy.load()


    @always_inline
    fn reset_acceleration(self):
        memset_zero(self.acc_x, self.N)
        memset_zero(self.acc_y, self.N)
        memset_zero(self.acc_z, self.N)
