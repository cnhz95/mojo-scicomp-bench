from memory import memset_zero
from algorithm import parallel_memcpy
from algorithm.functional import vectorize
from sys.info import simd_width_of
from math import sqrt, iota
from nbody_trait import NBody

comptime G = 1.0
comptime DT = 0.001
comptime SOFTENING = 0.001
comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2

struct NBodySystem(NBody):
    var N: Int
    var pos_x: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var pos_y: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var pos_z: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var vel_x: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var vel_y: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var vel_z: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var acc_x: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var acc_y: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var acc_z: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var mass: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]


    fn __init__(
        out self,
        N: Int, 
        pos: UnsafePointer[mut=False, Scalar[DTYPE]], 
        vel: UnsafePointer[mut=False, Scalar[DTYPE]], 
        mass: UnsafePointer[mut=False, Scalar[DTYPE]]
    ):
        self.N = N
        self.pos_x = alloc[Scalar[DTYPE]](N)
        self.pos_y = alloc[Scalar[DTYPE]](N)
        self.pos_z = alloc[Scalar[DTYPE]](N)
        self.vel_x = alloc[Scalar[DTYPE]](N)
        self.vel_y = alloc[Scalar[DTYPE]](N)
        self.vel_z = alloc[Scalar[DTYPE]](N)
        self.acc_x = alloc[Scalar[DTYPE]](N)
        self.acc_y = alloc[Scalar[DTYPE]](N)
        self.acc_z = alloc[Scalar[DTYPE]](N)
        self.mass = alloc[Scalar[DTYPE]](N)

        parallel_memcpy(dest=self.pos_x, src=pos, count=N)
        parallel_memcpy(dest=self.pos_y, src=pos + N, count=N)
        parallel_memcpy(dest=self.pos_z, src=pos + 2 * N, count=N)
        parallel_memcpy(dest=self.vel_x, src=vel, count=N)
        parallel_memcpy(dest=self.vel_y, src=vel + N, count=N)
        parallel_memcpy(dest=self.vel_z, src=vel + 2 * N, count=N)
        parallel_memcpy(dest=self.mass, src=mass, count=N)
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
        for i in range(self.N):
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]

            @parameter
            fn accumulate_pairwise_acceleration[width: Int](offset: Int):
                dx_vec = self.pos_x.load[width=width](offset) - pos_x_i  # Broadcast scalar value
                dy_vec = self.pos_y.load[width=width](offset) - pos_y_i
                dz_vec = self.pos_z.load[width=width](offset) - pos_z_i
                mass_j_vec = self.mass.load[width=width](offset)
                
                # Mask out self-interaction
                indices = iota[DType.int32, width](offset)  # [offset, offset+1, offset+2, ..., offset+width-1]
                mask_vec = indices.ne(i).select(1.0, 0.0)

                distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                distance_vec = sqrt(distance_squared_vec)
                f_vec = G * mass_j_vec / (distance_vec * distance_squared_vec) * mask_vec

                self.acc_x[i] += (f_vec * dx_vec).reduce_add()
                self.acc_y[i] += (f_vec * dy_vec).reduce_add()
                self.acc_z[i] += (f_vec * dz_vec).reduce_add()

            vectorize[accumulate_pairwise_acceleration, NELTS](self.N)


    @always_inline
    fn advance(self):
        """Perform one leapfrog integration step, updating positions and velocities for all particles."""
        comptime HALF_DT = 0.5 * DT

        @parameter
        fn kick_step[width: Int](i: Int):
            self.vel_x.store[width=width](i, self.vel_x.load[width=width](i) + self.acc_x.load[width=width](i) * HALF_DT)
            self.vel_y.store[width=width](i, self.vel_y.load[width=width](i) + self.acc_y.load[width=width](i) * HALF_DT)
            self.vel_z.store[width=width](i, self.vel_z.load[width=width](i) + self.acc_z.load[width=width](i) * HALF_DT)

        vectorize[kick_step, NELTS](self.N)
        
        @parameter
        fn drift_step[width: Int](i: Int):
            self.pos_x.store[width=width](i, self.pos_x.load[width=width](i) + self.vel_x.load[width=width](i) * DT)
            self.pos_y.store[width=width](i, self.pos_y.load[width=width](i) + self.vel_y.load[width=width](i) * DT)
            self.pos_z.store[width=width](i, self.pos_z.load[width=width](i) + self.vel_z.load[width=width](i) * DT)

        vectorize[drift_step, NELTS](self.N)

        self.reset_acceleration()
        self.compute_acceleration()

        vectorize[kick_step, NELTS](self.N)


    @always_inline
    fn compute_energy(self) -> Float64:
        """Compute total system energy (kinetic + potential)."""
        energy = 0.0

        @parameter
        fn accumulate_kinetic_energy[width: Int](offset: Int):
            vel_x_vec = self.vel_x.load[width=width](offset)
            vel_y_vec = self.vel_y.load[width=width](offset)
            vel_z_vec = self.vel_z.load[width=width](offset)
            mass_vec = self.mass.load[width=width](offset)

            vel_squared_vec = vel_x_vec * vel_x_vec + vel_y_vec * vel_y_vec + vel_z_vec * vel_z_vec
            
            # Kinetic energy: 0.5 * m * v^2
            energy += 0.5 * (mass_vec * vel_squared_vec).reduce_add()

        vectorize[accumulate_kinetic_energy, NELTS](self.N)

        for i in range(self.N):
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]
            mass_i = self.mass[i]
            
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
                energy -= G * mass_i * (mass_j_vec / distance_vec).reduce_add()
                
            vectorize[accumulate_potential_energy, NELTS](self.N - (i + 1))
            
        return energy


    @always_inline
    fn reset_acceleration(self):
        memset_zero(self.acc_x, self.N)
        memset_zero(self.acc_y, self.N)
        memset_zero(self.acc_z, self.N)
