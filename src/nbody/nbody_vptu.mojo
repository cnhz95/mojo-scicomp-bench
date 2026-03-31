from memory import memset_zero
from algorithm import parallel_memcpy
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from os.atomic import Atomic
from math import sqrt, iota
from nbody_trait import NBody

comptime G = 1.0
comptime DT = 0.001
comptime SOFTENING = 0.001
comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2
comptime TILE_SIZE = 64
comptime UNROLL_FACTOR = 4

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
        @parameter
        fn compute_acceleration_for_tile(tile_idx: Int):
            i_start = tile_idx * TILE_SIZE
            i_end = min(i_start + TILE_SIZE, self.N)
            num_j_tiles = (self.N + TILE_SIZE - 1) // TILE_SIZE
    
            for j_tile in range(num_j_tiles):
                j_start = j_tile * TILE_SIZE
                j_end = min(j_start + TILE_SIZE, self.N)

                for i in range(i_start, i_end):
                    pos_x_i = self.pos_x[i]
                    pos_y_i = self.pos_y[i]
                    pos_z_i = self.pos_z[i]
                    acc_x_local = 0.0
                    acc_y_local = 0.0
                    acc_z_local = 0.0

                    @parameter
                    fn accumulate_range_for_i(j_begin: Int, j_finish: Int):
                        @parameter
                        fn accumulate_pairwise_acceleration[width: Int](offset: Int):
                            j = j_begin + offset
                            dx_vec = self.pos_x.load[width=width](j) - pos_x_i  # Broadcast scalar value
                            dy_vec = self.pos_y.load[width=width](j) - pos_y_i
                            dz_vec = self.pos_z.load[width=width](j) - pos_z_i
                            mass_j_vec = self.mass.load[width=width](j)
                            
                            distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                            distance_vec = sqrt(distance_squared_vec)
                            force_vec = G * mass_j_vec / (distance_vec * distance_squared_vec)
                            
                            acc_x_local += (force_vec * dx_vec).reduce_add()
                            acc_y_local += (force_vec * dy_vec).reduce_add()
                            acc_z_local += (force_vec * dz_vec).reduce_add()

                        range_width = j_finish - j_begin
                        vectorize[accumulate_pairwise_acceleration, NELTS, unroll_factor=UNROLL_FACTOR](range_width)

                    if i < j_start or i >= j_end:
                        # Tile contains i
                        accumulate_range_for_i(j_start, j_end)
                    else:
                        # Avoid self-interaction
                        accumulate_range_for_i(j_start, i)
                        accumulate_range_for_i(i + 1, j_end)

                    self.acc_x[i] += acc_x_local
                    self.acc_y[i] += acc_y_local
                    self.acc_z[i] += acc_z_local

        num_i_tiles = (self.N + TILE_SIZE - 1) // TILE_SIZE
        parallelize[compute_acceleration_for_tile](num_i_tiles)


    @always_inline
    fn advance(self):
        """Perform one leapfrog integration step, updating positions and velocities for all particles."""
        comptime HALF_DT = 0.5 * DT

        @parameter
        fn kick_step[width: Int](offset: Int):
            self.vel_x.store[width=width](offset, self.vel_x.load[width=width](offset) + self.acc_x.load[width=width](offset) * HALF_DT)
            self.vel_y.store[width=width](offset, self.vel_y.load[width=width](offset) + self.acc_y.load[width=width](offset) * HALF_DT)
            self.vel_z.store[width=width](offset, self.vel_z.load[width=width](offset) + self.acc_z.load[width=width](offset) * HALF_DT)

        vectorize[kick_step, NELTS, unroll_factor=UNROLL_FACTOR](self.N)
        
        @parameter
        fn drift_step[width: Int](offset: Int):
            self.pos_x.store[width=width](offset, self.pos_x.load[width=width](offset) + self.vel_x.load[width=width](offset) * DT)
            self.pos_y.store[width=width](offset, self.pos_y.load[width=width](offset) + self.vel_y.load[width=width](offset) * DT)
            self.pos_z.store[width=width](offset, self.pos_z.load[width=width](offset) + self.vel_z.load[width=width](offset) * DT)

        vectorize[drift_step, NELTS, unroll_factor=UNROLL_FACTOR](self.N)

        self.reset_acceleration()
        self.compute_acceleration()

        vectorize[kick_step, NELTS, unroll_factor=UNROLL_FACTOR](self.N)
        

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

        vectorize[accumulate_kinetic_energy, NELTS, unroll_factor=UNROLL_FACTOR](self.N)
        
        @parameter
        fn accumulate_potential_for_tile(tile_idx: Int):
            i_start = tile_idx * TILE_SIZE
            i_end = min(i_start + TILE_SIZE, self.N)
            num_j_tiles = (self.N + TILE_SIZE - 1) // TILE_SIZE
            tile_potential_energy = 0.0

            for j_tile in range(tile_idx, num_j_tiles):
                j_start = j_tile * TILE_SIZE
                j_end = min(j_start + TILE_SIZE, self.N)

                for i in range(i_start, i_end):
                    pos_x_i = self.pos_x[i]
                    pos_y_i = self.pos_y[i]
                    pos_z_i = self.pos_z[i]
                    mass_i = self.mass[i]
                    j_begin = i + 1 if j_tile == tile_idx else j_start
                    
                    @parameter
                    fn accumulate_potential_energy[width: Int](offset: Int):
                        j = j_begin + offset  # Avoid double counting
                        dx_vec = self.pos_x.load[width=width](j) - pos_x_i
                        dy_vec = self.pos_y.load[width=width](j) - pos_y_i
                        dz_vec = self.pos_z.load[width=width](j) - pos_z_i
                        mass_j_vec = self.mass.load[width=width](j)

                        distance_squared_vec = dx_vec * dx_vec + dy_vec * dy_vec + dz_vec * dz_vec + SOFTENING * SOFTENING
                        distance_vec = sqrt(distance_squared_vec)
                        
                        # Potential energy: -G * m1 * m2 / r
                        tile_potential_energy -= G * mass_i * (mass_j_vec / distance_vec).reduce_add()
                        
                    range_width = j_end - j_begin
                    vectorize[accumulate_potential_energy, NELTS, unroll_factor=UNROLL_FACTOR](range_width)

            _ = potential_energy.fetch_add(tile_potential_energy)

        num_i_tiles = (self.N + TILE_SIZE - 1) // TILE_SIZE
        parallelize[accumulate_potential_for_tile](num_i_tiles)

        return kinetic_energy + potential_energy.load()


    @always_inline
    fn reset_acceleration(self):
        memset_zero(self.acc_x, self.N)
        memset_zero(self.acc_y, self.N)
        memset_zero(self.acc_z, self.N)
