from memory import memset_zero
from algorithm import parallel_memcpy
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from os.atomic import Atomic
from math import sqrt
from jacobi_trait import Jacobi

comptime DX = 1.0
comptime DY = 1.0
comptime R_X = 1.0 / (DX * DX)
comptime R_Y = 1.0 / (DY * DY)
comptime CENTER_COEFF = 2.0 * (R_X + R_Y)
comptime INITIAL_TEMP = 20.0
comptime DTYPE = DType.float64
comptime NELTS = simd_width_of[DTYPE]() * 2
comptime TILE_SIZE = 32
comptime UNROLL_FACTOR = 4

struct Heat2DJacobi(Jacobi, ImplicitlyCopyable):
    var NX: Int
    var NY: Int
    var MAX_ITER: Int
    var T_curr: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]
    var T_next: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]


    fn __init__(out self, NX: Int, NY: Int, MAX_ITER: Int):
        self.NX = NX
        self.NY = NY
        self.MAX_ITER = MAX_ITER
        self.T_curr = alloc[Scalar[DTYPE]](self.NX * self.NY)
        self.T_next = alloc[Scalar[DTYPE]](self.NX * self.NY)
        self.initialize_temperature()
        self.apply_boundary_conditions(self.T_curr)


    fn __getitem__(self, i: Int, j: Int) -> Float64:
        return self.T_curr[self.idx(i, j)]


    fn __del__(deinit self):
        self.T_curr.free()
        self.T_next.free()


    @always_inline
    fn idx(self, i: Int, j: Int) -> Int:
        """Convert 2D grid coordinates to 1D array index."""
        return i * self.NY + j


    @always_inline
    fn initialize_temperature(self):
        """Initialize temperature field."""
        @parameter
        fn initial_temperature[width: Int](offset: Int):
            self.T_curr.store[width=width](offset, INITIAL_TEMP)

        vectorize[initial_temperature, NELTS, unroll_factor=UNROLL_FACTOR](self.NX * self.NY)
        memset_zero(self.T_next, self.NX * self.NY)


    @always_inline
    fn apply_boundary_conditions(self, T: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        @parameter
        fn apply_top_bottom[width: Int](offset: Int):
            T.store[width=width](self.idx(0, offset), 100.0)                                    # Top
            T.store[width=width](self.idx(self.NX - 1, offset), 0.0)                            # Bottom
            
        vectorize[apply_top_bottom, NELTS, unroll_factor=UNROLL_FACTOR](self.NY)

        @parameter
        fn apply_left_right[width: Int](offset: Int):
            T.offset(self.idx(offset, 0)).strided_store[width=width](0.0, self.NY)              # Left
            T.offset(self.idx(offset, self.NY - 1)).strided_store[width=width](50.0, self.NY)   # Right
        
        vectorize[apply_left_right, NELTS, unroll_factor=UNROLL_FACTOR](self.NX)


    @always_inline
    fn jacobi_update(self):
        num_i_tiles = (self.NX - 2 + TILE_SIZE - 1) // TILE_SIZE
        num_j_tiles = (self.NY - 2 + TILE_SIZE - 1) // TILE_SIZE

        @parameter
        fn update_row(tile_idx: Int):
            tile_i = tile_idx // num_j_tiles
            tile_j = tile_idx % num_j_tiles

            i_start = tile_i * TILE_SIZE + 1
            i_end = min(i_start + TILE_SIZE, self.NX - 1)
            j_start = tile_j * TILE_SIZE + 1
            j_end = min(j_start + TILE_SIZE, self.NY - 1)

            # Process all rows in this tile
            for i in range(i_start, i_end):
                # Vectorize columns within this tile
                @parameter
                fn update_row_segment[width: Int](offset: Int):
                    j = j_start + offset
                    self.T_next.store[width=width](self.idx(i, j), (
                        R_X * (self.T_curr.load[width=width](self.idx(i + 1, j)) +
                               self.T_curr.load[width=width](self.idx(i - 1, j))) +
                        R_Y * (self.T_curr.load[width=width](self.idx(i, j + 1)) +
                               self.T_curr.load[width=width](self.idx(i, j - 1)))
                        ) / CENTER_COEFF
                    )
                
                tile_width = j_end - j_start
                vectorize[update_row_segment, NELTS, unroll_factor=UNROLL_FACTOR](tile_width)

        total_tiles = num_i_tiles * num_j_tiles
        parallelize[update_row](total_tiles)


    @always_inline
    fn residual_norm(self) -> Float64:
        """Compute the relative residual norm."""
        res_squared = Atomic[DTYPE](0.0)
        norm_squared = Atomic[DTYPE](0.0)

        num_i_tiles = (self.NX - 2 + TILE_SIZE - 1) // TILE_SIZE
        num_j_tiles = (self.NY - 2 + TILE_SIZE - 1) // TILE_SIZE

        @parameter
        fn accumulate_tile_residual(tile_idx: Int):
            tile_i = tile_idx // num_j_tiles
            tile_j = tile_idx % num_j_tiles

            i_start = tile_i * TILE_SIZE + 1
            i_end = min(i_start + TILE_SIZE, self.NX - 1)
            j_start = tile_j * TILE_SIZE + 1
            j_end = min(j_start + TILE_SIZE, self.NY - 1)

            local_res_sum = 0.0
            local_norm_sum = 0.0

            # Process all rows in this tile
            for i in range(i_start, i_end):
                # Vectorize columns within this tile
                @parameter
                fn compute_residual_segment[width: Int](offset: Int):
                    j = j_start + offset
                    center = self.T_next.load[width=width](self.idx(i , j))
                    Ax = CENTER_COEFF * center - (
                        R_X * (self.T_next.load[width=width](self.idx(i + 1, j)) +
                               self.T_next.load[width=width](self.idx(i - 1, j))) +
                        R_Y * (self.T_next.load[width=width](self.idx(i, j + 1)) +
                               self.T_next.load[width=width](self.idx(i, j - 1)))
                    )

                    local_res_sum += (Ax * Ax).reduce_add()
                    local_norm_sum += (center * center).reduce_add()

                tile_width = j_end - j_start
                vectorize[compute_residual_segment, NELTS, unroll_factor=UNROLL_FACTOR](tile_width)

            _ = res_squared.fetch_add(local_res_sum)
            _ = norm_squared.fetch_add(local_norm_sum)

        total_tiles = num_i_tiles * num_j_tiles
        parallelize[accumulate_tile_residual](total_tiles)

        return sqrt(res_squared.load() / norm_squared.load()) if norm_squared.load() > 0 else sqrt(res_squared.load())
        

    @always_inline
    fn solve(var self) -> Int:
        """Solve the 2D heat equation using the Jacobi iterative method."""
        parallel_memcpy(dest=self.T_next, src=self.T_curr, count=self.NX * self.NY)  # Initial guess: T_next = T_curr

        # Jacobi iteration loop
        for iter in range(self.MAX_ITER):
            self.jacobi_update()
            self.apply_boundary_conditions(self.T_next)
            res = self.residual_norm()

            if res < 1e-8:
                swap(self.T_curr, self.T_next)
                return iter + 1

            # Continue iterating
            swap(self.T_curr, self.T_next)

        # Did not converge
        return self.MAX_ITER
