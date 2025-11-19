from memory import memset_zero, memcpy
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from os.atomic import Atomic
from math import sqrt
from jacobi_trait import Jacobi

alias DX = 1.0
alias DY = 1.0
alias R_X = 1.0 / (DX * DX)
alias R_Y = 1.0 / (DY * DY)
alias CENTER_COEFF = 2.0 * (R_X + R_Y)
alias INITIAL_TEMP = 20.0
alias MAX_ITER = 20000
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2

struct Heat2DJacobi(Jacobi, ImplicitlyCopyable):
    var NX: Int
    var NY: Int
    var T_curr: UnsafePointer[Scalar[DTYPE]]
    var T_next: UnsafePointer[Scalar[DTYPE]]


    fn __init__(out self, NX: Int, NY: Int):
        self.NX = NX
        self.NY = NY
        self.T_curr = UnsafePointer[Scalar[DTYPE]].alloc(self.NX * self.NY)
        self.T_next = UnsafePointer[Scalar[DTYPE]].alloc(self.NX * self.NY)
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
        fn init_temp[width: Int](offset: Int):
            self.T_curr.store[width=width](offset, INITIAL_TEMP)

        vectorize[init_temp, NELTS](self.NX * self.NY)
        memset_zero(self.T_next, self.NX * self.NY)


    @always_inline
    fn apply_boundary_conditions(self, T: UnsafePointer[Scalar[DTYPE]]):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        @parameter
        fn apply_top_bottom[width: Int](offset: Int):
            T.store[width=width](self.idx(0, offset), 100.0)                                    # Top
            T.store[width=width](self.idx(self.NX - 1, offset), 0.0)                            # Bottom
            
        vectorize[apply_top_bottom, NELTS](self.NY)

        @parameter
        fn apply_left_right[width: Int](offset: Int):
            T.offset(self.idx(offset, 0)).strided_store[width=width](0.0, self.NY)              # Left
            T.offset(self.idx(offset, self.NY - 1)).strided_store[width=width](50.0, self.NY)   # Right
        
        vectorize[apply_left_right, NELTS](self.NX)


    @always_inline
    fn jacobi_update(self):
        """Core Jacobi iteration kernel."""
        @parameter
        fn update_row(t: Int):
            i = t + 1
            @parameter
            fn update_row_segment[width: Int](offset: Int): 
                j = offset + 1  # Interior columns (j = 1..NY-2)
                self.T_next.store[width=width](self.idx(i, j), (
                    R_X * (self.T_curr.load[width=width](self.idx(i + 1, j)) +
                           self.T_curr.load[width=width](self.idx(i - 1, j))) +
                    R_Y * (self.T_curr.load[width=width](self.idx(i, j + 1)) +
                           self.T_curr.load[width=width](self.idx(i, j - 1)))
                    ) / CENTER_COEFF
                )

            vectorize[update_row_segment, NELTS](self.NY - 2)  # NY-2 interior columns to process per row
        parallelize[update_row](self.NX - 2)  # Spawn NX-2 tasks, each computing one interior row
        
        
    @always_inline
    fn residual_norm(self) -> Float64:
        """Compute the relative residual norm."""
        res_squared = Atomic[DTYPE](0.0)
        norm_squared = Atomic[DTYPE](0.0)

        @parameter
        fn accumulate_residual_row(t: Int):
            i = t + 1
            local_res_sum = 0.0
            local_norm_sum = 0.0

            @parameter
            fn compute_residual_segment[width: Int](offset: Int):  # Row segment (j = 1..NY-2)
                j = offset + 1
                center = self.T_next.load[width=width](self.idx(i , j))
                Ax = CENTER_COEFF * center - (
                    R_X * (self.T_next.load[width=width](self.idx(i + 1, j)) +
                           self.T_next.load[width=width](self.idx(i - 1, j))) +
                    R_Y * (self.T_next.load[width=width](self.idx(i, j + 1)) +
                           self.T_next.load[width=width](self.idx(i, j - 1)))
                )

                local_res_sum += (Ax * Ax).reduce_add()
                local_norm_sum += (center * center).reduce_add()

            vectorize[compute_residual_segment, NELTS](self.NY - 2)

            _ = res_squared.fetch_add(local_res_sum)
            _ = norm_squared.fetch_add(local_norm_sum)

        parallelize[accumulate_residual_row](self.NX - 2)

        return sqrt(res_squared.load() / norm_squared.load()) if norm_squared.load() > 0 else sqrt(res_squared.load())


    @always_inline
    fn solve(var self) -> Int:
        """Solve the 2D heat equation using the Jacobi iterative method."""
        memcpy(self.T_next, self.T_curr, self.NX * self.NY)  # Initial guess: T_next = T_curr

        # Jacobi iteration loop
        for iter in range(MAX_ITER):
            self.jacobi_update()
            self.apply_boundary_conditions(self.T_next)
            res = self.residual_norm()

            if res < 1e-8:
                swap(self.T_curr, self.T_next)
                return iter + 1

            # Continue iterating
            swap(self.T_curr, self.T_next)

        # Did not converge
        return MAX_ITER
