from memory import memset_zero, memcpy
from algorithm.functional import vectorize, parallelize
from sys.info import simd_width_of
from os.atomic import Atomic
from math import sqrt
from time import perf_counter

alias NX = 1 << 10
alias NY = 1 << 10
alias DX = 1.0
alias DY = 1.0
alias R_X = 1.0 / (DX * DX)
alias R_Y = 1.0 / (DY * DY)
alias CENTER_COEFF = 2.0 * (R_X + R_Y)
alias SIZE = NX * NY
alias TOL = 1e-8
alias MAX_ITER = 500
alias INITIAL_TEMP = 20.0
alias DTYPE = DType.float64
alias NELTS = simd_width_of[DTYPE]() * 2
alias UNROLL_FACTOR = 4

struct Heat2DJacobi(ImplicitlyCopyable, Movable):
    var T_curr: UnsafePointer[Scalar[DTYPE]]
    var T_next: UnsafePointer[Scalar[DTYPE]]

    fn __init__(out self):
        self.T_curr = UnsafePointer[Scalar[DTYPE]].alloc(SIZE)
        self.T_next = UnsafePointer[Scalar[DTYPE]].alloc(SIZE)
        self.initialize_temperature()
        self.apply_boundary_conditions(self.T_curr)


    fn __del__(deinit self):
        self.T_curr.free()
        self.T_next.free()


    @always_inline
    fn idx(self, i: Int, j: Int) -> Int:
        """Convert 2D grid coordinates to 1D array index."""
        return i * NY + j  # Row stride = NY


    @always_inline
    fn initialize_temperature(self):
        """Initialize temperature field."""
        @parameter
        fn initial_temperature[nelts: Int](offset: Int):
            self.T_curr.store[width=nelts](offset, INITIAL_TEMP)

        vectorize[initial_temperature, NELTS, unroll_factor=UNROLL_FACTOR](SIZE)
        memset_zero(self.T_next, SIZE)


    @always_inline
    fn apply_boundary_conditions(self, T: UnsafePointer[Scalar[DTYPE]]):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        @parameter
        fn apply_top_bottom[nelts: Int](offset: Int):
            T.store[width=nelts](self.idx(0, offset), 100.0)    # Top
            T.store[width=nelts](self.idx(NX - 1, offset), 0.0) # Bottom
            
        vectorize[apply_top_bottom, NELTS, unroll_factor=UNROLL_FACTOR](NY)

        T.strided_store[width=NELTS](0.0, NY)                   # Left
        T.offset(NY - 1).strided_store[width=NELTS](50.0, NY)   # Right


    @always_inline
    fn jacobi_update(self):
        """Jacobi iteration kernel."""
        @parameter
        fn process_row(t: Int):
            i = t + 1
            @parameter
            fn process_cols[nelts: Int](offset: Int):
                j = offset + 1
                self.T_next.store[width=nelts](self.idx(i, j), (
                    R_X * (self.T_curr.load[width=nelts](self.idx(i + 1, j)) +
                           self.T_curr.load[width=nelts](self.idx(i - 1, j))) + 
                    R_Y * (self.T_curr.load[width=nelts](self.idx(i, j + 1)) +
                           self.T_curr.load[width=nelts](self.idx(i, j - 1))))
                    / CENTER_COEFF
                )

            vectorize[process_cols, NELTS, unroll_factor=UNROLL_FACTOR](NY - 2)  # NY - 2 interior columns to process per row
        parallelize[process_row](NX - 2)  # Spawn NX - 2 tasks, each computing one interior row


    @always_inline
    fn residual_norm(self) -> Float64:
        """Compute the relative residual norm."""
        res_acc = Atomic[DTYPE](0.0)
        norm_acc = Atomic[DTYPE](0.0)

        @parameter
        fn process_row(thread_id: Int):
            i = thread_id + 1
            local_res_sum = 0.0
            local_norm_sum = 0.0

            @parameter
            fn compute_residual[nelts: Int](offset: Int):
                j = offset + 1
                center = self.T_next.load[width=nelts](self.idx(i , j))
                Ax = CENTER_COEFF * center - (
                    R_X * (self.T_next.load[width=nelts](self.idx(i + 1, j)) +
                           self.T_next.load[width=nelts](self.idx(i - 1, j))) +
                    R_Y * (self.T_next.load[width=nelts](self.idx(i, j + 1)) +
                           self.T_next.load[width=nelts](self.idx(i, j - 1)))
                )
                local_res_sum += (Ax * Ax).reduce_add()
                local_norm_sum += (center * center).reduce_add()

            vectorize[compute_residual, NELTS, unroll_factor=UNROLL_FACTOR](NY - 2)
            _ = res_acc.fetch_add(local_res_sum)
            _ = norm_acc.fetch_add(local_norm_sum)

        parallelize[process_row](NX - 2)

        res_squared = res_acc.load()
        norm_squared = norm_acc.load()
                
        return sqrt(res_squared / norm_squared) if norm_squared > 0 else sqrt(res_squared)


    @always_inline
    fn solve(var self) -> Int:
        """Solve the 2D heat equation using the Jacobi iterative method."""
        memcpy(self.T_next, self.T_curr, SIZE)  # Initial guess: T_next = T_curr

        for iter in range(MAX_ITER):
            self.jacobi_update()
            self.apply_boundary_conditions(self.T_next)
            res = self.residual_norm()

            if res < TOL:
                swap(self.T_curr, self.T_next)
                return iter + 1

            # Continue iterating
            swap(self.T_curr, self.T_next)

        # Did not converge
        return MAX_ITER


    @always_inline
    fn print_grid(self):
        """Print the temperature grid."""
        for i in range(NX):
            for j in range(NY):
                end = "\t" if j < NY - 1 else "\n"
                print(round(self.T_curr[self.idx(i, j)], 4), end=end)
        

fn main() raises:
    solver = Heat2DJacobi()

    start_time = perf_counter()
    iters = solver.solve()
    end_time = perf_counter()

    # solver.print_grid()
    print("Execution time:\t\t", end_time - start_time, "s")
    print("Number of iterations:\t", iters)
    print("Final residual norm:\t", solver.residual_norm())
