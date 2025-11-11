from memory import memset_zero, memcpy
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
        for i in range(SIZE):
            self.T_curr[i] = INITIAL_TEMP
        
        memset_zero(self.T_next, SIZE)


    @always_inline
    fn apply_boundary_conditions(self, T: UnsafePointer[Scalar[DTYPE]]):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        for j in range(NY):
            T[self.idx(0, j)] = 100.0       # Top
            T[self.idx(NX - 1, j)] = 0.0    # Bottom
            
        for i in range(NX):
            T[self.idx(i, 0)] = 0.0         # Left
            T[self.idx(i, NY - 1)] = 50.0   # Right


    @always_inline
    fn jacobi_update(self):
        for i in range(1, NX - 1):
            for j in range(1, NY - 1):
                self.T_next[self.idx(i, j)] = (
                    R_X * (self.T_curr[self.idx(i + 1, j)] + self.T_curr[self.idx(i - 1, j)]) +
                    R_Y * (self.T_curr[self.idx(i, j + 1)] + self.T_curr[self.idx(i, j - 1)])
                ) / CENTER_COEFF


    @always_inline
    fn residual_norm(self) -> Float64:
        res2 = 0.0
        norm2 = 0.0

        for i in range(1, NX - 1):
            for j in range(1, NY - 1):
                center = self.T_next[self.idx(i, j)]
                Ax = CENTER_COEFF * center - (
                    R_X * (self.T_next[self.idx(i + 1, j)] + self.T_next[self.idx(i - 1,j)]) +
                    R_Y * (self.T_next[self.idx(i, j + 1)] + self.T_next[self.idx(i, j - 1)]))

                res2 += Ax * Ax
                norm2 += center * center
                
        return sqrt(res2 / norm2) if norm2 > 0 else sqrt(res2)


    @always_inline
    fn solve(var self) -> Int:
        memcpy(self.T_next, self.T_curr, SIZE)  # Initial guess: T_next = T_curr

        for iter in range(MAX_ITER):
            self.jacobi_update()
            self.apply_boundary_conditions(self.T_next)
            res = self.residual_norm()

            if res < TOL:
                swap(self.T_curr, self.T_next)
                return iter + 1

            swap(self.T_curr, self.T_next)

        # Did not converge
        return MAX_ITER


    @always_inline
    fn print_grid(self):
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
