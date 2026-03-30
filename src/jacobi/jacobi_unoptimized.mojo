from memory import memset_zero
from algorithm import parallel_memcpy
from math import sqrt
from jacobi_trait import Jacobi

comptime DX = 1.0
comptime DY = 1.0
comptime R_X = 1.0 / (DX * DX)
comptime R_Y = 1.0 / (DY * DY)
comptime CENTER_COEFF = 2.0 * (R_X + R_Y)
comptime INITIAL_TEMP = 20.0
comptime DTYPE = DType.float64

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
        return i * self.NY + j  # Row stride = NY


    @always_inline
    fn initialize_temperature(self):
        """Initialize temperature field."""
        for i in range(self.NX * self.NY):
            self.T_curr[i] = INITIAL_TEMP
        
        memset_zero(self.T_next, self.NX * self.NY)


    @always_inline
    fn apply_boundary_conditions(self, T: UnsafePointer[mut=True, Scalar[DTYPE], MutOrigin.external]):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        for j in range(self.NY):
            T[self.idx(0, j)] = 100.0           # Top
            T[self.idx(self.NX - 1, j)] = 0.0   # Bottom
            
        for i in range(self.NX):
            T[self.idx(i, 0)] = 0.0             # Left
            T[self.idx(i, self.NY - 1)] = 50.0  # Right


    @always_inline
    fn jacobi_update(self):
        """Core Jacobi iteration kernel."""
        # Update interior points
        for i in range(1, self.NX - 1):
            for j in range(1, self.NY - 1):
                self.T_next[self.idx(i, j)] = (
                    R_X * (self.T_curr[self.idx(i + 1, j)] + self.T_curr[self.idx(i - 1, j)]) +
                    R_Y * (self.T_curr[self.idx(i, j + 1)] + self.T_curr[self.idx(i, j - 1)])
                ) / CENTER_COEFF


    @always_inline
    fn residual_norm(self) -> Float64:
        """Compute relative residual norm."""
        res_squared = 0.0
        norm_squared = 0.0

        # Compute residual over interior points
        for i in range(1, self.NX - 1):
            for j in range(1, self.NY - 1):
                center = self.T_next[self.idx(i, j)]
                Ax = CENTER_COEFF * center - (
                    R_X * (self.T_next[self.idx(i + 1, j)] + self.T_next[self.idx(i - 1,j)]) +
                    R_Y * (self.T_next[self.idx(i, j + 1)] + self.T_next[self.idx(i, j - 1)]))

                res_squared += Ax * Ax
                norm_squared += center * center
                
        return sqrt(res_squared / norm_squared) if norm_squared > 0 else sqrt(res_squared)


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
        