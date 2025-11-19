import numpy as np

DX = 1.0
DY = 1.0
R_X = 1.0 / (DX * DX)
R_Y = 1.0 / (DY * DY)
CENTER_COEFF = 2.0 * (R_X + R_Y)
INITIAL_TEMP = 20.0
MAX_ITER = 20000

class Heat2DJacobi:
    def __init__(self, NX, NY):
        self.NX = NX
        self.NY = NY
        self.T_curr = np.full((self.NX, self.NY), INITIAL_TEMP)
        self.T_next = np.zeros((self.NX, self.NY))
        self.apply_boundary_conditions(self.T_curr)


    def apply_boundary_conditions(self, T):
        """Apply Dirichlet boundary conditions to all four edges of the grid."""
        T[:, 0] = 0.0   # Left
        T[:, -1] = 50.0 # right
        T[0, :] = 100   # Top
        T[-1, :] = 0.0  # Bottom


    def jacobi_iteration(self):
        """Core Jacobi iteration kernel."""
        # Update interior points
        self.T_next[1:-1, 1:-1] = (
            R_X * (self.T_curr[2:, 1:-1] + self.T_curr[:-2, 1:-1]) +
            R_Y * (self.T_curr[1:-1, 2:] + self.T_curr[1:-1, :-2])
        ) / CENTER_COEFF


    def residual_norm(self):
        """Compute the relative residual norm."""
        interior = self.T_next[1:-1, 1:-1]
        
        Ax = CENTER_COEFF * interior - (
            R_X * (self.T_next[2:, 1:-1] + self.T_next[:-2, 1:-1]) +
            R_Y * (self.T_next[1:-1, 2:] + self.T_next[1:-1, :-2])
        )
        
        res_squared = np.sum(Ax * Ax)
        norm_squared = np.sum(interior * interior)
        
        return np.sqrt(res_squared / norm_squared) if norm_squared > 0 else np.sqrt(res_squared)


    def solve(self):
        """Solve the 2D heat equation using the Jacobi iterative method."""
        self.T_next[:] = self.T_curr[:]  # Initial guess: T_next = T_curr

        # Jacobi iteration loop
        for iter in range(MAX_ITER):
            self.jacobi_iteration()
            self.apply_boundary_conditions(self.T_next)
            res = self.residual_norm()

            if res < 1e-8:
                self.T_curr, self.T_next = self.T_next, self.T_curr
                return iter + 1

            # Continue iterating
            self.T_curr, self.T_next = self.T_next, self.T_curr

        # Did not converge
        return MAX_ITER
    

    def verify_solution(self):
        # Check the boundary conditions
        top = np.allclose(self.T_curr[0, :], 100.0)
        bottom = np.allclose(self.T_curr[-1, :], 0.0)
        left = np.allclose(self.T_curr[1:-1, 0], 0.0)
        right = np.allclose(self.T_curr[1:-1, -1], 50.0)

        # Check that temperature is within physical bounds
        temp_range = np.all((self.T_curr >= 0.0) & (self.T_curr <= 100.0))

        # Check that interior changed from initial conditions
        not_initial_temp = not np.allclose(self.T_curr[1:-1, 1:-1], INITIAL_TEMP)

        # Check that average temperature descreases per row
        temp_decreases = np.all(np.diff(np.mean(self.T_curr, axis=1)) <= 1e-6)

        return top and bottom and left and right and temp_range and not_initial_temp and temp_decreases

    
    def get_grid(self):
        return self.T_curr
        