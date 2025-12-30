import numpy as np

G = 1.0
DT = 0.001
SOFTENING = 0.001

class NBodySystem:
    def __init__(self, N, pos, vel, mass):
        self.N = N
        self.pos = pos
        self.vel = vel
        self.acc = np.zeros((N, 3))
        self.mass = mass


    def compute_acceleration(self):
        """Computation gravitational acceleration."""
        self.acc[:] = 0.0  # Reset acceleration
        
        # Compute pairwise differences
        dx = self.pos[None, :, :] - self.pos[:, None, :]

        distance_squared = np.sum(dx**2, axis=2) + SOFTENING * SOFTENING
        distance = np.sqrt(distance_squared)
        force = G * self.mass[None, :] / (distance * distance_squared)

        # Zero out diagonal to avoid self-interaction
        np.fill_diagonal(force, 0.0)

        # Compute acceleration for each body
        self.acc = np.sum(force[:, :, None] * dx, axis=1)


    def advance(self):
        """Perform one leapfrog integration step, updating positions and velocities for all particles."""
        HALF_DT = 0.5 * DT
        
        # Kick step
        self.vel += self.acc * HALF_DT
        
        # Drift step
        self.pos += self.vel * DT
        
        # Recompute acceleration at new positions
        self.compute_acceleration()
        
        # Kick step
        self.vel += self.acc * HALF_DT
    

    def compute_energy(self):
        """Compute total energy (kinetic + potential)."""
        velocity_squared = np.sum(self.vel**2, axis=1)

        # Kinetic energy: 0.5 * m * v^2
        kinetic = 0.5 * np.sum(self.mass * velocity_squared)
        
        # Compute pairwise differences
        dx = self.pos[None, :, :] - self.pos[:, None, :]

        distance_squared = np.sum(dx**2, axis=2) + SOFTENING * SOFTENING
        distance = np.sqrt(distance_squared)
        
        # Upper triangular mask to avoid double-counting
        i, j = np.triu_indices(self.N, k=1)

        # Potential energy: -G * m1 * m2 / r
        potential = -G * np.sum(self.mass[i] * self.mass[j] / distance[i, j])
        
        return kinetic + potential
 