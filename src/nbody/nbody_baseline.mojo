from memory import memset_zero, memcpy
from math import sqrt
from nbody_trait import NBody

alias N = 1 << 10
alias G = 1.0
alias DT = 0.01
alias SOFTENING = 0.001
alias DTYPE = DType.float64

struct NBodySystem(NBody):
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


    fn __init__(out self, pos: UnsafePointer[Scalar[DTYPE]], vel: UnsafePointer[Scalar[DTYPE]], mass: UnsafePointer[Scalar[DTYPE]]):
        self.pos_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.pos_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.pos_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.vel_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_x = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_y = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.acc_z = UnsafePointer[Scalar[DTYPE]].alloc(N)
        self.mass = UnsafePointer[Scalar[DTYPE]].alloc(N)

        memcpy(self.pos_x, pos, N)
        memcpy(self.pos_y, pos + N, N)
        memcpy(self.pos_z, pos + 2 * N, N)
        memcpy(self.vel_x, vel, N)
        memcpy(self.vel_y, vel + N, N)
        memcpy(self.vel_z, vel + 2 * N, N)
        memcpy(self.mass, mass, N)
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
        for i in range(N):
            pos_x_i = self.pos_x[i]
            pos_y_i = self.pos_y[i]
            pos_z_i = self.pos_z[i]

            for j in range(N):
                if i == j:  # Skip self-interaction
                    continue
                dx = self.pos_x[j] - pos_x_i
                dy = self.pos_y[j] - pos_y_i
                dz = self.pos_z[j] - pos_z_i
                distance_squared = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING
                distance = sqrt(distance_squared)
                force = G * self.mass[j] / (distance * distance_squared)
                
                self.acc_x[i] += force * dx
                self.acc_y[i] += force * dy
                self.acc_z[i] += force * dz


    @always_inline
    fn advance(self):
        """Perform one leapfrog integration step, updating positions and velocities for all particles."""
        alias HALF_DT = 0.5 * DT

        @parameter
        fn kick_step():
            for i in range(N):
                self.vel_x[i] += self.acc_x[i] * HALF_DT
                self.vel_y[i] += self.acc_y[i] * HALF_DT
                self.vel_z[i] += self.acc_z[i] * HALF_DT

        @parameter
        fn drift_step():
            for i in range(N):
                self.pos_x[i] += self.vel_x[i] * DT
                self.pos_y[i] += self.vel_y[i] * DT
                self.pos_z[i] += self.vel_z[i] * DT

        kick_step()
        drift_step()
        self.reset_acceleration()
        self.compute_acceleration()
        kick_step()


    @always_inline
    fn compute_energy(self) -> Float64:
        """Compute total system energy (kinetic + potential)."""
        energy = 0.0

        for i in range(N):
            vel_x_i = self.vel_x[i]
            vel_y_i = self.vel_y[i]
            vel_z_i = self.vel_z[i]
            mass_i = self.mass[i]

            # Kinetic energy: 0.5 * m * v^2
            energy += 0.5 * mass_i * (vel_x_i * vel_x_i + vel_y_i * vel_y_i + vel_z_i * vel_z_i)

            for j in range(i + 1, N):
                dx = self.pos_x[j] - self.pos_x[i]
                dy = self.pos_y[j] - self.pos_y[i]
                dz = self.pos_z[j] - self.pos_z[i]
                distance_squared = dx * dx + dy * dy + dz * dz + SOFTENING * SOFTENING
                distance = sqrt(distance_squared)

                # Potential energy: -G * m1 * m2 / r
                energy -= G * mass_i * self.mass[j] / distance

        return energy


    @always_inline
    fn reset_acceleration(self):
        memset_zero(self.acc_x, N)
        memset_zero(self.acc_y, N)
        memset_zero(self.acc_z, N)
