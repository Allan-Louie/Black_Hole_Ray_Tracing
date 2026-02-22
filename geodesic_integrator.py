import numpy as np

class SchwarzschildGeodesic:
    def __init__(self, M):
        self.M = M

    def derivatives(self, u):
        """Return du/dlambda for geodesic equation - handles full 9-element state"""
        t, r, theta, phi, dt, dr, dtheta, dphi, affine = u
        M = self.M

        # Avoid division by zero
        if r <= 2*M + 1e-3:
            return np.zeros(9)  # Return 9 zeros to match state vector shape

        f = 1 - 2*M/r
        f_r = 2*M/(r**2)

        # Compute second derivatives (accelerations)
        d2t = - (f_r/f) * dt * dr
        d2r = - (M*f/r**2) * dt**2 + (M/(r**2*f)) * dr**2 + r * \
            f * dtheta**2 + r*f * np.sin(theta)**2 * dphi**2
        d2theta = - (2/r) * dr * dtheta + np.sin(theta)*np.cos(theta) * dphi**2
        d2phi = - (2/r) * dr * dphi - 2 * np.cos(theta) / \
            np.sin(theta) * dtheta * dphi

        # The affine parameter derivative is always 1 (d(affine)/dlambda = 1)
        daffine_dlambda = 1.0

        # Return derivatives for ALL 9 components
        return np.array([dt, dr, dtheta, dphi, d2t, d2r, d2theta, d2phi, daffine_dlambda])

    def rk4_step(self, u, dlambda):
        """4th order Runge-Kutta step for 9-element state vector"""
        k1 = dlambda * self.derivatives(u)
        k2 = dlambda * self.derivatives(u + 0.5 * k1)
        k3 = dlambda * self.derivatives(u + 0.5 * k2)
        k4 = dlambda * self.derivatives(u + k3)

        u_new = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        return u_new
    

class LightRay:
    """
    Input an initial 4-position and 3-velocity, and sign.
    Computes the time component of the 4-velocity to satisfy null condition.
    """

    def __init__(self, pos, v, sign=True, M=1.0):
        """
        pos: [t, r, theta, phi] - initial position
        v: [dr/dlambda, dtheta/dlambda, dphi/dlambda] - spatial components of 4-velocity
        sign: True for future-directed, False for past-directed
        M: Black hole mass (needed for proper initialization)
        """
        t, r, theta, phi = pos
        dr_dlambda, dtheta_dlambda, dphi_dlambda = v

        # Schwarzschild metric components
        f = 1 - 2*M/r

        # Solve null condition: g_μν u^μ u^ν = 0 for dt/dlambda
        # -(1-2M/r)(dt)^2 + (dr)^2/(1-2M/r) + r^2(dtheta)^2 + r^2 sin^2θ (dphi)^2 = 0
        spatial_part = (dr_dlambda**2 / f) + (r**2 * dtheta_dlambda **
                                              2) + (r**2 * np.sin(theta)**2 * dphi_dlambda**2)
        dt_dlambda_squared = spatial_part / f

        if dt_dlambda_squared < 0:
            raise ValueError(
                "Initial conditions cannot satisfy null condition!")

        dt_dlambda = np.sqrt(dt_dlambda_squared)
        if not sign:
            dt_dlambda = -dt_dlambda

        initial_conditions = np.array(
            [t, r, theta, phi, dt_dlambda, dr_dlambda, dtheta_dlambda, dphi_dlambda, 0.0])
        self.history = [initial_conditions.copy()]

    def evolve(self, M, total_lambda, n_steps=1000):
        """Evolve the ray"""
        dlambda = total_lambda / n_steps
        spacetime = SchwarzschildGeodesic(M)

        for _ in range(n_steps):
            current = self.history[-1].copy()
            new_state = spacetime.rk4_step(current, dlambda)

            position_change = np.linalg.norm(new_state[:4] - current[:4])
            if position_change < 1e-5:
                break

            self.history.append(new_state)

        return np.array(self.history)



class LightBeam:

    def __init__(self, central_pos, beam_direction, sign, beam_angle, num_rays, M):
        """
        central_pos: [t, r, theta, phi] - center of the beam
        beam_direction: [dr, dtheta, dphi] - central direction
        beam_angle: angular spread of the beam (radians)
        num_rays: number of rays in the beam
        """
        self.rays = []
        self.M = M

        # Generate multiple rays with slight directional variations
        for i in range(num_rays):
            # Add random angular offset to create beam spread
            if num_rays == 1:
                # Single ray case
                offset_dtheta, offset_dphi = 0, 0
            else:
                # Random offsets within beam angle
                offset_angle = beam_angle * \
                    (i / (num_rays - 1) - 0.5)  # Evenly spaced
                offset_dtheta = offset_angle * np.random.normal(0, 1)
                offset_dphi = offset_angle * np.random.normal(0, 1)

            # Create direction for this ray
            ray_direction = [
                # Small radial variation
                beam_direction[0] + np.random.normal(0, beam_angle * 0.1),
                beam_direction[1] + offset_dtheta,
                beam_direction[2] + offset_dphi
            ]

            # Create the ray
            ray = LightRay(central_pos, ray_direction, sign, M=M)
            self.rays.append(ray)

    def evolve_all(self, total_lambda, n_steps=500):
        """Evolve all rays in the beam"""
        for ray in self.rays:
            ray.evolve(self.M, total_lambda, n_steps)

    def get_all_histories(self):
        """Get history of all rays"""
        return [ray.history for ray in self.rays]