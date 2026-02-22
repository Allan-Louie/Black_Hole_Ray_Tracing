from geodesic_integrator import SchwarzschildGeodesic
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %% camera geometry rendering


def cartesian_to_spherical_velocity(v, pos):
    """
    Convert Cartesian tangent vectors to spherical coordinate tangent vectors

    Parameters:
    v: 3-velocity in cartesian coordinates to be tangent mapped
    pos: 4-position in spherical coordinates defining the local tangent map (time is ignored)

    Returns:
    (dr_dlambda, dtheta_dlambda, dphi_dlambda): Spherical coordinate velocity components
    """

    _, r, theta, phi = pos
    x_dot, y_dot, z_dot = v

    if r < 1e-10:
        return 0.0, 0.0, 0.0

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # The transformation: [ṙ, θ̇, φ̇]ᵀ = J⁻¹ * [ẋ, ẏ, ż]ᵀ
    dr_dlambda = (x_dot * sin_theta * cos_phi +
                  y_dot * sin_theta * sin_phi +
                  z_dot * cos_theta)

    dtheta_dlambda = (x_dot * cos_theta * cos_phi +
                      y_dot * cos_theta * sin_phi -
                      z_dot * sin_theta) / r

    if abs(sin_theta) < 1e-10:
        dphi_dlambda = 0.0
    else:
        dphi_dlambda = (-x_dot * sin_phi + y_dot * cos_phi) / (r * sin_theta)

    return dr_dlambda, dtheta_dlambda, dphi_dlambda


class Camera:
    """
    Simulates a pinhole camera in curved spacetime
    """

    def __init__(self, pos, orientation, fov, resolution):
        """
        pos: [t, r, theta, phi] - camera position
        orientation: [theta_view, phi_view] - camera pointing direction in RADIANS
        fov: field of view in DEGREES horizontally and vertically
        resolution: (width, height) of the image
        """
        if fov[0] >= 180:
            raise Exception(
                'Field of view cannot exceed 180 degrees in horizontal direction')

        if fov[1] >= 180:
            raise Exception(
                'Field of view cannot exceed 180 degrees in vertical direction')

        if orientation[0] in [0, np.pi]:
            raise Exception('Choose another point of view.')

        self.pos = np.array(pos)
        self.orientation = np.array(orientation)
        self.fovx = np.radians(fov[0])
        self.fovy = np.radians(fov[1])
        self.resolution = resolution
        self.width, self.height = resolution

        self.__setup_camera_basis()

    def __setup_camera_basis(self):
        """Set up camera coordinate system, x, y, z hat"""
        theta_cam, phi_cam = self.orientation

        # Camera forward direction (looking direction)
        self.forward = np.array([
            np.sin(theta_cam) * np.cos(phi_cam),
            np.sin(theta_cam) * np.sin(phi_cam),
            np.cos(theta_cam)
        ])

        # Camera up direction (approximate)
        self.up = np.array([0, 0, 1])  # Z-up initially

        # Right direction
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)

        # Recompute proper up direction
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)

    def get_ray_directions(self):
        """
        Generate ray directions for each pixel in camera coordinates
        Returns: list of [dr, dtheta, dphi] for each pixel
        """
        rays = []
        pix = []

        tan_fovx = np.tan(self.fovx / 2)
        tan_fovy = np.tan(self.fovy / 2)
        pos = self.pos

        for i in range(self.height):
            for j in range(self.width):

                x_ndc = (2 * j / self.width - 1) * tan_fovx
                y_ndc = (1 - 2 * i / self.height) * tan_fovy

                # Ray direction in camera space
                ray_dir_camera = np.array([x_ndc, y_ndc, 1])
                ray_dir_camera /= np.linalg.norm(ray_dir_camera)

                # Transform to global coordinates
                ray_dir_global = (
                    ray_dir_camera[0] * self.right +
                    ray_dir_camera[1] * self.up +
                    ray_dir_camera[2] * self.forward
                )

                # Convert Cartesian direction to spherical coordinates
                dr_dlambda, dtheta_dlambda, dphi_dlambda = cartesian_to_spherical_velocity(
                    ray_dir_global, pos)

                ray_direction = [dr_dlambda, dtheta_dlambda, dphi_dlambda]

                rays.append(ray_direction)
                pix.append((i, j))

        return rays, pix

# %% camera-associated beam integrator and radiation physics feedback


def render_camera_view(camera_pos, orientation, fov, resolution, M, total_lambda, n_steps, inner_radius, outer_radius, height):
    """
    Render what a camera at camera_pos would see
    """
    camera = Camera(camera_pos, orientation, fov, resolution)

    # Get ray directions for all pixels
    ray_directions, pixel_info = camera.get_ray_directions()

    # Create image buffer
    image = np.zeros((camera.height, camera.width, 3))

    for idx, ray_dir in enumerate(ray_directions):
        if idx % 500 == 0:
            print(f"Progress: {idx}/{len(ray_directions)}")

        i, j = pixel_info[idx]

        ray = AccretionDiskTracer(camera_pos, ray_dir, False, M)

        color = ray.evolve( M, total_lambda, n_steps, inner_radius, outer_radius, height)
        image[i, j] = color

    return image

# %% decision making criteria for sourcing radiation

class AccretionDiskTracer:
    """
    Input an initial 4-position and 3-velocity, and sign.
    Computes the time component of the 4-velocity to satisfy null condition.
    """

    def __init__(self, pos, v, sign, M):
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
        
        self.initial_conditions=initial_conditions
        self.history = [initial_conditions.copy()]

    def evolve(self, M, total_lambda, n_steps, inner_radius, outer_radius, height):
        """Evolve the ray"""
        dlambda = total_lambda / n_steps
        spacetime = SchwarzschildGeodesic(M)
        disk = AccretionDisk(inner_radius, outer_radius, height)

        for _ in range(n_steps):
            current = self.history[-1].copy()
            new_state = spacetime.rk4_step(current, dlambda)
            self.history.append(new_state)

            position_change = np.linalg.norm(new_state[:4] - current[:4])
            
            if position_change < 1e-5:
                return [0,0,0]
            
            if disk.intersects(new_state[1], new_state[2], new_state[3]):
                return [1.0, 0.784, 0.196]
            
            if new_state[1]>=max(outer_radius,self.initial_conditions[1]):
                return [0,0,0]
            
        return [0,0,0]
            



class AccretionDisk:
    def __init__(self, inner_radius, outer_radius, height):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.height = height

    def intersects(self, r, theta, phi):
        """Check if point is in accretion disk"""
        # Disk is in equatorial plane with some thickness
        in_radial_range = self.inner_radius <= r <= self.outer_radius
        in_vertical_range = abs(
            np.cos(theta)) < self.height / (2*r) if r > 0 else False
        return in_radial_range and in_vertical_range


# %% Running the static image


def quick_camera_render(camera_pos, orientation, fov, resolution, M, total_lambda, n_steps, inner_radius, outer_radius, height):
    """
    Quick camera render for testing
    """
    image = render_camera_view(camera_pos, orientation, fov, resolution,
                               M, total_lambda, n_steps, inner_radius, outer_radius, height)
    fovx, fovy = fov[0], fov[1]
    plt.figure(figsize=(fovx, fovy))
    plt.imshow(image, origin='upper')
    plt.axis('off')
    plt.title(f'Camera at r={camera_pos[1]:.1f}')
    plt.tight_layout()
    plt.show()

    return image

def orbit_animation(camera_r, theta_range, fov, resolution, 
                   M, total_lambda, n_steps, inner_radius, outer_radius, height, 
                   n_frames=24, save_video=True, save_frames=True, video_path="blackhole_orbit.mp4"):
    
    theta_start, theta_end = np.radians(theta_range)
    theta_values = np.linspace(theta_start, theta_end, n_frames)

    pre_rendered_frames = []
    print("Pre-rendering all frames...")

    for frame in range(n_frames):
        theta = theta_values[frame]
        camera_pos = [0, camera_r, theta, 0]
        orientation = [np.pi-theta, np.pi]
        
        image = render_camera_view(
            camera_pos=camera_pos,
            orientation=orientation,
            fov=fov, resolution=resolution, M=M,
            total_lambda=total_lambda, n_steps=n_steps,
            inner_radius=inner_radius, outer_radius=outer_radius, height=height
        )
        pre_rendered_frames.append(image)
        
        progress = (frame + 1) / n_frames * 100
        print(f"Pre-rendered frame {frame+1}/{n_frames} ({progress:.1f}%) - theta: {np.degrees(theta):.1f}°")
    
    # === AUTO-SAVE FRAMES TO DISK ===
    if save_frames:
        import pickle
        frames_path = "pre_rendered_frames_backup.pkl"
        with open(frames_path, 'wb') as f:
            pickle.dump(pre_rendered_frames, f)
        print(f"✅ Frames backed up to: {frames_path}")
    
    fig = plt.figure(figsize=(fov[0]/10, fov[1]/10), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.set_axis_off()
    
    im = ax.imshow(pre_rendered_frames[0], origin='upper')
    plt.tight_layout(pad=0)

    def update_frame(frame):
        im.set_array(pre_rendered_frames[frame])
        return [im]
    
    anim = FuncAnimation(fig, update_frame, frames=n_frames, 
                        interval=2000/n_frames, blit=True, repeat=False)
    
    # === AUTO-SAVE VIDEO ===
    if save_video:
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            print(f"🔄 Saving video to: {video_path}")
            anim.save(video_path, writer='ffmpeg', fps=n_frames/2, dpi=100, 
                    bitrate=2000, extra_args=['-vcodec', 'libx264'])
            print(f"✅ Video saved to: {video_path}")
        except Exception as e:
            print(f"❌ Failed to save video with ffmpeg: {e}")
            # Try fallback save with more debugging
            try:
                gif_path = video_path.replace('.mp4', '.gif')
                # Ensure directory exists for GIF too
                os.makedirs(os.path.dirname(gif_path), exist_ok=True)
                print(f"🔄 Attempting GIF fallback to: {gif_path}")
                anim.save(gif_path, writer='pillow', fps=n_frames/2)
                print(f"✅ Fallback GIF saved as '{gif_path}'")
            except Exception as e2:
                print(f"❌ GIF fallback also failed: {e2}")
                print("❌ Could not save video at all")
    
    plt.show()
    return anim, pre_rendered_frames

def save_animation_later(anim, n_frames, save_path="blackhole_saved.mp4"):
    """Save an existing animation object"""
    try:
        gif_path = save_path.replace('.mp4', '.gif')  # ← Convert to GIF
        print(f"Saving animation to: {gif_path}")
        anim.save(gif_path, writer='pillow', fps=n_frames/2)
        print("✅ Save successful!")
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False