import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def blackhole_animator(ray_history, M):
    """
    Simple smooth animation with black hole sphere
    """
    # Extract data
    history_array = np.array(ray_history)
    t_coord = history_array[:, 0]
    r = history_array[:, 1]
    theta = history_array[:, 2]
    phi = history_array[:, 3]

    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Create interpolation functions
    x_interp = interp1d(t_coord, x, kind='cubic')
    y_interp = interp1d(t_coord, y, kind='cubic')
    z_interp = interp1d(t_coord, z, kind='cubic')

    # Create smooth time grid for animation
    t_smooth = np.linspace(t_coord[0], t_coord[-1], 150)

    # Pre-compute smooth trajectory
    x_smooth = x_interp(t_smooth)
    y_smooth = y_interp(t_smooth)
    z_smooth = z_interp(t_smooth)

    # Set up the plot
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('black')

    # Draw black hole as a simple black sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_bh = 2*M * np.outer(np.cos(u), np.sin(v))
    y_bh = 2*M * np.outer(np.sin(u), np.sin(v))
    z_bh = 2*M * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_bh, y_bh, z_bh, color='white', alpha=0.8)

    # Initialize trajectory and point
    trajectory, = ax.plot([], [], [], 'white', linewidth=2)
    point, = ax.plot([], [], [], 'cyan', markersize=6)
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    # Set fixed limits
    all_x = np.concatenate([x_smooth, x_bh.flatten()])
    all_y = np.concatenate([y_smooth, y_bh.flatten()])
    all_z = np.concatenate([z_smooth, z_bh.flatten()])

    # Find the overall maximum range across all axes
    x_range = np.ptp(all_x)  # peak-to-peak (max-min)
    y_range = np.ptp(all_y)
    z_range = np.ptp(all_z)
    # Half range for center-based limits
    max_range = max(x_range, y_range, z_range) * 0.5

    # Find centers of each axis
    mid_x = (np.max(all_x) + np.min(all_x)) * 0.5
    mid_y = (np.max(all_y) + np.min(all_y)) * 0.5
    mid_z = (np.max(all_z) + np.min(all_z)) * 0.5

    # Set IDENTICAL ranges for all axes
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1, 1, 1])

    ax.set_axis_off()

    def animate(frame):
        trajectory.set_data(x_smooth[:frame+1], y_smooth[:frame+1])
        trajectory.set_3d_properties(z_smooth[:frame+1])
        point.set_data([x_smooth[frame]], [y_smooth[frame]])
        point.set_3d_properties([z_smooth[frame]])
        time_text.set_text(f'Time: t = {t_smooth[frame]:.2f}')
        return trajectory, point, time_text

    anim = FuncAnimation(fig, animate, frames=len(t_smooth),
                         interval=40, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()
    return anim


def blackhole_animator_beams(beam_histories, M):
    """
    Animates the 3D projection of the Schwarzchild geodesic for a beam of light rays.
    """

    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('black')

    # Draw black hole as a simple black sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_bh = 2*M * np.outer(np.cos(u), np.sin(v))
    y_bh = 2*M * np.outer(np.sin(u), np.sin(v))
    z_bh = 2*M * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_bh, y_bh, z_bh, color='white', alpha=0.8)

    interpolated_data = []
    trajectories = []
    points = []

    for i, history in enumerate(beam_histories):
        history_array = np.array(history)
        t_coord = history_array[:, 0]
        r = history_array[:, 1]
        theta = history_array[:, 2]
        phi = history_array[:, 3]
        
        # Convert to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Interpolate for smooth animation
        x_interp = interp1d(t_coord, x, kind='cubic')
        y_interp = interp1d(t_coord, y, kind='cubic')
        z_interp = interp1d(t_coord, z, kind='cubic')
        
        t_smooth = np.linspace(t_coord[0], t_coord[-1], 150)
        x_smooth = x_interp(t_smooth)
        y_smooth = y_interp(t_smooth)
        z_smooth = z_interp(t_smooth)
        
        # STORE IN NEW LIST instead of modifying original
        interpolated_data.append((t_smooth, x_smooth, y_smooth, z_smooth))
        
        # Create trajectory and point for this ray
        traj, = ax.plot([], [], [], color='white', linewidth=1.5, alpha=0.7)
        pt, = ax.plot([], [], [], color='white', markersize=4)
        
        trajectories.append(traj)
        points.append(pt)

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, 
                         color='white', fontsize=12)

    all_x, all_y, all_z = [], [], []

    for history in interpolated_data:
        _, x_smooth, y_smooth, z_smooth = history
        all_x.extend(x_smooth)
        all_y.extend(y_smooth)
        all_z.extend(z_smooth)
    
    max_abs = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), np.max(np.abs(all_z))) * 1.2
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_zlim(-max_abs, max_abs)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    n_stars = 150
    star_x = np.random.uniform(-max_abs, max_abs, n_stars)
    star_y = np.random.uniform(-max_abs, max_abs, n_stars)
    star_z = np.random.uniform(-max_abs, max_abs, n_stars)
    ax.scatter(star_x, star_y, star_z, c='white', s=1, alpha=0.5)

    def animate(frame):
        for i, history in enumerate(interpolated_data):
            t_smooth, x_smooth, y_smooth, z_smooth = history
            trajectories[i].set_data(x_smooth[:frame+1], y_smooth[:frame+1])
            trajectories[i].set_3d_properties(z_smooth[:frame+1])
            points[i].set_data([x_smooth[frame]], [y_smooth[frame]])
            points[i].set_3d_properties([z_smooth[frame]])
        
        time_text.set_text(f'Time: t = {t_smooth[frame]:.2f}')
        return trajectories + points + [time_text]
    
    anim = FuncAnimation(fig, animate, frames=len(t_smooth), 
                        interval=40, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    return anim
