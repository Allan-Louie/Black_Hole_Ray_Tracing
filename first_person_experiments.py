from first_person_rendering import quick_camera_render, cartesian_to_spherical_velocity
import numpy as np

M = 10.0

# Camera position - elevated above equatorial plane
camera_pos = [0, 30*M, np.pi/2-np.radians(89), 0]  
# [t, r, theta, phi] where theta = π/2 - 20° = 70° from north pole

# Orientation - looking down toward black hole
orientation = [np.pi/2 + np.radians(89), np.pi]  
# [theta_view, phi_view] - looking 10° below horizontal

# Disk parameters  
inner_radius = 6*M
outer_radius = 11*M  
height = 2*M

# Render settings
fov = [60, 45]
resolution = (20, 15)
total_lambda = 400.0
n_steps = 100

# Run it!
image = quick_camera_render(
    camera_pos=camera_pos,
    orientation=orientation, 
    fov=fov,
    resolution=resolution,
    M=M,
    total_lambda=total_lambda,
    n_steps=n_steps,
    inner_radius=inner_radius,
    outer_radius=outer_radius, 
    height=height
)


#%%

def test_your_conversion():
    """Test your implementation"""
    
    # Test case: Pure radial outward motion at equator
    pos = [0, 5.0, np.pi/2, np.pi/2]  # [t, r, theta, phi]
    v_cartesian = [1.0, 0.0, 0.0]  # Moving in +x direction
    
    r_dot, theta_dot, phi_dot = cartesian_to_spherical_velocity(v_cartesian, pos)
    
    print(f"Input Cartesian velocity: {v_cartesian}")
    print(f"Output Spherical velocity: ({r_dot:.3f}, {theta_dot:.3f}, {phi_dot:.3f})")

#test_your_conversion()
