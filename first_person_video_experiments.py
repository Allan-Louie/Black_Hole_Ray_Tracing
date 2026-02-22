from first_person_rendering import orbit_animation, save_animation_later

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

M=10.0
camera_r= 30*M
theta_range=(1,179)
fov = [60, 45]
resolution = (10, 8)
total_lambda = 400.0
n_steps = 100
inner_radius = 6*M
outer_radius = 11*M  
height = 2*M


video, pre_rendered_frames = orbit_animation(
    camera_r, theta_range, fov, resolution, 
    M, total_lambda, n_steps, inner_radius, outer_radius, height, 
    n_frames=24, save_video=True, save_frames=True, 
    video_path=f"C:\\Users\\User\\Videos\\blackhole_orbit_{timestamp}.mp4"
)

    
save_animation_later(video, n_frames=24, save_path=f"my_video_{timestamp}.mp4")