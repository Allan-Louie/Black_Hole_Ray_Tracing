from geodesic_integrator import LightRay, LightBeam
from observer import blackhole_animator, blackhole_animator_beams
from first_person_rendering import cartesian_to_spherical_velocity
import numpy as np

M = 10.0

#Observer view: One light ray influenced by Schwarzchild curvature, circular motion

initial_pos = [0.0, 3*M, np.pi/2, 0.0]
initial_v = [0.0, 0.0, 0.5]
#ray = LightRay(initial_pos, initial_v, sign=True, M=M)
#trajectory = ray.evolve(M, total_lambda=70.0, n_steps=500)
#anim = blackhole_animator(ray.history, M)


#Observer view: One light beam influenced by Schwarzchild curvature, around 0 scattering on average

central_pos1 = [0.0, 3.03*M, np.pi/2, 0.0]
central_dir1 = [-0.12, 0.0, 0.5]  # Inward spiral

#beam1 = LightBeam(central_pos1, central_dir1, sign=True, beam_angle=0.01, num_rays=12, M=M)
#beam1.evolve_all(total_lambda=70.0, n_steps=1000)

#histories = beam1.get_all_histories()
#anim = blackhole_animator_beams(histories, M)

central_pos = [0, 60*M, np.pi/2, 0]  
orientation = [np.pi/2, np.pi]  
theta_cam, phi_cam= orientation[0], orientation[1]
forward=np.array([
            np.sin(theta_cam) * np.cos(phi_cam),
            np.sin(theta_cam) * np.sin(phi_cam),
            np.cos(theta_cam)
        ])

central_dir=cartesian_to_spherical_velocity(forward, central_pos)


beam = LightBeam(central_pos, central_dir, sign=False, beam_angle=0.001, num_rays=20, M=M)
beam.evolve_all(total_lambda=800.0, n_steps=1000)

histories = beam.get_all_histories()
anim = blackhole_animator_beams(histories, M)