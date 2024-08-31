# in play.py
        if MOVE_CAMERA:
            env.camera_tracking()

# in robot.py

def camera_tracking(self):
    # camera_direction = np.array([0., 2., 1.3])
    z_direction = np.array([0., 0., 0.1])
    camera_actor = self.camera_actor * (self.num_envs // self.num_groups)
    camera_direction  = self.projected_forward[camera_actor].cpu().numpy()
    root_state = self.root_states[camera_actor, :3].cpu().numpy()

    camera_position = root_state + 0.25 * camera_direction - z_direction
    self.set_camera(camera_position, camera_position - camera_direction)