solo8 init


```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.35
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -2.5
        torques = -1e-3
        lin_vel_z = 5.
        # feet_contact_forces = -0.01


class commands( LeggedRobotCfg.commands ):
    class ranges:
        lin_vel_x = [-1.0, 1.0] # min max [m/s]
        lin_vel_y = [-0.1, 0.1]   # min max [m/s]
        ang_vel_yaw = [-0.05, 0.05]    # min max [rad/s]
        heading = [-3.14, 3.14]
    heading_command = False
    resampling_time = 10.
    
```

solo9 init

```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -0.5
        base_height = -0.1 
        torques = -1e-3
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -0.5
        feet_contact_forces = -0.05
        
    
class commands( LeggedRobotCfg.commands ):
    class ranges:
        lin_vel_x = [-0.8, 0.8] # min max [m/s]
        ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
        heading = [-3.14, 3.14]
    heading_command = False
    resampling_time = 6.
```

solo9 

lin_vel_z = 0   feet_contact_forces = 0

lin_vel_x = [0, 1.0]

ang_vel_yaw = [-0.5, 0.5]

```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -0.5
        base_height = -0.1 
        torques = -1e-3
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -0.0
        feet_contact_forces = -0.00
        
    
class commands( LeggedRobotCfg.commands ):
    class ranges:
        lin_vel_x = [0.0, 1.0] # min max [m/s]
        ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
        heading = [-3.14, 3.14]
    heading_command = False
    resampling_time = 6.
```

solo9_ex1

```python
  
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -2.5 #
        base_height = -0.4 #
        torques = -1e-3
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -1.0 #
        feet_contact_forces = -0.0
```

solo9_ex2

``` python

class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -1.5 #
        # base_height = -1.0 #
        torques = -1e-3 
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -1.0
        feet_air_time = 2. #
        feet_contact_forces = -0.01 #
```

solo9_ex3

``` python

class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -1.5
        base_height = -1.0 #
        torques = -1e-3 
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -1.0
        feet_air_time = 2.
        feet_contact_forces = -0.01
```

solo9_exp4

```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -1.5
        base_height = -5.0 #
        collision = 5.0 #
        torques = -1e-3 
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -1.0
        feet_air_time = 2.
        contact_forces = -0.01
```

solo9_exp5

```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -1.5
        base_height = -2.0 #
        collision = 5.0 
        torques = -1e-3 
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -0.5 #
        feet_air_time = 2.
        contact_forces = -0.01
```

solo9 exp6

```python
class rewards( LeggedRobotCfg.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    max_contact_force = 80.
    class scales( LeggedRobotCfg.rewards.scales ):
        dof_pos_limits = -10.0
        orientation = -1.5
        base_height = -2.0 
        collision = 5.0 
        torques = -1e-3 
        tracking_lin_vel = 2.0
        tracking_ang_vel = 2.0
        lin_vel_z = -2.0 #
        feet_air_time = 2.
        contact_forces = -0.01
```

solo9 exs0

```python
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232 #
        max_contact_force = 80.
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = -10.0
            orientation = -1.5
            base_height = -2.5 # 
            action_bias = 0.005 #!
            collision = -5.0 
            torques = -1e-3
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0 
            feet_air_time = 2.
            feet_contact_forces = -0.01
```

solo9 exs1


```python
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232 #
        max_contact_force = 80.
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = -10.0
            orientation = -1.5
            base_height = -1.5 # 
            action_bias = 0.005
            collision = -5.0 
            torques = -1e-3
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0 
            feet_air_time = 2.
            feet_contact_forces = -0.01
```

solo9 exs2

```python
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232 #
        max_contact_force = 80.
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = -10.0
            orientation = -1.5
            base_height = -1.5 # 
            action_bias = 0.005
            collision = -5.0 
            torques = -1e-3
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -2.0 
            feet_air_time = 2.
            feet_contact_forces = -0.01
```