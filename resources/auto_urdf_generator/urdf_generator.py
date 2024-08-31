# %%
import yaml

def generate_urdf(config, leglist):
    urdf_template = """
    <!-- URDF Template -->
    {content}
    """
    link_template = """
  <link name="{name}_UPPER_LEG">

    <!-- UPPER LEG LINK VISUAL -->
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}"/>
      </geometry>
      <material name="fthigh_mat">
        <color rgba="{up_color}"/>
      </material>
    </visual>

    <!-- UPPER LEG LINK COLLISION -->
    <collision>
      <origin rpy="0 0 0" xyz="{up_ori}"/>
      <geometry>
        <sphere radius="{up_radius}"/>
      </geometry>
    </collision>
    
  <link name="{name}_LOWER_LEG">

    <!-- LOWER LEG LINK VISUAL -->
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="../meshes/solo_lower_leg_left_side.dae"/>
      </geometry>
      <material name="fshin_mat">
        <color rgba="0.1 0.2 0.7 0.4"/>
      </material>
    </visual>
    
    <!-- LOWER LEG LINK COLLISION -->
    <collision>
      <origin rpy="0 0 0" xyz="{low_ori}"/>
      <geometry>
        <box size="0.02 0.012 0.11"/>
      </geometry>
    </collision>
    
    <!-- FOOT COLLISION -->
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="{foot_radius}"/>
      </geometry>
    </collision>

    <!-- ... (existing code) ... -->
  </link>
    """

    content = ""
    for leg in leglist:
      leg_config = config[leg]
      content += link_template.format(
          name=leg,
          up_radius=leg_config['radius_upper_leg'],
          foot_radius=leg_config['radius_foot'],
          up_ori=' '.join(map(str, leg_config['up_origin_xyz'])),
          up_color=' '.join(map(str, leg_config['up_material_color'])),
          low_ori=' '.join(map(str, leg_config['low_origin_xyz'])),
          low_color=' '.join(map(str, leg_config['low_material_color'])),
          mesh_filename=leg_config['mesh_filename']
    )

    return urdf_template.format(content=content)

if __name__ == "__main__":
    with open('example.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    leg_list = ['FL', 'FR', 'HL', 'HR']
    urdf_content = generate_urdf(config, leg_list)

    with open('generated_robot.urdf', 'w') as file:
        file.write(urdf_content)