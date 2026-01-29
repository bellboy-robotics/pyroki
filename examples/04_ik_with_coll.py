"""IK with Collision

Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import time

import jax
import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere, colldist_from_sdf
from pyroki.collision import _utils
from robot_descriptions.loaders.yourdfpy import load_robot_description
import yourdfpy
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import argparse


visualize_collision = True

def main(urdf_path: str | None = None):
    """Main function for basic IK with collision."""
    if urdf_path is None:
        # load default example urdf
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
    else:
        urdf = yourdfpy.URDF.load(urdf_path)
        target_link_name = "link_tcp"
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = RobotCollision.from_urdf(urdf, min_capsule=True)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create interactive controller for IK target.
    init_position = np.array([0.3, 0.0, 0.15, 0, 1, 0, 0])
    ik_target_handle = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=init_position[:3], wxyz=init_position[3:]
    )

    # # Create interactive controller and mesh for the sphere obstacle.
    # sphere_handle = server.scene.add_transform_controls(
    #     "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    # )
    # server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    last_solution = init_position
    current_collision_cost = None
    t_0 = time.time()
    while True:
        start_time = time.time()

        # sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
        #     wxyz=np.array(sphere_handle.wxyz),
        #     position=np.array(sphere_handle.position),
        # )

        # world_coll_list = [plane_coll, sphere_coll_world_current]
        world_coll_list = []#[plane_coll]

        # target_position = np.array([-3.97903890e-01,  1.23259516e-32,  4.13623306e-01])
        # target_wxyz = np.array( [ 7.31564766e-33,  8.42437486e-01, -3.65782383e-33, -5.38794100e-01])
        target_position = ik_target_handle.position
        target_wxyz = ik_target_handle.wxyz

        solution = pks.solve_ik_with_collision(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )

        # Update timing handle.
        timing_handle.value = (time.time() - start_time) * 1000

        # Update visualizer.
        urdf_vis.update_cfg(solution)

        if visualize_collision:
            current_collision_cost = visualize_collision(robot_coll, robot, server, solution, current_collision_cost)
        

def visualize_collision(robot_coll: RobotCollision, robot: pk.Robot, server: viser.ViserServer, solution: jax.Array, current_collision_cost: float):
    color_palette = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (255, 150, 100),  # Orange
        (150, 100, 255),  # Purple
        (100, 255, 150),  # Mint
        (255, 200, 100),  # Gold
    ]

    coll_world = robot_coll.at_config(robot, solution)
    for i, link_name in enumerate(robot_coll.link_names):
        # Extract individual capsule in world frame
        capsule = jax.tree.map(lambda x: x[i] if hasattr(x, '__len__') else x, coll_world)
        # Create mesh for this capsule
        capsule_mesh = capsule.to_trimesh()
        # Assign color based on link index (cycle through palette)
        color = color_palette[i % len(color_palette)]
        # Set the mesh color
        capsule_mesh.visual.vertex_colors = color
        # Add/update in scene
        server.scene.add_mesh_trimesh(
            f"/robot/collision/{link_name}",
            mesh=capsule_mesh
        )

    # Evaluate self collision cost
    active_distances = robot_coll.compute_self_collision_distance(robot, solution)
    residual = pk.collision.colldist_from_sdf(active_distances, 0.01)
    self_collision_cost = (residual * 5.0).flatten()
    self_collision_cost_scalar = np.linalg.norm(self_collision_cost)

    if current_collision_cost != self_collision_cost_scalar:
        print(self_collision_cost_scalar)
        current_collision_cost = self_collision_cost_scalar

        pairs = list(zip(robot_coll.active_idx_i, robot_coll.active_idx_j))
        for (i, j), dist in zip(pairs, active_distances.tolist()):
            if dist < 0.01:
                print(f"{robot.links.names[i]} <-> {robot.links.names[j]} : {dist:.4f}")
    
    return self_collision_cost_scalar
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-path", type=str, default=None)
    args = parser.parse_args()
    main(args.urdf_path)