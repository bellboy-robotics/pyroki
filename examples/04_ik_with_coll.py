"""IK with Collision

Basic Inverse Kinematics with Collision Avoidance using PyRoKi.
"""

import time

import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere, colldist_from_sdf
from pyroki.collision import _utils
from robot_descriptions.loaders.yourdfpy import load_robot_description
import yourdfpy
from viser.extras import ViserUrdf
import pyroki_snippets as pks
import os
import jaxlie

    # Load URDF (the filename handler allows the mesh paths to resolve).
# def filename_handler(fname: str):
#     return yourdfpy.filename_handler_magic(fname, dir=os.path.dirname(fname))

urdf_path = "/Users/friyuval/releases/env/BILLIE-01/urdf/XI130511D43A0E.urdf" 
urdf = yourdfpy.URDF.load(urdf_path)


def main():
    """Main function for basic IK with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"
    target_link_name = "link_eef"
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
    init_position = np.array([0.5, 0.0, 0.5, 0, 0, 0, 1])
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
    t_0 = time.time()
    while True:
        start_time = time.time()

        # sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
        #     wxyz=np.array(sphere_handle.wxyz),
        #     position=np.array(sphere_handle.position),
        # )

        # world_coll_list = [plane_coll, sphere_coll_world_current]
        world_coll_list = [plane_coll]

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


if __name__ == "__main__":
    main()