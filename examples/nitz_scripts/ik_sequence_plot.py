import json
import time
from pathlib import Path
from typing import List

import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

import pyroki as pk
import yourdfpy

# -----------------------------
# Variable definitions
# -----------------------------

class PoseVar(
    jaxls.Var,
    default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    tangent_dim=0,
    retract_fn=lambda x, d: x,  # identity retract (parameter, not optimised)
):
    """Holds target pose as [wxyz, xyz], treated as constant parameter (tangent_dim=0)."""


# -----------------------------
# Cost that reads pose from PoseVar
# -----------------------------

@jaxls.Cost.create_factory(jac_mode="auto")
def variable_pose_cost(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    pose_var: PoseVar,
    target_link_index: jax.Array,
    pos_weight: float,
    ori_weight: float,
) -> jax.Array:
    """Residual 6-vector with pose_var providing target pose."""
    q = vals[joint_var]
    pose7 = vals[pose_var]
    target_quat = pose7[:4]
    target_pos = pose7[4:]
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_quat), target_pos
    )
    Ts_world = robot.forward_kinematics(q)
    T_world_ee = jaxlie.SE3(Ts_world[target_link_index])
    error = (target_pose.inverse() @ T_world_ee).log()
    weighted = error * jnp.array([pos_weight] * 3 + [ori_weight] * 3)
    return weighted

# -----------------------------
# Helper conversions
# -----------------------------

def xarm_pose_to_position_quat(pose: np.ndarray, degrees: bool = False):
    """Convert xArm pose (mm, rotvec rad) to meters + wxyz quaternion (numpy arrays)."""
    position = pose[0:3] / 1000.0  # mm -> m
    quat_xyzw = R.from_rotvec(pose[3:6], degrees=degrees).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)  # move scalar to front
    return position.astype(np.float32), quat_wxyz.astype(np.float32)


def quat_wxyz_to_euler(quat_wxyz: np.ndarray, degrees: bool = True):
    """Quaternion wxyz -> Euler xyz angles."""
    quat_xyzw = np.roll(quat_wxyz, -1)
    euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=degrees)
    return euler

# -----------------------------
# Main routine
# -----------------------------

def main():
    data_path = Path(__file__).parent / "poses_and_joints.json"
    if not data_path.exists():
        raise FileNotFoundError("poses_and_joints.json not found next to script")

    data = json.loads(data_path.read_text())
    assert isinstance(data, list) and len(data) >= 2

    def get_key(e, keys):
        for k in keys:
            if k in e:
                return e[k]
        raise KeyError

    poses_np = np.array([get_key(e, ["pose", "observation.xarm_pose", "xarm_pose"]) for e in data], dtype=np.float32)
    joints_np = np.array([get_key(e, ["joints", "observation.xarm_joints", "xarm_joints"]) for e in data], dtype=np.float32)
    if np.max(np.abs(joints_np)) > 3.2:
        joints_np = np.deg2rad(joints_np)

    urdf_path = Path("/app/src/xarm_controller/xarm_urdf/xarm6_with_200mm_tcp.urdf")
    def filename_handler(fname: str):
        return yourdfpy.filename_handler_magic(fname, dir=urdf_path.parent)

    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
    robot = pk.Robot.from_urdf(urdf)
    target_link_index = robot.links.names.index("tcp_link")

    # Build variables and problem once
    joint_var = robot.joint_var_cls(0)
    pose_var = PoseVar(0)

    factors = [
        variable_pose_cost(robot, joint_var, pose_var, jnp.array(target_link_index, dtype=jnp.int32), 50., 10.0),
        pk.costs.limit_cost(robot, joint_var, weight=100.0),
    ]

    analyzed = jaxls.LeastSquaresProblem(factors, [joint_var, pose_var]).analyze()

    achieved_poses = [poses_np[0]]
    achieved_joints = [joints_np[0]]
    solve_times = []
    seed_cfg = joints_np[0]

    for pose in poses_np[1:]:
        target_pos, target_quat = xarm_pose_to_position_quat(pose)
        pose7 = np.concatenate([target_quat, target_pos])

        vals = jaxls.VarValues.make([
            joint_var.with_value(jnp.array(seed_cfg, dtype=jnp.float32)),
            pose_var.with_value(jnp.array(pose7, dtype=jnp.float32)),
        ])

        t0 = time.time()
        sol = analyzed.solve(initial_vals=vals, verbose=False, linear_solver="dense_cholesky")
        dt = time.time() - t0
        solve_times.append(dt)
        print(f"Frame {len(solve_times):02d}: {dt*1000:.2f} ms")

        cfg_sol = np.array(sol[joint_var])
        achieved_joints.append(cfg_sol)
        seed_cfg = cfg_sol

        # forward kinematics for achieved pose
        fk_T = robot.forward_kinematics(cfg_sol)
        T_world_ee = jaxlie.SE3(fk_T[target_link_index])
        ach_pos = np.array(T_world_ee.translation()) * 1000.0
        ach_rotvec = R.from_quat(np.roll(np.array(T_world_ee.rotation().wxyz), -1)).as_rotvec()
        achieved_poses.append(np.concatenate([ach_pos, ach_rotvec]))

        # Error metrics
        diff = np.abs(np.concatenate([ach_pos, ach_rotvec]) - pose)
        pos_err = diff[:3].max()  # mm
        ang_err_deg = np.degrees(diff[3:].max())
        print(f"Errors frame {len(solve_times):02d}: pos {pos_err:.3f} mm | ori {ang_err_deg:.3f} deg")

    achieved_poses = np.stack(achieved_poses)
    achieved_joints = np.stack(achieved_joints)

    # ---------------- Plot 1: Pose comparison ----------------
    t = np.arange(len(poses_np))
    fig_pose = go.Figure()
    labels = ["x [mm]", "y [mm]", "z [mm]", "rx", "ry", "rz"]
    colors = ["red", "green", "blue", "orange", "purple", "cyan"]

    for idx in range(6):
        fig_pose.add_trace(
            go.Scatter(x=t, y=poses_np[:, idx], mode="lines", name=f"rec {labels[idx]}", line=dict(color=colors[idx]))
        )
        fig_pose.add_trace(
            go.Scatter(x=t, y=achieved_poses[:, idx], mode="lines", name=f"ach {labels[idx]}", line=dict(color=colors[idx], dash="dash"))
        )

    fig_pose.update_layout(title="Recorded vs Achieved Poses", xaxis_title="Pose index")
    fig_pose.write_html("pose_comparison.html")

    # ---------------- Plot 2: Joint comparison ----------------
    fig_joint = go.Figure()
    joint_labels = [f"joint_{i}" for i in range(achieved_joints.shape[1])]
    colors_joint = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "brown"]

    rec_j_mod = np.mod(joints_np, 2 * np.pi)
    ach_j_mod = np.mod(achieved_joints, 2 * np.pi)

    for idx in range(achieved_joints.shape[1]):
        c = colors_joint[idx % len(colors_joint)]
        fig_joint.add_trace(
            go.Scatter(x=t, y=rec_j_mod[:, idx], mode="lines", name=f"rec {joint_labels[idx]}", line=dict(color=c))
        )
        fig_joint.add_trace(
            go.Scatter(x=t, y=ach_j_mod[:, idx], mode="lines", name=f"ach {joint_labels[idx]}", line=dict(color=c, dash="dash"))
        )

    fig_joint.update_layout(title="Recorded vs Achieved Joint Angles (wrapped 0-2π)", xaxis_title="Pose index")
    fig_joint.write_html("joint_comparison.html")

    print("Plots saved to pose_comparison.html and joint_comparison.html")

    if solve_times:
        avg_time = np.mean(solve_times)
        print(f"Average IK solve time over {len(solve_times)} runs: {avg_time*1000:.2f} ms")


if __name__ == "__main__":
    main() 