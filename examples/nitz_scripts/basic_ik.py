"""Basic IK without visualization

Minimal Inverse Kinematics Example using PyRoki without any visualization.
"""
import sys
sys.path.append("/app/vendor/pyroki/examples/")
import time
from pathlib import Path
import torch
import numpy as np
import pyroki as pk
import yourdfpy
import pyroki_snippets as pks
from scipy.spatial.transform import Rotation as R
import jax
import jax.numpy as jnp
import jaxlie

def _xarm_pose_to_curobo_position_and_quaternion(
    pose: np.ndarray, degrees: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a pose from xArm format (mm, rotvec in rads) to CuRobo's position (meters) and quaternion format.
    """
    position = torch.tensor(pose[0:3], device="cuda:0", dtype=torch.float32) / 1000.0
    quaternion = torch.roll(
        torch.tensor(R.from_rotvec(pose[3:6], degrees=degrees).as_quat(), device="cuda:0", dtype=torch.float32), shifts=1
    )  # note cuRobo uses scalar first quaternions rather than scipy's scalar last hence the use of roll
    return (position, quaternion)

def _xarm_pose_to_cpu_position_and_quaternion(
    pose: np.ndarray, degrees: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a pose from xArm format (mm, rotvec in rads) to CPU JAX arrays.
    """
    position = jnp.array(pose[0:3] / 1000.0, dtype=jnp.float32)  # Convert mm to meters
    quaternion = jnp.array(np.roll(R.from_rotvec(pose[3:6], degrees=degrees).as_quat(), shift=1), dtype=jnp.float32)  # scalar first
    return (position, quaternion)

def warm_up_jax():
    """Warm up JAX with various operations to trigger compilation"""
    print("=== Warming up JAX ===")
    
    # Basic JAX operations
    print("1. Basic JAX operations...")
    start = time.time()
    x = jnp.ones((10, 10))
    y = jnp.dot(x, x)
    y.block_until_ready()  # Ensure computation completes
    basic_time = time.time() - start
    print(f"   Basic operations: {basic_time*1000:.2f} ms")
    
    # JIT compilation test
    print("2. JIT compilation test...")
    @jax.jit
    def simple_function(x):
        return jnp.sum(x ** 2)
    
    start = time.time()
    result = simple_function(jnp.ones(100))
    result.block_until_ready()
    jit_time = time.time() - start
    print(f"   JIT compilation: {jit_time*1000:.2f} ms")
    
    # Gradient computation
    print("3. Gradient computation...")
    start = time.time()
    grad_fn = jax.grad(lambda x: jnp.sum(x ** 2))
    grad_result = grad_fn(jnp.ones(6))  # 6-DOF robot
    grad_result.block_until_ready()
    grad_time = time.time() - start
    print(f"   Gradient computation: {grad_time*1000:.2f} ms")
    
    # Multiple runs to see if compilation time decreases
    print("4. Multiple JIT runs...")
    times = []
    for i in range(5):
        start = time.time()
        result = simple_function(jnp.ones(100))
        result.block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f} ms")
    
    print(f"   Average after warmup: {np.mean(times)*1000:.2f} ms")
    print(f"   First run vs average: {times[0]/np.mean(times):.2f}x slower")

def warm_up_pyroki(robot, target_link_name):
    """Warm up PyRoKi specific operations"""
    print("\n=== Warming up PyRoKi ===")
    
    # Forward kinematics
    print("1. Forward kinematics...")
    start = time.time()
    fk_result = robot.forward_kinematics(np.zeros(robot.joints.num_actuated_joints))
    fk_result.block_until_ready()
    fk_time = time.time() - start
    print(f"   Forward kinematics: {fk_time*1000:.2f} ms")
    
    # Multiple FK runs
    print("2. Multiple FK runs...")
    times = []
    for i in range(5):
        start = time.time()
        fk_result = robot.forward_kinematics(np.zeros(robot.joints.num_actuated_joints))
        fk_result.block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f} ms")
    
    print(f"   Average FK time: {np.mean(times)*1000:.2f} ms")
    
    # Test IK compilation (but don't solve)
    print("3. Testing IK function compilation...")
    try:
        # Create dummy inputs for IK - USE THE SAME TYPE AS ACTUAL SOLVE
        # Convert dummy pose using the same function as actual solve
        dummy_xarm_pose = np.array([500.0, 0.0, 500.0, 0.0, 0.0, 0.0])  # dummy pose
        dummy_position, dummy_wxyz = _xarm_pose_to_curobo_position_and_quaternion(dummy_xarm_pose)
        
        # This will trigger JIT compilation of the IK function
        start = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=dummy_position,
            target_wxyz=dummy_wxyz,
        )
        # solution.block_until_ready()
        ik_compile_time = time.time() - start
        print(f"   First IK compilation: {ik_compile_time*1000:.2f} ms")
        
        # Test multiple IK runs
        print("4. Multiple IK runs...")
        times = []
        for i in range(3):
            start = time.time()
            solution = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=dummy_position,
                target_wxyz=dummy_wxyz,
            )
            # solution.block_until_ready()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Run {i+1}: {elapsed*1000:.2f} ms")
        
        print(f"   Average IK time: {np.mean(times)*1000:.2f} ms")
        if len(times) > 1:
            print(f"   First vs average: {times[0]/np.mean(times[1:]):.2f}x slower")
            
    except Exception as e:
        print(f"   IK warmup failed: {e}")

def solve_ik(target_position, target_wxyz, robot, target_link_name):    # Solve IK
    start_time = time.time()
    solution = pks.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=target_position,
        target_wxyz=target_wxyz,
    )
    elapsed_time = time.time() - start_time

    print(f"IK solved in {elapsed_time*1000:.2f} ms")
    print(f"Joint configuration: {solution}")
    print(f"Number of joints: {robot.joints.num_actuated_joints}")
    print(f"Joint names: {robot.joints.actuated_names}")
    
    # You can also verify the solution by computing forward kinematics
    fk_result = robot.forward_kinematics(solution)
    print(f"Forward kinematics result shape: {fk_result.shape}")

def run_IK(robot, target_link_name, xarm_pose, device="gpu", num_runs=10):
    """
    Run IK with warmup and multiple iterations for performance testing.
    
    Args:
        robot: PyRoKi Robot
        target_link_name: Target link name
        xarm_pose: Pose in xArm format
        device: "gpu" or "cpu"
        num_runs: Number of IK runs to average
    
    Returns:
        dict with timing results
    """
    print(f"\n=== Running IK on {device.upper()} ===")
    
    # Configure JAX backend based on device
    if device == "gpu":
        target_position, target_wxyz = _xarm_pose_to_curobo_position_and_quaternion(xarm_pose)
    else:  # cpu
        target_position, target_wxyz = _xarm_pose_to_cpu_position_and_quaternion(xarm_pose)
    
    # Verify JAX is using the correct backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Target position: {target_position}")
    print(f"Target quaternion (wxyz): {target_wxyz}")
    
    # Warmup run (not timed in final results)
    print("Performing warmup run...")
    start = time.time()
    warmup_solution = pks.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=target_position,
        target_wxyz=target_wxyz,
    )
    warmup_time = time.time() - start
    print(f"Warmup time: {warmup_time*1000:.2f} ms")
    print(f"Warmup solution: {warmup_solution}")
    
    # Verify warmup solution with forward kinematics
    print("\nVerifying warmup solution with forward kinematics...")
    fk_start = time.time()
    fk_result = robot.forward_kinematics(warmup_solution)
    fk_time = time.time() - fk_start
    print(f"Forward kinematics time: {fk_time*1000:.2f} ms")
    
    # Get the target link index to extract the correct pose
    target_link_index = robot.links.names.index(target_link_name)
    achieved_pose_array = fk_result[target_link_index]  # This is a JAX array with shape (7,) in wxyz_xyz format
    
    # Convert to SE3 object to extract position and quaternion
    achieved_pose = jaxlie.SE3(achieved_pose_array)
    achieved_position = achieved_pose.translation()
    achieved_quaternion = achieved_pose.rotation().wxyz  # PyRoKi uses wxyz format
    
    print(f"Achieved position: {achieved_position}")
    print(f"Achieved quaternion (wxyz): {achieved_quaternion}")
    
    # Calculate errors
    if device == "gpu":
        # Convert target tensors to numpy for comparison
        target_pos_np = target_position.detach().cpu().numpy()
        target_quat_np = target_wxyz.detach().cpu().numpy()
    else:
        target_pos_np = target_position
        target_quat_np = target_wxyz
    
    achieved_pos_np = np.array(achieved_position)
    achieved_quat_np = np.array(achieved_quaternion)
    
    position_error = np.linalg.norm(achieved_pos_np - target_pos_np)
    
    # Quaternion error (angular distance)
    # Handle quaternion double cover (q and -q represent same rotation)
    dot_product = np.abs(np.dot(achieved_quat_np, target_quat_np))
    dot_product = np.clip(dot_product, 0.0, 1.0)  # Numerical stability
    quaternion_error = 2 * np.arccos(dot_product)  # Angular distance in radians
    
    print(f"\nPose verification:")
    print(f"Position error: {position_error*1000:.3f} mm")
    print(f"Orientation error: {np.degrees(quaternion_error):.3f} degrees ({quaternion_error:.6f} rad)")
    
    # Convert back to xArm format for comparison
    achieved_rotvec = R.from_quat(np.roll(achieved_quat_np, -1)).as_rotvec()  # Convert wxyz to xyzw for scipy
    achieved_xarm_pose = np.concatenate([achieved_pos_np * 1000, achieved_rotvec])  # Convert m to mm
    
    print(f"Original xArm pose: {xarm_pose}")
    print(f"Achieved xArm pose: {achieved_xarm_pose}")
    xarm_pose_error = np.abs(achieved_xarm_pose - xarm_pose)
    print(f"xArm pose errors: {xarm_pose_error}")
    print(f"Max xArm pose error: {np.max(xarm_pose_error):.3f}")
    
    # Timed runs
    print(f"\nPerforming {num_runs} timed runs...")
    times = []
    solutions = []
    
    for i in range(num_runs):
        start = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )
        elapsed = time.time() - start
        times.append(elapsed)
        solutions.append(solution)
        print(f"Run {i+1:2d}: {elapsed*1000:6.2f} ms - Solution: {solution}")
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n{device.upper()} Results Summary:")
    print(f"Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Min time:  {min_time*1000:.2f} ms")
    print(f"Max time:  {max_time*1000:.2f} ms")
    print(f"Range:     {(max_time-min_time)*1000:.2f} ms")
    
    # Check for trends (compare first half vs second half)
    if num_runs >= 4:
        first_half = times[:num_runs//2]
        second_half = times[num_runs//2:]
        first_half_mean = np.mean(first_half)
        second_half_mean = np.mean(second_half)
        print(f"First half mean: {first_half_mean*1000:.2f} ms")
        print(f"Second half mean: {second_half_mean*1000:.2f} ms")
        print(f"Trend: {'Speeding up' if second_half_mean < first_half_mean else 'Slowing down' if second_half_mean > first_half_mean else 'Stable'}")
    
    # Verify all solutions are similar
    solution_array = np.array(solutions)
    solution_std = np.std(solution_array, axis=0)
    print(f"Solution consistency (std per joint): {solution_std}")
    print(f"Max solution std: {np.max(solution_std):.6f}")
    
    return {
        'device': device,
        'warmup_time': warmup_time,
        'fk_time': fk_time,
        'times': times,
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'solutions': solutions,
        'solution_std': solution_std,
        'position_error': position_error,
        'quaternion_error': quaternion_error,
        'achieved_xarm_pose': achieved_xarm_pose,
        'xarm_pose_error': xarm_pose_error
    }

def main():
    """Main function for basic IK without visualization."""
    print('JAX DEVICES:', jax.devices())
    print(f'JAX DEFAULT BACKEND:', jax.default_backend())

    # Warm up JAX first
    # warm_up_jax()

    # Load your own URDF file
    urdf_path = Path("/app/src/xarm_controller/xarm_urdf/xarm6_with_200mm_tcp.urdf")  # Replace with your URDF path
    
    # Optional: If your URDF references mesh files, you may need a filename handler
    def filename_handler(fname: str) -> str:
        base_path = urdf_path.parent
        return yourdfpy.filename_handler_magic(fname, dir=base_path)
    
    # Load the URDF
    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
    
    # Update target link name to match your robot's end effector
    target_link_name = "tcp_link"  # Replace with your robot's end effector link name

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    xarm_pose = np.array([-305.8, 69.1, 526.4, 2.6337, -0.3979, -1.3683])
    xarm_joints = np.array([-0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000])

    num_runs = 10
    
    # Test on GPU
    with jax.default_device(jax.devices("gpu")[0]):
        gpu_results = run_IK(robot, target_link_name, xarm_pose, device="gpu", num_runs=num_runs)
    
    # Test on CPU
    # with jax.default_device(jax.devices("cpu")[0]):
    #     cpu_results = run_IK(robot, target_link_name, xarm_pose, device="cpu", num_runs=num_runs)
    
    # # Compare results
    # print(f"\n{'='*50}")
    # print("PERFORMANCE COMPARISON")
    # print(f"{'='*50}")
    # print(f"GPU Mean: {gpu_results['mean_time']*1000:6.2f} ms ± {gpu_results['std_time']*1000:.2f} ms")
    # print(f"CPU Mean: {cpu_results['mean_time']*1000:6.2f} ms ± {cpu_results['std_time']*1000:.2f} ms")
    
    # speedup = cpu_results['mean_time'] / gpu_results['mean_time']
    # print(f"GPU Speedup: {speedup:.2f}x faster than CPU")
    
    # # Compare solution accuracy
    # gpu_solution = gpu_results['solutions'][0]  # First solution
    # cpu_solution = cpu_results['solutions'][0]  # First solution
    # solution_diff = np.abs(gpu_solution - cpu_solution)
    # print(f"\nSolution difference (GPU vs CPU):")
    # print(f"Max joint difference: {np.max(solution_diff):.6f} rad")
    # print(f"Mean joint difference: {np.mean(solution_diff):.6f} rad")
    
    # # Verify with recorded joints
    # gpu_error = np.abs(gpu_solution - xarm_joints)
    # cpu_error = np.abs(cpu_solution - xarm_joints)
    # print(f"\nComparison to recorded joints:")
    # print(f"GPU max error: {np.max(gpu_error):.6f} rad")
    # print(f"CPU max error: {np.max(cpu_error):.6f} rad")

if __name__ == "__main__":
    main()