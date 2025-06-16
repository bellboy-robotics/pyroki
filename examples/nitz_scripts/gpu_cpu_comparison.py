"""Benchmark CPU vs GPU performance for JAX operations"""

import time
import numpy as np
import jax
import jax.numpy as jnp

def benchmark_jax_operations():
    """Benchmark basic JAX operations on CPU vs GPU"""
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Test different problem sizes
    sizes = [10, 100, 1000, 5000]
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Testing matrix size: {size}x{size}")
        print(f"{'='*50}")
        
        # Create test data
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # CPU test
        print("\nCPU Test:")
        jax.config.update('jax_platform_name', 'cpu')
        
        # Warm up
        for _ in range(3):
            _ = jnp.dot(jnp.array(a), jnp.array(b))
        
        # Benchmark
        times_cpu = []
        for i in range(5):
            start = time.time()
            result = jnp.dot(jnp.array(a), jnp.array(b))
            result.block_until_ready()  # Ensure computation completes
            elapsed = time.time() - start
            times_cpu.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
        
        print(f"  CPU Average: {np.mean(times_cpu)*1000:.2f} ms")
        
        # GPU test (if available)
        print("\nGPU Test:")
        jax.config.update('jax_platform_name', 'gpu')
        
        # Warm up
        for _ in range(3):
            _ = jnp.dot(jnp.array(a), jnp.array(b))
        
        # Benchmark
        times_gpu = []
        for i in range(5):
            start = time.time()
            result = jnp.dot(jnp.array(a), jnp.array(b))
            result.block_until_ready()  # Ensure computation completes
            elapsed = time.time() - start
            times_gpu.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
        
        print(f"  GPU Average: {np.mean(times_gpu)*1000:.2f} ms")
        
        # Compare
        if times_cpu and times_gpu:
            speedup = np.mean(times_cpu) / np.mean(times_gpu)
            print(f"  GPU Speedup: {speedup:.2f}x")
            if speedup < 1.0:
                print(f"  ⚠️  GPU is {1/speedup:.2f}x slower than CPU!")

def benchmark_optimization():
    """Benchmark optimization operations (similar to IK)"""
    
    print(f"\n{'='*60}")
    print("Benchmarking optimization operations (similar to IK)")
    print(f"{'='*60}")
    
    # Simple optimization problem (similar to IK)
    def simple_optimization(x):
        return jnp.sum((x - 1.0) ** 2)
    
    # CPU test
    print("\nCPU Optimization Test:")
    jax.config.update('jax_platform_name', 'cpu')
    
    x_init = jnp.zeros(6)  # 6-DOF robot
    
    # Warm up
    for _ in range(3):
        _ = jax.grad(simple_optimization)(x_init)
    
    # Benchmark
    times_cpu = []
    for i in range(10):
        start = time.time()
        grad = jax.grad(simple_optimization)(x_init)
        grad.block_until_ready()
        elapsed = time.time() - start
        times_cpu.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
    
    print(f"  CPU Average: {np.mean(times_cpu)*1000:.2f} ms")
    
    # GPU test
    # if len(jax.devices()) > 1 or 'gpu' in str(jax.devices()):
    print("\nGPU Optimization Test:")
    jax.config.update('jax_platform_name', 'gpu')
    
    # Warm up
    for _ in range(3):
        _ = jax.grad(simple_optimization)(x_init)
    
    # Benchmark
    times_gpu = []
    for i in range(10):
        start = time.time()
        grad = jax.grad(simple_optimization)(x_init)
        grad.block_until_ready()
        elapsed = time.time() - start
        times_gpu.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
    
    print(f"  GPU Average: {np.mean(times_gpu)*1000:.2f} ms")
        
        # Compare
    if times_cpu and times_gpu:
        speedup = np.mean(times_cpu) / np.mean(times_gpu)
        print(f"  GPU Speedup: {speedup:.2f}x")
        if speedup < 1.0:
            print(f"  ⚠️  GPU is {1/speedup:.2f}x slower than CPU!")

def check_jetson_specific():
    """Check Jetson-specific configurations"""
    
    print(f"\n{'='*60}")
    print("Jetson-specific checks")
    print(f"{'='*60}")
    
    # Check environment variables
    import os
    env_vars = [
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Check GPU memory
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"GPU Memory: {result.stdout.strip()}")
    except:
        print("Could not check GPU memory")

if __name__ == "__main__":
    benchmark_jax_operations()
    benchmark_optimization()
    check_jetson_specific()