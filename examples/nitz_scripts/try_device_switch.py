import os, time
# os.environ["JAX_PLATFORMS"] = "cpu"   # comment this out if you just want the block
import jax, jax.numpy as jnp

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0] if jax.devices("gpu") else None

def matmul(a, b):
    return a @ b

key = jax.random.key(0)
A = jax.random.normal(key, (2048, 2048), dtype=jnp.float32)
B = jax.random.normal(key, (2048, 2048), dtype=jnp.float32)

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# jax.config.update('jax_platform_name', 'cpu')              # compile & run on CPU
print(jax.devices())
mm_cpu = jax.jit(matmul)
t0 = time.perf_counter()
mm_cpu(A, B).block_until_ready()
print("CPU time:", time.perf_counter() - t0)

if gpu:
    # os.environ["JAX_PLATFORM_NAME"] = "gpu"
    # jax.config.update('jax_platform_name', 'gpu')
    print(jax.devices())
    mm_gpu = jax.jit(matmul)
    t0 = time.perf_counter()
    mm_gpu(A, B).block_until_ready()
    print("GPU time:", time.perf_counter() - t0)
