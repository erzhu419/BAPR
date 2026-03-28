"""cProfile analysis of one full training iteration with Brax env."""
import cProfile
import pstats
import io
import time
import numpy as np
import os, sys

# CUDA setup
NVIDIA_LIB = None
for p in sys.path:
    candidate = os.path.join(p, "nvidia")
    if os.path.isdir(candidate):
        NVIDIA_LIB = candidate
        break
if NVIDIA_LIB is None:
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia")
        if os.path.isdir(candidate):
            NVIDIA_LIB = candidate
            break
if NVIDIA_LIB is not None:
    for subdir in os.listdir(NVIDIA_LIB):
        lib_path = os.path.join(NVIDIA_LIB, subdir, "lib")
        if os.path.isdir(lib_path):
            os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import jax
import jax.numpy as jnp
from jax_experiments.configs.default import Config
from jax_experiments.algos.resac import RESAC
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.envs.brax_env import BraxNonstationaryEnv

config = Config()
config.algo = 'resac'

# Setup
env = BraxNonstationaryEnv('Hopper-v2', seed=8)
agent = RESAC(env.obs_dim, env.act_dim, config, seed=8)
buf = ReplayBuffer(env.obs_dim, env.act_dim)
tasks = env.sample_tasks(40)
env.set_nonstationary_para(tasks, 100, 10)

print("=== Phase 0: Fill buffer with random data ===")
obs = env.reset()
for i in range(12000):
    a = env.action_space.sample()
    no, r, d, _ = env.step(a)
    buf.push(obs, a, r, no, d, env.current_task_id)
    obs = no if not d else env.reset()
print(f"Buffer size: {buf.size}")

# Warmup JIT
print("\n=== Phase 1: JIT warmup ===")
t0 = time.perf_counter()
stacked = buf.sample_stacked(250, 256)
m = agent.multi_update(stacked)
t1 = time.perf_counter()
print(f"JIT compile + first update: {t1-t0:.1f}s")

# Second warmup (cache stabilization)
stacked = buf.sample_stacked(250, 256)
m = agent.multi_update(stacked)
t2 = time.perf_counter()
print(f"Second update (cache warm): {t2-t1:.1f}s")

# Third (should be fast)
stacked = buf.sample_stacked(250, 256)
m = agent.multi_update(stacked)
t3 = time.perf_counter()
print(f"Third update (steady): {t3-t2:.1f}s")

print("\n=== Phase 2: Profile one full iteration ===")

def one_iteration():
    """Simulate one training iteration."""
    # Part A: Data collection (4000 steps, sequential)
    obs = env.reset()
    ep_rewards = []
    ep_rew = 0.0
    for _ in range(4000):
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        buf.push(obs, action, reward, next_obs, done, env.current_task_id)
        ep_rew += reward
        obs = next_obs
        if done:
            ep_rewards.append(ep_rew)
            ep_rew = 0.0
            obs = env.reset()
    
    # Part B: Gradient updates (250 steps, scan-fused)
    stacked = buf.sample_stacked(250, 256)
    metrics = agent.multi_update(stacked)
    
    # Part C: Evaluation (5 episodes)
    eval_rewards = []
    for ep in range(5):
        env.set_task(tasks[ep % len(tasks)])
        obs = env.reset()
        er = 0.0
        for _ in range(1000):
            action = agent.select_action(obs, deterministic=True)
            obs, r, d, _ = env.step(action)
            er += r
            if d:
                break
        eval_rewards.append(er)
    
    return ep_rewards, metrics, eval_rewards

# Profile
profiler = cProfile.Profile()
t_start = time.perf_counter()
profiler.enable()
ep_r, met, eval_r = one_iteration()
profiler.disable()
t_end = time.perf_counter()

print(f"\nTotal iteration time: {t_end-t_start:.1f}s")
print(f"Train episodes: {len(ep_r)}, Eval mean: {np.mean(eval_r):.1f}")

# Print top 40 by cumulative time
print("\n=== Top 40 by CUMULATIVE time ===")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

# Print top 30 by total time (self time)
print("\n=== Top 30 by SELF time ===")
s2 = io.StringIO()
ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
ps2.print_stats(30)
print(s2.getvalue())

# Also time each phase separately
print("\n=== Phase 3: Time each phase individually ===")
# Collection only
obs = env.reset()
t0 = time.perf_counter()
for _ in range(4000):
    action = agent.select_action(obs, deterministic=False)
    next_obs, reward, done, info = env.step(action)
    buf.push(obs, action, reward, next_obs, done, env.current_task_id)
    obs = next_obs if not done else env.reset()
t1 = time.perf_counter()
print(f"Collection (4000 steps): {t1-t0:.1f}s")

# Collection with random actions (no policy)
obs = env.reset()
t0 = time.perf_counter()
for _ in range(4000):
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    buf.push(obs, action, reward, next_obs, done, env.current_task_id)
    obs = next_obs if not done else env.reset()
t1 = time.perf_counter()
print(f"Collection random (4000 steps, no policy): {t1-t0:.1f}s")

# Env step only (no buffer, no policy)
obs = env.reset()
t0 = time.perf_counter()
for _ in range(4000):
    action = np.random.uniform(-1, 1, 3).astype(np.float32)
    next_obs, reward, done, info = env.step(action)
    obs = next_obs if not done else env.reset()
t1 = time.perf_counter()
print(f"Env step only (4000 steps): {t1-t0:.1f}s")

# Sample + update
t0 = time.perf_counter()
stacked = buf.sample_stacked(250, 256)
t1 = time.perf_counter()
m = agent.multi_update(stacked)
t2 = time.perf_counter()
print(f"Sample stacked: {(t1-t0)*1000:.0f}ms")
print(f"Update (scan): {(t2-t1)*1000:.0f}ms")

# Eval only
t0 = time.perf_counter()
eval_rewards = []
for ep in range(5):
    env.set_task(tasks[ep % len(tasks)])
    obs = env.reset()
    er = 0.0
    for _ in range(1000):
        action = agent.select_action(obs, deterministic=True)
        obs, r, d, _ = env.step(action)
        er += r
        if d:
            break
    eval_rewards.append(er)
t1 = time.perf_counter()
print(f"Eval (5 episodes): {t1-t0:.1f}s")
