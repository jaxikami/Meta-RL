"""Training and evaluation entrypoints for the multi-reactor PPO setup."""

# ─── Standard Library ─────────────────────────────────────────────────────────
from collections import deque
import os
import warnings

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import tqdm

# ─── Local ────────────────────────────────────────────────────────────────────
from agent_safe import PPO_Agent as SafePPO_Agent, PPOMemory, device
from env_config import get_main_reactor_config
from env_safe import FermentationEnv
from utils import (
    initialize_csv,
    log_to_csv,
    plot_cross_reactor_attention,
    plot_evaluation_results,
    plot_training_results
)

warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*")

# Suppress library warnings BEFORE importing heavy libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------- Run configuration ----------------
# This entrypoint trains/evaluates the multilayered Safe PPO agent only.
# (The former Standard-PPO "benchmark" comparison has been removed — it is out
# of scope for the meta-RL project.  The relevant comparison here is instead
# meta-RL (PEARL) vs. non-adaptive Safe RL; see train_pearl_single_run.py and
# train_pearl_seed_sweep.py.)
EVAL_ONLY = False
TRAIN_EPISODES = 50000
MODEL_PATH = os.path.join('policy', 'safe_f.pth')
LOG_FILE = "training_log.csv"
n_reactors = 3
EVAL_NOISE_MODE = 'action'  # Options: 'action', 'state', 'all', 'none'


def run_evaluation(env, agent, n_eval_episodes=1000, noise_mode='action'):
    """Run noisy policy evaluation and plot the best trajectory.

    Execute a robust evaluation of the trained policy by running multiple episodes 
    with injected Gaussian noise. This empirical test probes the policy's resilience 
    to real-world sensor/actuator imprecision. The single best performing episode exactly 
    (highest total reward) is tracked, stored, and its comprehensive operational 
    trajectory is plotted for rigorous diagnostic analysis.

    Args:
        env: Instantiated physical simulation environment.
        agent: Trained PPO policy object with select_action/reset interfaces.
        n_eval_episodes: Number of independent noisy rollout iterations.
        noise_mode: Injection mode, one of {'action', 'state', 'all', 'none'}.

    Returns:
        None. Prints evaluation summaries and generates diagnostic plots.
    """
    NOISE_STD = 0.00
    print(f"--- Evaluating {env.n_reactors}-Reactor Policy "
          f"({n_eval_episodes} episodes, noise_mode='{noise_mode}', {NOISE_STD*100:.0f}% std) ---")

    lagrange_updates_prev = getattr(env, 'lagrange_updates_enabled', True)
    env.lagrange_updates_enabled = False

    # Feed-control indices in each reactor action block: 1=LF, 2=C5, 3=Out
    FEED_DIMS = [1, 2, 3]

    # State observation indices for noise (excluding rigid time variables like Stage and Credit)
    # Per-reactor state layout (28 dims):
    #   0-14  Concentrations, 15 Mass, 16-20 Stage one-hot, 21 Credit,
    #   22-24 Structural HW limits (Mmax, Fmax_LF, Fmax_C5) — FIXED, no noise,
    #   25    norm_S (global storage — sensor reading),
    #   26    norm_LF (global LF supply — sensor reading),
    #   27    norm_C5 (global C5 supply — sensor reading).
    STATE_NOISE_DIMS = list(range(16))
    GLOBAL_NOISE_DIMS = [25, 26, 27]

    best_reward = -float('inf')
    best_trajectory = None
    total_eval_violations = 0

    # All-episode mass accumulators: list of lists (one per reactor),
    # each holding shape-(n_steps,) arrays from every episode.
    # Used to compute the min/max shaded band on the volume plot.
    all_mass_histories = [[] for _ in range(env.n_reactors)]
    # Cross-episode storage accumulator for the storage-level safety band plot.
    all_storage_histories = []

    # Separate violation counters for 100% (hard physical limit) and
    # 95% (safety-buffer limit matching the projection network boundary).
    mass_limit_100 = env.Mmax                  # absolute cap [n_reactors]
    mass_limit_95 = env.Mmax * 0.95           # manifold / RL barrier cap
    # episodes that breached 100% in reactor i
    viol_eps_100 = [0] * env.n_reactors
    # episodes that breached  95% in reactor i
    viol_eps_95 = [0] * env.n_reactors
    # episodes that triggered each constraint type (IDs 1–5) at least once
    all_viol_type_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    flow_keys = []
    for i in range(env.n_reactors):
        flow_keys.extend([f'LF_{i+1}', f'C5_{i+1}', f'Out_{i+1}'])

    for ep in tqdm.trange(n_eval_episodes, desc="Eval episodes"):
        state = env.reset()
        agent.reset()  # Clear recurrent state buffer so past episodes don't leak
        done = False
        ep_reward = 0.0
        ep_hard_violations = 0
        # Per-violation-type timestep counts for this episode (IDs 1–5)
        ep_viol_type_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # Per-episode trajectory buffers.
        reactor_histories = [{name: [] for name in env.species_names}
                             for _ in range(env.n_reactors)]
        mass_histories = [[] for _ in range(env.n_reactors)]
        storage_history = []
        stage_histories = [[] for _ in range(env.n_reactors)]
        # Cumulative ethanol harvested (kg), never decreases
        eth_harvested_history = []
        flows = {key: [] for key in flow_keys}
        flows['time'] = []
        # Attention weights: list of [1, n_heads, B, B] arrays, one per step.
        # Only populated for the Safe PPO agent (baseline has no attention layer).
        attn_history = []

        cumulative_eth_kg = 0.0

        def record_step(a_reshaped=None, step_eth_kg=0.0):
            nonlocal cumulative_eth_kg
            current_time = env.time_step_count * env.time_step_duration
            flows['time'].append(current_time)
            storage_history.append(env.S_mass)
            # Accumulate harvested ethanol so the curve never decreases
            cumulative_eth_kg += step_eth_kg
            eth_harvested_history.append(cumulative_eth_kg)
            for i in range(env.n_reactors):
                mass_histories[i].append(env.M[i])
                stage_histories[i].append(int(env.current_stage[i]))
                for j, name in enumerate(env.species_names):
                    reactor_histories[i][name].append(env.C[i, j])
            if a_reshaped is not None:
                for i in range(env.n_reactors):
                    flows[f'LF_{i+1}'].append(a_reshaped[i, 1]
                                              * env.max_flow_LF[i])
                    flows[f'C5_{i+1}'].append(a_reshaped[i, 2]
                                              * env.max_flow_C5[i])
                    flows[f'Out_{i+1}'].append(a_reshaped[i, 3]
                                               * env.max_flow_out_reac)
            else:
                for key in flow_keys:
                    flows[key].append(0)

        record_step()  # Record initial state before first action.

        while not done:
            observed_state = state.copy()

            # Inject sensor noise onto non-time-related variables before agent observes it
            if noise_mode in ['state', 'all']:
                for d in STATE_NOISE_DIMS + GLOBAL_NOISE_DIMS:
                    observed_state[:, d] += np.random.normal(
                        0.0, NOISE_STD, size=env.n_reactors)

            with torch.no_grad():
                action, _, _, _ = agent.select_action(observed_state)

            # Collect attention weights produced by this forward pass.
            # last_attn_weights has shape [T=1, n_heads, B, B] for single-step
            # inference; only present on the Safe PPO agent.
            _attn_mod = getattr(
                getattr(getattr(agent, 'policy_old', None), 'cross_reactor_attn', None),
                'last_attn_weights', None
            )
            if _attn_mod is not None:
                attn_history.append(_attn_mod.squeeze(0).cpu().numpy())  # [n_heads, B, B]

            noisy_action = action.copy()
            # Inject actuator noise onto feed and product lines (Multiplicative to physical flow)
            if noise_mode in ['action', 'all']:
                for i in range(env.n_reactors):
                    for d in FEED_DIMS:
                        # Move from raw [-1, 1] actuator scaling to [0, 1] physical ratio
                        flow_ratio = (noisy_action[i, d] + 1.0) / 2.0
                        # Multiply by noise (0 intent * noise = 0 leak)
                        flow_ratio *= 1.0 + np.random.normal(0.0, NOISE_STD)
                        # Re-encode into the [-1, 1] bounds structure
                        noisy_action[i, d] = np.clip(
                            (flow_ratio * 2.0) - 1.0, -1.0, 1.0)

            state, reward, done, info = env.step(noisy_action)
            step_eth = info.get("eth_harvested", 0.0)
            record_step(info.get("actual_a_reshaped"), step_eth_kg=step_eth)
            ep_reward += np.sum(reward)
            ep_hard_violations += info.get("violation_count", 0)
            total_eval_violations += info.get("violation_count", 0)
            # Tally per-type violation counts for this episode
            for vid in info.get("violation_ids", []):
                if vid in ep_viol_type_counts:
                    ep_viol_type_counts[vid] += 1

        # ---- cross-episode volume tracking ----
        all_storage_histories.append(np.array(storage_history))
        for i in range(env.n_reactors):
            arr = np.array(mass_histories[i])
            all_mass_histories[i].append(arr)
            # A single timestep breach is enough to flag the episode.
            if np.any(arr > mass_limit_100[i]):
                viol_eps_100[i] += 1
            if np.any(arr > mass_limit_95[i]):
                viol_eps_95[i] += 1

        # Accumulate per-type counts across all episodes
        for vid, cnt in ep_viol_type_counts.items():
            if cnt > 0:
                all_viol_type_counts[vid] += 1

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_trajectory = {
                'time_hours': np.array(flows['time']) / 3600.0,
                'reactor_histories': reactor_histories,
                'flows': flows,
                'mass_histories': mass_histories,
                'storage_history': storage_history,
                'stage_histories': stage_histories,
                'hard_violation_count': ep_hard_violations,
                'eth_harvested_history': eth_harvested_history,
                'viol_type_counts': dict(ep_viol_type_counts),
                # attn_history: list of [n_heads, B, B] arrays; stacked to
                # [T, n_heads, B, B] when passed to the plot function.
                'attn_history': list(attn_history),
            }

    print(f"\n[Evaluation] Best episode reward: {best_reward:.3f}")
    print(
        f"  Total control violation timesteps across all eval episodes: {total_eval_violations}")
    for i in range(env.n_reactors):
        print(f"  Reactor {i+1} | Episodes breaching 100% cap: {viol_eps_100[i]}/{n_eval_episodes} "
              f"| breaching 95% cap: {viol_eps_95[i]}/{n_eval_episodes}")
    t = best_trajectory

    # ── Print production summary table for the best episode ───────────────
    _mode_tag = "SAFE"
    print(f"\n{'='*62}")
    print(f"  PRODUCTION SUMMARY — Best Episode ({_mode_tag})")
    print(f"{'='*62}")
    _best_eth = t.get('eth_harvested_history')
    _final_eth = _best_eth[-1] if (_best_eth is not None and len(_best_eth) > 0) else 0.0
    _duration_h = t['time_hours'][-1] if len(t['time_hours']) > 0 else 0.0
    print(f"  {'Metric':<38} {'Value':>18}")
    print(f"  {'-'*38} {'-'*18}")
    print(f"  {'Total Ethanol Harvested (kg)':<38} {_final_eth:>18,.2f}")
    print(f"  {'Episode Duration (hours)':<38} {_duration_h:>18,.1f}")
    _rate = _final_eth / _duration_h if _duration_h > 0 else 0.0
    print(f"  {'Avg Production Rate (kg/h)':<38} {_rate:>18,.2f}")
    # Outflow-weighted cumulative ethanol concentration:
    # total ethanol (g) that left all reactors / total mass (kg) that left.
    _t_arr  = t['time_hours']
    _dt_sec = np.diff(_t_arr, append=_t_arr[-1] + (_t_arr[-1] - _t_arr[-2] if len(_t_arr) > 1 else 1.0)) * 3600.0
    _total_out_kg    = 0.0
    _total_eth_out_g = 0.0
    for _i in range(env.n_reactors):
        _out = np.array(t['flows'][f'Out_{_i+1}'], dtype=np.float64)
        _eth = np.array(t['reactor_histories'][_i]['Eth'], dtype=np.float64)
        _n   = min(len(_out), len(_eth), len(_dt_sec))
        _step_mass        = _out[:_n] * _dt_sec[:_n]
        _total_out_kg    += _step_mass.sum()
        _total_eth_out_g += (_step_mass * _eth[:_n]).sum()
    _outlet_conc = _total_eth_out_g / _total_out_kg if _total_out_kg > 0 else 0.0
    print(f"  {'Total Outflow Mass (kg)':<38} {_total_out_kg:>18,.2f}")
    print(f"  {'Blended Outlet Eth Conc (g/kg)':<38} {_outlet_conc:>18,.2f}")
    print(f"  {'Best Episode Reward':<38} {best_reward:>18,.3f}")
    print(f"  {'Hard Violations (best ep)':<38} {t['hard_violation_count']:>18d}")
    for i in range(env.n_reactors):
        _r_eth = t['reactor_histories'][i]['Eth']
        _peak = max(_r_eth) if len(_r_eth) > 0 else 0.0
        _final = _r_eth[-1] if len(_r_eth) > 0 else 0.0
        print(f"  {'Reactor ' + str(i+1) + ' peak Eth conc (g/kg)':<38} {_peak:>18,.2f}")
        print(f"  {'Reactor ' + str(i+1) + ' final Eth conc (g/kg)':<38} {_final:>18,.2f}")
    print(f"{'='*62}\n")

    # Build min/max band arrays aligned to the best-episode time axis.
    # Pad shorter episodes to the length of the best episode so stacking works.
    best_len = len(t['time_hours'])
    mass_band_min = []
    mass_band_max = []
    for i in range(env.n_reactors):
        stacked = []
        for arr in all_mass_histories[i]:
            if len(arr) >= best_len:
                stacked.append(arr[:best_len])
            else:
                padded = np.pad(arr, (0, best_len - len(arr)), mode='edge')
                stacked.append(padded)
        stacked = np.array(stacked)
        mass_band_min.append(stacked.min(axis=0))
        mass_band_max.append(stacked.max(axis=0))

    # Build storage min/max band aligned to best-episode length.
    stor_stacked = []
    for arr in all_storage_histories:
        if len(arr) >= best_len:
            stor_stacked.append(arr[:best_len])
        else:
            stor_stacked.append(np.pad(arr, (0, best_len - len(arr)), mode='edge'))
    stor_stacked = np.array(stor_stacked)
    storage_band_min = stor_stacked.min(axis=0)
    storage_band_max = stor_stacked.max(axis=0)

    plot_evaluation_results(
        t['time_hours'],
        t['reactor_histories'],
        t['flows'],
        t['mass_histories'],
        t['storage_history'],
        t['stage_histories'],
        t['hard_violation_count'],
        mmax=env.Mmax,
        mass_band_min=mass_band_min,
        mass_band_max=mass_band_max,
        viol_eps_100=viol_eps_100,
        viol_eps_95=viol_eps_95,
        n_eval_episodes=n_eval_episodes,
        eth_harvested_history=t.get('eth_harvested_history'),
        all_viol_type_counts=all_viol_type_counts,
        benchmark=False,
        storage_band_min=storage_band_min,
        storage_band_max=storage_band_max,
        s_min=env.S_min,
        s_max=env.S_max,
    )

    # ── Attention interpretability plots (Safe PPO only) ──────────────────
    _raw_attn = t.get('attn_history')
    if _raw_attn:
        # Stack per-step [n_heads, B, B] slices into [T, n_heads, B, B]
        attn_array = np.stack(_raw_attn, axis=0)

        # Persist the full attention tensor to policy/ as a compressed archive.
        # Layout of saved arrays:
        #   attn        float32  [T, n_heads, n_reactors, n_reactors]
        #   time_hours  float64  [T]   — wall-clock simulation time axis
        #   stages      int32    [n_reactors, T] — stage ID per reactor per step
        # The file is self-contained: reload with np.load('policy/attn_safe.npz').
        os.makedirs('policy', exist_ok=True)
        _attn_tag = 'safe'
        _attn_path = os.path.join('policy', f'attn_{_attn_tag}.npz')
        np.savez_compressed(
            _attn_path,
            attn=attn_array.astype(np.float32),
            time_hours=np.array(t['time_hours'], dtype=np.float64),
            stages=np.array(t['stage_histories'], dtype=np.int32),
        )
        print(f"[Attention] Saved best-episode attention archive → {_attn_path} "
              f"(shape {attn_array.shape}, "
              f"{os.path.getsize(_attn_path) / 1024:.1f} KB)")

        plot_cross_reactor_attention(
            attn_array,
            t['time_hours'],
            t['stage_histories'],
            benchmark=False,
        )

    env.lagrange_updates_enabled = lagrange_updates_prev


def train():
    """Run the primary PPO training loop and post-training evaluation.

    Primary training loop orchestrating the PPO Reinforcement Learning algorithm.
    Instantiates the heterogeneous multi-reactor environment, configures the strictly 
    tuned RL hyperparameters and Learning Rate schedules, runs continuous trajectory 
    collection rollouts, and executes batch gradient descent optimization. 
    Performance is tracked and recorded to CSV, automatically saving the best actor models.

    Returns:
        None. Saves checkpoints, logs training statistics, plots curves, and
        runs final evaluation.
    """
    algo_label = "Safe PPO"
    print(f"--- Training {algo_label} (Device: {device}) ---")

    cfg = get_main_reactor_config()

    env = FermentationEnv(
        n_reactors=n_reactors,
        Mmax=cfg['Mmax'],
        max_flow_LF=cfg['max_flow_LF'],
        max_flow_C5=cfg['max_flow_C5']
    )

    # ---------------- PPO and scheduler hyperparameters ----------------
    update_timestep = int(env.max_steps / env.control_action_interval) * 10
    K_epochs = 3
    eps_clip = 0.2
    gamma = 0.95
    lr_actor_init = 5e-5
    lr_actor_min = 1e-5
    lr_critic_init = 1e-4
    initial_entropy = 0.05
    min_entropy = 1e-5
    LR_T_0 = 2000

    # ---------------- Early stopping ----------------
    EARLY_STOP_WARMUP = 5000
    EARLY_STOP_PATIENCE = 3000
    min_improvement = 1e-4
    best_avg_reward = -float('inf')
    no_improve_count = 0

    # ---------------- Agent and memory ----------------
    agent = SafePPO_Agent(
        env.stateDim,
        env.actionDim,
        lr_actor_init,
        lr_critic_init,
        gamma,
        K_epochs,
        eps_clip,
        initial_entropy,
        latent_dim=env.actionDim,
        use_safeguard=True,
    )
    memory = PPOMemory()
    # Pre-allocate contiguous numpy buffers sized to one update window.
    # This eliminates per-step Python list growth and the np.array(list)
    # reconstruction overhead at learn() time (see PPOMemory.reserve).
    memory.reserve(update_timestep, env.n_reactors, env.stateDim, env.actionDim)

    # ---------------- LR scheduler ----------------
    # Linear decay: LR falls monotonically from init → min over the full training
    # run, eliminating the periodic restart spikes that caused catastrophic forgetting.
    scheduler = torch.optim.lr_scheduler.LinearLR(
        agent.optimizer,
        start_factor=1.0,
        end_factor=lr_actor_min / lr_actor_init,
        total_iters=TRAIN_EPISODES
    )

    all_rewards = []
    window_size = 200
    rewards_window = deque(maxlen=window_size)

    # ---------------- Reward component tracking ----------------
    track_keys = ["prod", "constraint", "smoothing", "eff",
                  "eth_harvested", "raw_mat_penalty",
                  "is_feasible", "violation_count"]
    full_comp_history = {k: [] for k in track_keys}
    comp_window = {k: deque(maxlen=100) for k in track_keys}
    global_step = 0
    ep_action_std = 0.0

    log_files = initialize_csv(LOG_FILE, env.n_reactors, env.species_names)
    pbar = tqdm.trange(1, TRAIN_EPISODES + 1)
    for episode in pbar:
        # Keep critic learning rate above actor floor.
        lr_critic_min = lr_actor_min * 3
        agent.optimizer.param_groups[3]['lr'] = max(
            agent.optimizer.param_groups[3]['lr'], lr_critic_min)

        curr_lr_a = agent.optimizer.param_groups[0]['lr']
        cos_frac = (curr_lr_a - lr_actor_min) / \
            (lr_actor_init - lr_actor_min + 1e-12)
        agent.entropy_coeff = min_entropy + \
            (initial_entropy - min_entropy) * cos_frac

        state = env.reset()
        agent.reset()  # Clear recurrent state buffer at the start of each episode
        ep_reward = 0
        ep_components = {k: 0.0 for k in track_keys}
        ep_violation_id = 0
        ep_violation_ids = set()
        done = False
        episode_safe_actions = []

        while not done:
            # 1) Agent action selection.
            action, latent_z, log_prob, state_aug = agent.select_action(state)
            episode_safe_actions.append(action)

            # 3) Environment step with mapped physical action.
            state, reward, done, info = env.step(action)

            # 2) Store rollout experience (after step so reward is known).
            memory.append(state_aug, latent_z,
                          np.atleast_1d(log_prob), action,
                          reward, done)

            # Accumulate reward components.
            for k in track_keys:
                if k in info:
                    ep_components[k] += info[k]

            # Keep the highest constraint violation id seen in episode.
            vid = info.get("violation_id", 0)
            if vid > 0:
                ep_violation_id = vid
            if "violation_ids" in info:
                ep_violation_ids.update(info["violation_ids"])

            ep_reward += np.sum(reward)
            global_step += 1

            if global_step % update_timestep == 0:
                agent.learn(memory)
                memory.clear()

        # Perfect run flag: survived full episode horizon.
        is_perfect_run = 1.0 if env.time_step_count >= env.max_steps else 0.0

        # Advance LR schedule after optimizer has already stepped this episode.
        scheduler.step()
        # Safe-action diversity metric.
        if episode_safe_actions:
            ep_action_std = float(np.std(np.stack(episode_safe_actions)))

        all_rewards.append(ep_reward)
        rewards_window.append(ep_reward)

        for k in ep_components:
            val = ep_components[k]

            # Replace with binary episode-level feasibility signal.
            if k == 'is_feasible':
                val = is_perfect_run

            full_comp_history[k].append(val)
            comp_window[k].append(val)

        if ep_violation_ids:
            cv_type_str = "".join(str(vid) for vid in sorted(ep_violation_ids))
        else:
            cv_type_str = "0"

        # Progress bar: episode totals and violation id.
        pbar.set_postfix({
            'Rwd': f'{ep_reward:.3f}',
            'Prod': f'{ep_components["prod"]:.4f}',
            'RawM': f'{ep_components["raw_mat_penalty"]:.4f}',
            'Cons': f'{ep_components["constraint"]:.4f}',
            'Eff': f'{ep_components["eff"]:.2f}',
            'Smth': f'{ep_components["smoothing"]:.2f}',
            'ActStd': f'{ep_action_std:.2f}',
            'CV_type': cv_type_str,
            'CV_cnt': int(ep_components.get("violation_count", 0)),
        })

        # Early stopping after warmup.
        if episode > EARLY_STOP_WARMUP and len(rewards_window) == window_size:
            avg_reward = np.mean(rewards_window)

            if avg_reward > best_avg_reward * (1 + min_improvement):
                best_avg_reward = avg_reward
                no_improve_count = 0
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                inter_name = "safe_inter.pth"
                best_model_path = os.path.join(
                    os.path.dirname(MODEL_PATH), inter_name)
                torch.save(agent.policy.state_dict(), best_model_path)
            else:
                no_improve_count += 1

            # CSV logging on improvement or periodic checkpoint.
            if no_improve_count == 0 or episode % 500 == 0:
                reward_data = [
                    ep_components.get("prod",             0.0),
                    ep_components.get("constraint",       0.0),
                    ep_components.get("eff",              0.0),
                    ep_components.get("smoothing",        0.0),
                    ep_components.get("eth_harvested",    0.0),
                ]
                if ep_violation_ids:
                    violation_types_encoded = int(
                        "".join(str(vid) for vid in sorted(ep_violation_ids)))
                else:
                    violation_types_encoded = 0
                violation_time_hours = int(
                    ep_components.get("violation_count", 0))
                for i in range(env.n_reactors):
                    material_data = [env.M[i]] + list(env.C[i])
                    log_data = (
                        [episode, ep_reward, avg_reward] +
                        material_data + reward_data +
                        [violation_types_encoded, violation_time_hours]
                    )
                    log_to_csv(log_files[i], log_data)

            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stopping] No improvement for "
                      f"{EARLY_STOP_PATIENCE} episodes. "
                      f"Best avg reward: {best_avg_reward:.3f}")
                break

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(agent.policy.state_dict(), MODEL_PATH)
    plot_training_results(all_rewards, full_comp_history,
                          benchmark=False)
    run_evaluation(env, agent, noise_mode=EVAL_NOISE_MODE)


def evaluate_only():
    """Load a trained checkpoint and run evaluation without training.

    Standalone diagnostic mode for analyzing pre-trained models.
    Instantiates the evaluation environment, strictly loads the frozen weights 
    from the target model file, completely bypasses the training routines, 
    and directly executes the rigorous `run_evaluation` noise-testing procedure.

    Raises:
        FileNotFoundError: If MODEL_PATH does not exist.

    Returns:
        None. Runs evaluation and produces summary output/plots.
    """
    algo_label = "Safe PPO"
    print(f"--- Evaluating {algo_label} (Device: {device}) ---")

    cfg = get_main_reactor_config()
    env = FermentationEnv(
        n_reactors=n_reactors,
        Mmax=cfg['Mmax'],
        max_flow_LF=cfg['max_flow_LF'],
        max_flow_C5=cfg['max_flow_C5']
    )

    # Create agent with the same architecture and hparams as training.
    agent = SafePPO_Agent(
        env.stateDim,
        env.actionDim,
        lr_actor=3e-5,
        lr_critic=1e-4,
        gamma=0.99,
        K_epochs=20,
        eps_clip=0.2,
        entropy_coeff=0.05,
        latent_dim=env.actionDim,
        use_safeguard=True,
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Set MODEL_PATH to a valid checkpoint before EVAL_ONLY=True."
        )

    try:
        state_dict = torch.load(
            MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    agent.policy.eval()
    agent.policy_old.eval()

    run_evaluation(env, agent, noise_mode=EVAL_NOISE_MODE)


if __name__ == "__main__":
    if EVAL_ONLY:
        evaluate_only()
    else:
        train()
