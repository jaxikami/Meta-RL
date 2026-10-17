"""Safe fermentation environment with Lagrangian constraint shaping.

This module defines FermentationEnv, the environment variant used with the
safe PPO agent. It includes biochemical simulation, staged process logic, and
constraint-aware reward shaping with both soft buffers and adaptive Lagrange
multipliers.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np

# ─── Local ────────────────────────────────────────────────────────────────────
from env_config import (
    INITIAL_LF, INITIAL_C5, S_MAX, S_MIN, S_OUT_RATE, S_INITIAL,
    SPECIES_NAMES, SPECIES_IDX, C_C5_BASE, C_LF_BASE,
    BASE_CREDITS, TIME_STEP_DURATION, MAX_FLOW_OUT_REAC, C_MAX_LIMITS,
    nominal_task, sample_task,
)
from env_core import (
    calculate_rates_numba, _ETA, _PH_G_FACTOR, _PH_X_FACTOR,
    ode_system, integrate_rk4, PARAM_DIM,
)

# =============================================================================
# ENVIRONMENT CLASS DEFINITION
# =============================================================================


class FermentationEnv:
    """Full RL environment wrapping biochemical physics and safe shaping.

    Manages reactor dynamics, normalized observation construction, process
    stage progression, and constraint-aware rewards including Lagrangian
    penalties.

    Attributes:
        control_action_interval: Number of inner 1-hour steps per outer action.
        n_reactors: Number of parallel reactors.
        stateDim: Per-reactor observation width (includes shared global fields).
        actionDim: Per-reactor action width.
        max_steps: Episode horizon in inner steps.
    """
    @staticmethod
    def _array_expand(arr, default_value, size):
        """Expand scalar/short arrays to match reactor count.

        Args:
            arr: Optional input array-like to expand.
            default_value: Fill value when arr is None or shorter than size.
            size: Target output length.

        Returns:
            NumPy array of length size with dtype float64.

        Raises:
            ValueError: If arr has more elements than size.
        """
        if arr is None:
            return np.full(size, default_value, dtype=np.float64)
        arr = np.array(arr, dtype=np.float64)
        if arr.size > size:
            raise ValueError(
                f"Array length {arr.size} exceeds number of reactors {size}.")
        if arr.size < size:
            padding = np.full(size - arr.size, default_value, dtype=np.float64)
            arr = np.concatenate([arr, padding])
        return arr

    def __init__(self, n_reactors=3, Mmax=None, max_flow_LF=None, max_flow_C5=None, control_action_interval=5):
        """Initialise the safe fermentation environment.

        Args:
            n_reactors: Number of reactors to simulate in parallel.
            Mmax: Optional per-reactor max mass values.
            max_flow_LF: Optional per-reactor LF max flow rates.
            max_flow_C5: Optional per-reactor C5 max flow rates.
            control_action_interval: Number of inner simulation steps to repeat
                each outer agent action.
        """
        # ---------------- Core dimensions ----------------
        self.control_action_interval = control_action_interval
        self.n_reactors = n_reactors
        self.max_stage_time = 150.0
        self.n_stage = 5

        # RL Agent Input Dimensionality Layout:
        # [15 Chemical Channels + 1 Mass + 5 Stage One-Hot + 1 Time Credit + 3 Structural Limits] = 25 per reactor.
        # Plus 3 global variables: 1 Central Storage Mass, 1 LF supply, 1 C5 supply.
        self.stateDim = 28

        # ── Uncertainty / task context (proposal §2.3) ───────────────────────
        # The latent process parameters theta enter the physics through
        # self.params (kinetics/enzyme/toxin, length PARAM_DIM) and the feed
        # compositions self.C_C5 / self.C_LF (feedstock).  A task is sampled once
        # per episode in reset() and held constant for the whole episode.
        #
        #   randomize_tasks : if True, reset() draws theta ~ p(theta); if False
        #                     (default) the env runs on the nominal theta_0, which
        #                     preserves the original single-regime Safe-RL setup
        #                     and is what the non-adaptive baseline trains on.
        #   fixed_task      : when set, reset() always applies this exact task
        #                     (used to replay specific in-distribution / OOD tasks
        #                     during evaluation).
        self.randomize_tasks = False
        self.fixed_task = None
        self.task_rng = np.random.default_rng()
        self.current_task = None
        # Disturbance schedule (proposal §3.2 stress tests): list of dicts, each
        # {kind, onset, magnitude, fired}.  Applied inside _step_internal.
        self._disturbances = []

        # Apply the nominal task so self.params / feeds exist before reset().
        self.params = np.empty(PARAM_DIM, dtype=np.float64)
        self._apply_task(nominal_task())

        # RL Agent Output Dimensionality:
        # 4 actions per reactor: [Time Multiplier, Feed LF, Feed C5, Outflow Unload]
        self.actionDim = 4
        self.max_flow_out_reac = MAX_FLOW_OUT_REAC

        # ---------------- Global Resource Pool ----------------
        self.initial_lf = INITIAL_LF
        self.initial_c5 = INITIAL_C5

        # Central Storage Hardware Specifications
        self.S_max = S_MAX
        self.S_min = S_MIN
        self.S_out_rate = S_OUT_RATE
        self.S_initial = S_INITIAL

        # ---------------- Reactor Limits ----------------
        self.Mmax = self._array_expand(Mmax, 220000.0, self.n_reactors)
        self.max_flow_LF = self._array_expand(
            max_flow_LF, (2487/3600), self.n_reactors)
        self.max_flow_C5 = self._array_expand(
            max_flow_C5, (628/3600), self.n_reactors)

        # ---------------- Tracked Species Definition ----------------
        self.species_names = SPECIES_NAMES
        self.idx = SPECIES_IDX

        # ---------------- Feed Flow Properties ----------------
        self.C_C5 = C_C5_BASE.copy()
        self.C_LF = C_LF_BASE.copy()

        # ---------------- Time Credit and Flow Integration ----------------
        self.base_credits = BASE_CREDITS.copy()
        self.time_step_duration = TIME_STEP_DURATION

        # RK4 sub-steps per 1-hour control tick.
        # 30 sub-steps (dt=120 s) is numerically stable for the fermentation
        # kinetics (fastest rates ~0.07/h → Δc < 0.003 per sub-step at dt=120s)
        # and halves the Numba ODE integration cost vs the original 60 sub-steps.
        self.n_inner_steps = 30
        self.smoothing_coef = 0.005

        # ── Lagrangian Multiplier Constraint Enforcement ─────────────────────
        # Each physical constraint (reactor overflow/underflow, storage overflow/
        # underflow) has a dedicated Lagrangian multiplier λ that grows when the
        # constraint is violated and decays when satisfied.  The penalty injected
        # into the reward = λ × violation_magnitude, giving the PPO policy gradient
        # an unambiguous negative signal proportional to both the severity and
        # persistence of violations.
        #
        # Two gradient-signal mechanisms work in concert:
        #   1. λ-weighted hard penalty — fires when constraint is actively violated;
        #      λ accumulates across episodes so repeated violations amplify.
        #   2. Buffer-zone soft wall  — λ-independent quadratic ramp that activates
        #      BEFORE the hard constraint, providing smooth early-warning gradient
        #      so the policy steers away before hitting the cliff.
        self.lagrange_lr = 0.5             # Δλ = lr × violation per violated step
        self.lagrange_decay = 0.5 / 600.0  # Per-step exponential decay when satisfied
        self.lagrange_target_slack = 0.0   # Zero tolerance: any violation grows λ
        self.lagrange_max = 10.0           # Upper clamp prevents runaway λ

        # Reactor dry floor: physical minimum mass (kg).  Critically, this is the
        # normalization denominator for underflow violations so the gradient signal
        # spans [0, 1] instead of being crushed by division by Mmax (~220,000).
        self.DRY_FLOOR = 10.0

        # Buffer-zone soft-wall coefficients.  A quadratic penalty ramps smoothly
        # from 0 at the buffer edge to BUFFER_COEF at the hard constraint boundary.
        # Gradient = 2·BUFFER_COEF·depth / zone_width — smooth, nonzero, monotonic.
        # This is λ-independent: fires every step regardless of violation history,
        # ensuring PPO gradient always has a nonzero constraint-aware signal.
        self.BUFFER_COEF = 0.02
        self.OVERFLOW_BUFFER_FRAC = 0.10        # Activates at 90% Mmax
        self.UNDERFLOW_BUFFER_KG = 100.0        # Activates at M < 100 kg
        self.STORAGE_OVERFLOW_BUFFER_FRAC = 0.10  # Activates at 90% S_max
        # Activates at S < 8 × S_min (~10% of S_max)
        self.STORAGE_UNDERFLOW_BUFFER_MULT = 8.0

        self.storage_overflow_lambda = 0.0
        self.storage_underflow_lambda = 0.0
        self.reactor_overflow_lambda = np.zeros(self.n_reactors)
        self.reactor_underflow_lambda = np.zeros(self.n_reactors)

        self.lagrange_updates_enabled = True
        self.reset_lambdas_per_episode = False

        # Episode-end numerical stability guardrail via mass conservation residual.
        self.mass_balance_abs_tol = 5.0
        self.mass_balance_rel_tol = 1e-6

        self.max_steps = 600

        self.reset()

    def reset(self):
        """Reset biological state, resources, and episode bookkeeping.

        Hard-resets the biological state and flow tracking for a new episode.

        Returns:
            Normalized observation array of shape [n_reactors, 28].
        """
        self.time_step_count = 0

        # 0. Task selection (uncertainty).  Priority: fixed_task > randomize > nominal.
        if self.fixed_task is not None:
            self._apply_task(self.fixed_task)
        elif self.randomize_tasks:
            self._apply_task(sample_task(self.task_rng))
        else:
            self._apply_task(nominal_task())
        # Re-arm any scheduled mid-episode disturbances for the new episode.
        for d in self._disturbances:
            d["fired"] = False

        # 1. Global Resource Initialization
        self.total_lf = self.initial_lf
        self.total_c5 = self.initial_c5
        self.S_mass = self.S_initial

        # 2. Per-episode Lagrange multiplier reset
        if self.reset_lambdas_per_episode:
            self.storage_overflow_lambda = 0.0
            self.storage_underflow_lambda = 0.0
            self.reactor_overflow_lambda = np.zeros(self.n_reactors)
            self.reactor_underflow_lambda = np.zeros(self.n_reactors)

        # 3. Reactor State Initialization
        self.M = np.full(self.n_reactors, 10000.0)
        self.C = np.zeros((self.n_reactors, len(self.species_names)))

        self.current_stage = np.zeros(self.n_reactors, dtype=np.int32)
        self.stage_time_remaining = np.zeros(self.n_reactors, dtype=np.float64)

        # 4. Inoculation Setup (Initial Yeast Loading)
        init_cell_conc = (1000 * 147.0) / 10000.0
        self.C[:, self.idx['Cell']] = init_cell_conc

        # 5. Tracking and Housekeeping
        self.prev_flow_action = np.zeros(self.n_reactors * 3)
        self.stage_credits = np.zeros(self.n_reactors, dtype=np.float64)
        for i in range(self.n_reactors):
            self.stage_credits[i] = self.base_credits[self.current_stage[i]]
        # Dense reward: track per-reactor ethanol mass for Δ computation
        self.prev_eth_total = np.zeros(self.n_reactors)

        self.episode_mass_discharged = 0.0
        self.episode_initial_total_mass = self._compute_total_tracked_mass()

        return self.get_state_norm()

    # --- TASK / UNCERTAINTY MANAGEMENT ---

    def _apply_task(self, task):
        """Install a sampled task theta into the physics.

        Sets the kinetic/enzyme/toxin parameter vector and the feed compositions
        from the task dict, and stores the task for context/diagnostics.  Also
        caches the un-disturbed feed/param values so mid-episode disturbances can
        be applied relative to the episode's true regime.

        Args:
            task: Task dict from env_config.nominal_task() / sample_task().
        """
        self.params = np.asarray(task["params"], dtype=np.float64).copy()
        self.C_C5 = np.asarray(task["C_C5"], dtype=np.float64).copy()
        self.C_LF = np.asarray(task["C_LF"], dtype=np.float64).copy()
        self.current_task = task
        # Baseline copies (pre-disturbance) for relative disturbance application.
        self._task_params0 = self.params.copy()
        self._task_C_C5_0 = self.C_C5.copy()
        self._task_C_LF_0 = self.C_LF.copy()

    def set_task(self, task):
        """Force a specific task to be used on every subsequent reset().

        Args:
            task: Task dict, or None to clear and fall back to randomize/nominal.
        """
        self.fixed_task = task

    def seed_tasks(self, seed):
        """Reseed the task sampler for reproducible task draws.

        Args:
            seed: Integer seed for the task-sampling RNG.
        """
        self.task_rng = np.random.default_rng(seed)

    def get_task_context(self):
        """Return the current task's true latent descriptor vector (diagnostics).

        Returns:
            NumPy array of latent descriptors, or None if no task is set.  The
            agent never receives this — it must infer the regime from transitions.
        """
        return None if self.current_task is None else self.current_task.get("context")

    def schedule_disturbance(self, kind, onset_step, magnitude, ramp_steps=0):
        """Register a mid-episode disturbance for stress-testing (proposal §3.2).

        Args:
            kind: One of:
                'inlet_step'  — abrupt step change in inlet composition
                                (feeds multiplied by ``magnitude`` at onset).
                'enzyme_drop' — sudden enzyme-efficacy degradation (adsorption
                                capacities and hydrolysis eta scaled by
                                ``magnitude`` at onset).
                'mass_drift'  — gradual per-step multiplicative reactor-mass drift
                                of ``magnitude`` (e.g. -0.01) that slowly alters
                                residence-time structure; credited to the mass
                                ledger so the balance guard is not tripped.
            onset_step: Inner-step index at which the disturbance begins.
            magnitude: Disturbance size (see per-kind semantics above).
            ramp_steps: Reserved for future ramped disturbances (unused for step
                disturbances).
        """
        self._disturbances.append({
            "kind": kind, "onset": int(onset_step),
            "magnitude": float(magnitude), "ramp_steps": int(ramp_steps),
            "fired": False,
        })

    def clear_disturbances(self):
        """Remove all scheduled disturbances."""
        self._disturbances = []

    def _apply_disturbances(self):
        """Apply any due mid-episode disturbances at the current inner step."""
        if not self._disturbances:
            return
        t = self.time_step_count
        for d in self._disturbances:
            if t < d["onset"]:
                continue
            kind, mag = d["kind"], d["magnitude"]
            if kind == "mass_drift":
                # Continuous drift every step from onset onward.  Credit the
                # signed mass change to the discharge ledger so the end-of-episode
                # mass-balance guard treats it as mass genuinely entering/leaving.
                delta = self.M * mag
                self.episode_mass_discharged += float(-np.sum(delta))
                self.M = np.maximum(1.0, self.M + delta)
                continue
            # One-shot step disturbances fire exactly once.
            if d["fired"]:
                continue
            if kind == "inlet_step":
                self.C_C5 = self._task_C_C5_0 * mag
                self.C_LF = self._task_C_LF_0 * mag
            elif kind == "enzyme_drop":
                self.params = self._task_params0.copy()
                self.params[0] *= mag       # hydrolysis eta
                self.params[3:6] *= mag     # enzyme adsorption capacities
            d["fired"] = True

    def _compute_total_tracked_mass(self):
        """Compute tracked in-system mass in kilograms.

        Includes reactor masses, central storage mass, and remaining feedstocks.

        Returns:
            Total tracked mass in kg.
        """
        return float(np.sum(self.M) + self.S_mass + self.total_lf + self.total_c5)

    def get_state_norm(self):
        """Build the normalized observation tensor for the RL policy.

        Builds the normalized flat array required by the RL algorithm tensor.
        Every dimension is strictly squashed to mathematically stable bounds (usually [0,1]).

        Returns:
            NumPy array of shape [n_reactors, 28].
        """
        # 1. Normalize Chemical Concentrations
        norm_C = self.C / C_MAX_LIMITS

        # 2. Normalize Reactor Masses
        norm_M = (self.M.reshape(-1, 1) / self.Mmax.reshape(-1, 1))

        # 3. One-Hot Encode Reactor Stages (5 stages: 0–4)
        # Each reactor gets a binary indicator vector of length n_stage.
        # e.g. stage 0 → [1,0,0,0,0], stage 2 → [0,0,1,0,0]
        stage_onehot = np.eye(self.n_stage, dtype=np.float32)[
            self.current_stage]

        # 4. Normalize Stage Credits
        # Calculates the percentage of credit remaining relative to the base credit for the CURRENT stage.
        current_base_credits = self.base_credits[self.current_stage].reshape(
            -1, 1)
        norm_credit_rem = (
            self.stage_credits.reshape(-1, 1) / current_base_credits)

        # 4.5 Normalize Structural Limits (Heterogeneous capabilities exposed to the network)
        norm_Mmax = (self.Mmax.reshape(-1, 1) / 220000.0)
        norm_Fmax_LF = (self.max_flow_LF.reshape(-1, 1) / (2487/3600 * 4.0))
        norm_Fmax_C5 = (self.max_flow_C5.reshape(-1, 1) / (628/3600 * 4.0))

        # 5. Normalize Central Storage Mass
        norm_S = np.array([self.S_mass / self.S_max])

        # 6. Normalize Remaining Feedstock Stocks
        norm_LF_stock = np.array([self.total_lf / self.initial_lf])
        norm_C5_stock = np.array([self.total_c5 / self.initial_c5])

        # Form the shared global observation state
        global_state = np.concatenate(
            [norm_S, norm_LF_stock, norm_C5_stock]).astype(np.float32)
        global_state_tiled = np.tile(global_state, (self.n_reactors, 1))

        # --- Grouping Per Reactor ---
        reactor_states = np.hstack(
            [norm_C, norm_M, stage_onehot, norm_credit_rem, norm_Mmax, norm_Fmax_LF, norm_Fmax_C5]).astype(np.float32)

        # Batch-shaped state output: (n_reactors, 28)
        state = np.hstack([reactor_states, global_state_tiled])
        return state

    @staticmethod
    def get_action_mask(state_tensor):
        """Calculate stage-based structural action mask from normalized state.

        Calculates the SE-RL discrete logistic constraint mask directly from the normalized state tensor.
        A return value of 1.0 means the flow action is structurally allowed for that reactor and stage.
        A return value of 0.0 means the action is 'greyed out' and forced to a safe baseline.

        Args:
            state_tensor: Tensor with final dimension 28 (or broadcastable batch
                shape ending in 28).

        Returns:
            Torch tensor mask with final dimension 4 ordered as
            [time_multiplier, LF, C5, outflow].
        """
        import torch
        # state_tensor shape is [..., 28]
        # stage_onehot is at indices 16:21.
        stage_0 = state_tensor[..., 16]
        stage_1 = state_tensor[..., 17]
        stage_2 = state_tensor[..., 18]
        stage_3 = state_tensor[..., 19]
        stage_4 = state_tensor[..., 20]

        # Time multiplier (index 0) is allowed except in Stage 4 (Clean up) where it is fixed to 1.0
        mask_time = 1.0 - stage_4

        # LF (index 1) is allowed in Stage 0 and 1
        mask_lf = stage_0 + stage_1

        # C5 (index 2) is allowed in Stage 1 only
        mask_c5 = stage_1

        # Outflow drain (index 3) is allowed in Stage 3
        mask_out = stage_3

        mask = torch.stack([mask_time, mask_lf, mask_c5, mask_out], dim=-1)
        return mask

    def _reset_single_reactor(self, i):
        """Reset one reactor after completing cleanup stage (4).

        Reinitializes a singular reactor upon completing Clean up Stage (4).
        Prepares the vessel for the next seamless batch integration.

        Mass-balance accounting:
          The cleanup physically empties the vessel (all M[i] kg exits the system)
          and then re-charges it with 10,000 kg of fresh inoculation from outside.
          Net effect on the system mass ledger = M[i] - 10,000 (signed).

          - M[i] > 10,000 → net positive discharge (more left than re-added)
          - M[i] < 10,000 → net negative discharge (inoculation adds mass into system)

          Using a signed accumulation keeps `episode_mass_discharged` in sync with
          the actual change in tracked in-system mass so the episode-end balance
          check (violation ID 6) does not fire spuriously.

                Args:
                        i: Reactor index to reset.
        """
        # Signed net discharge: positive = mass left system, negative = mass added from outside.
        net_discharge = float(self.M[i]) - 10000.0
        self.episode_mass_discharged += net_discharge

        self.M[i] = 10000.0
        self.C[i, :] = 0.0

        init_cell_conc = (1000 * 147.0) / 10000.0
        self.C[i, self.idx['Cell']] = init_cell_conc
        self.stage_credits[i] = self.base_credits[0]
        # reset dense-reward baseline for new batch
        self.prev_eth_total[i] = 0.0

    # --- INTERNAL STATE MANAGEMENT ---

    def _get_full_internal_state(self):
        """Capture full internal state for diagnostics and trajectory resumption.

        Returns:
            Dict containing copies of dynamic environment state fields.
        """
        return {
            'C': self.C.copy(),
            'M': self.M.copy(),
            'current_stage': self.current_stage.copy(),
            'stage_credits': self.stage_credits.copy(),
            'total_lf': self.total_lf,
            'total_c5': self.total_c5,
            'S_mass': self.S_mass,
            'time_step_count': self.time_step_count,
            'prev_flow_action': self.prev_flow_action.copy(),
            'episode_mass_discharged': self.episode_mass_discharged,
            'episode_initial_total_mass': self.episode_initial_total_mass,
            'storage_overflow_lambda': self.storage_overflow_lambda,
            'storage_underflow_lambda': self.storage_underflow_lambda,
            'reactor_overflow_lambda': self.reactor_overflow_lambda.copy(),
            'reactor_underflow_lambda': self.reactor_underflow_lambda.copy(),
        }

    def _set_full_internal_state(self, state_dict):
        """Restore full internal state from a diagnostic snapshot.

        Args:
            state_dict: Dict produced by _get_full_internal_state().
        """
        self.C = state_dict['C'].copy()
        self.M = state_dict['M'].copy()
        self.current_stage = state_dict['current_stage'].copy()
        self.stage_credits = state_dict['stage_credits'].copy()
        self.total_lf = state_dict['total_lf']
        self.total_c5 = state_dict['total_c5']
        self.S_mass = state_dict['S_mass']
        self.time_step_count = state_dict['time_step_count']
        self.prev_flow_action = state_dict['prev_flow_action'].copy()
        self.episode_mass_discharged = state_dict['episode_mass_discharged']
        self.episode_initial_total_mass = state_dict['episode_initial_total_mass']
        self.storage_overflow_lambda = state_dict['storage_overflow_lambda']
        self.storage_underflow_lambda = state_dict['storage_underflow_lambda']
        self.reactor_overflow_lambda = state_dict['reactor_overflow_lambda'].copy(
        )
        self.reactor_underflow_lambda = state_dict['reactor_underflow_lambda'].copy(
        )

    def _step_internal(self, action, is_first=True):
        """Execute one inner simulation step given the agent action request.

        Executes a 1-hour interval simulation given an Agent Flow Request.
        Integrates biology, time credits, penalty scaling, and Lagrangian constraints.

        Args:
            action: Flattened action array in [-1, 1], shape [n_reactors * 4].
            is_first: True when this is the first inner step of an outer action,
                used for smoothing penalty bookkeeping.

        Returns:
            A tuple of:
                - state: Next normalized state, shape [n_reactors, 28].
                - reward_arr: Per-reactor reward array.
                - done: Episode termination flag.
                - info: Dict of decomposed reward terms and constraint flags.
        """
        # Apply any due mid-episode disturbances (feeds / kinetics / mass drift)
        # BEFORE the physics uses self.params, self.C_C5/LF and self.M.
        self._apply_disturbances()

        # Lock action space bounds
        a_clipped = np.clip(action, -1.0, 1.0)
        a = (a_clipped + 1.0) / 2.0
        a_reshaped = a.reshape(self.n_reactors, 4)

        multipliers = (a_reshaped[:, 0] * 1.5) + 0.5
        original_a_reshaped = a_reshaped.copy()

        # ==================== Phase 1: Native Stage Rules (SE-RL Failsafe DEMOUNTED) ====================
        # Enforced solely by PyTorch probability masking boundary constraints to permit true leaky-actuator simulations.

        # Master stock constraints remain local to the physical environment payload:
        if self.total_lf <= 0:
            a_reshaped[:, 1] = 0.0
        if self.total_c5 <= 0:
            a_reshaped[:, 2] = 0.0

        # Removed SE-RL explicit penalty mapping as constraints are strictly upheld by the agent mask

        total_reactor_outflow, total_harvested_eth = 0.0, 0.0
        dt_hours = self.time_step_duration / 3600.0

        # ==================== Phase 2: Time Credit Handling ====================
        # Vectorised decrement: update all reactors in one NumPy op, then
        # handle stage transitions only for the (rare) reactors that hit 0.
        self.stage_credits -= multipliers * dt_hours
        for i in np.where(self.stage_credits <= 0)[0]:
            if self.current_stage[i] == 4:
                self.current_stage[i] = 0
                self._reset_single_reactor(i)
            elif self.current_stage[i] == 3 and self.M[i] >= 100.0:
                self.stage_credits[i] = 0.0
            else:
                self.current_stage[i] += 1
                self.stage_credits[i] = self.base_credits[self.current_stage[i]]

        # ==================== Phase 3: Hardware Execution ====================
        instability_detected = False

        # Use a constant number of RK4 inner steps regardless of stage.
        current_n_steps = self.n_inner_steps
        simulation_dt = self.time_step_duration / current_n_steps

        F_ins = np.zeros(self.n_reactors)
        Cins = np.zeros((self.n_reactors, 15))

        # Vectorised outflow / harvest computation.
        f_in_lf_req = a_reshaped[:, 1] * self.max_flow_LF
        f_in_c5_req = a_reshaped[:, 2] * self.max_flow_C5
        max_possible_f = np.maximum(
            0.0, (self.M - 50.0) / self.time_step_duration)
        F_outs = np.minimum(a_reshaped[:, 3] *
                            self.max_flow_out_reac, max_possible_f)
        total_reactor_outflow = float(np.sum(F_outs))
        total_harvested_eth = float(
            np.sum(F_outs * self.C[:, self.idx['Eth']]) * self.time_step_duration)

        # Feedstock allocation is inherently sequential (shared global supply).
        for i in range(self.n_reactors):
            act_lf_mass = min(
                f_in_lf_req[i] * self.time_step_duration, self.total_lf)
            act_c5_mass = min(
                f_in_c5_req[i] * self.time_step_duration, self.total_c5)
            self.total_lf -= act_lf_mass
            self.total_c5 -= act_c5_mass
            f_lf_act = act_lf_mass / self.time_step_duration
            f_c5_act = act_c5_mass / self.time_step_duration
            F_ins[i] = f_lf_act + f_c5_act
            Cins[i] = (f_c5_act * self.C_C5 + f_lf_act * self.C_LF) / \
                F_ins[i] if F_ins[i] > 0 else np.zeros(15)

        c_new, m_new, instability_detected, instability_severity = integrate_rk4(
            self.C.astype(np.float64), self.M.astype(
                np.float64), F_ins, F_outs, Cins, self.params, simulation_dt, current_n_steps
        )
        self.C = c_new
        self.M = m_new

        # ==================== Phase 4: Tank Storage Update ====================
        prev_s_mass = self.S_mass
        self.S_mass += (total_reactor_outflow * self.time_step_duration -
                        self.S_out_rate * self.time_step_duration)
        self.S_mass = np.maximum(0.0, self.S_mass)
        # Track outlet discharge actually removed from the in-system inventory.
        self.episode_mass_discharged += max(0.0, prev_s_mass +
                                            total_reactor_outflow * self.time_step_duration - self.S_mass)

        # ==================== Phase 5: Reward Construction ====================
        reward_coef = 0.5
        prod_reward = reward_coef * (total_harvested_eth/1e7)

        # Base spike coefficient: a guaranteed, λ-independent penalty that fires
        # the moment any hard constraint is violated.  Scales with violation
        # severity so the gradient signal increases with how far past the
        # boundary the state has drifted.  Without this, a fresh λ at 0 produces
        # zero Lagrangian penalty on the first violation, leaving the PPO
        # gradient blind until λ accumulates.
        BASE_SPIKE = 0.1

        # Default state: no violation this step
        constraint_violated = False
        violation_id = 0
        violated_constraints = set()
        constraint_violated_local = np.zeros(self.n_reactors, dtype=bool)

        # Lagrangian State Update Blocks
        lagrange_penalty_local = np.zeros(self.n_reactors)
        storage_lagrange_penalty_global = 0.0

        # ── Buffer-zone soft walls (vectorised) ───────────────────────────
        # Quadratic penalty = BUFFER_COEF × depth² where depth ∈ [0, 1]
        # measures normalized penetration into the buffer zone.
        overflow_edge = 1.0 - self.OVERFLOW_BUFFER_FRAC
        m_frac_arr = self.M / self.Mmax
        overflow_depth = np.clip(
            (m_frac_arr - overflow_edge) / self.OVERFLOW_BUFFER_FRAC, 0.0, 1.0)
        lagrange_penalty_local += self.BUFFER_COEF * overflow_depth * overflow_depth
        zone_width_uf = max(
            float(self.UNDERFLOW_BUFFER_KG - self.DRY_FLOOR), 1.0)
        underflow_depth = np.clip(
            (self.UNDERFLOW_BUFFER_KG - self.M) / zone_width_uf, 0.0, 1.0)
        lagrange_penalty_local += self.BUFFER_COEF * underflow_depth * underflow_depth

        # Storage buffer zones (applied to the global penalty shared across reactors).
        s_frac = self.S_mass / self.S_max
        overflow_edge_s = 1.0 - self.STORAGE_OVERFLOW_BUFFER_FRAC
        if s_frac > overflow_edge_s:
            depth = min(1.0, (s_frac - overflow_edge_s) /
                        self.STORAGE_OVERFLOW_BUFFER_FRAC)
            storage_lagrange_penalty_global += self.BUFFER_COEF * depth * depth
        underflow_edge_s = self.STORAGE_UNDERFLOW_BUFFER_MULT * self.S_min
        if self.S_mass < underflow_edge_s:
            zone_width_s = underflow_edge_s - self.S_min
            depth = min(1.0, (underflow_edge_s - self.S_mass) /
                        max(zone_width_s, 1.0))
            storage_lagrange_penalty_global += self.BUFFER_COEF * depth * depth

        # ── Hard constraint detection + Lagrangian penalties (vectorised) ─
        r_overflow = np.maximum(0.0, self.M / self.Mmax - 1.0)
        overflow_viol_mask = r_overflow > 0
        if np.any(overflow_viol_mask):
            constraint_violated = True
            constraint_violated_local |= overflow_viol_mask
            violation_id = 1
            violated_constraints.add(1)
        lagrange_r_overflow = np.maximum(
            0.0, r_overflow - self.lagrange_target_slack)
        lagrange_penalty_local += self.reactor_overflow_lambda * lagrange_r_overflow
        lagrange_penalty_local += np.where(
            overflow_viol_mask, BASE_SPIKE * (1.0 + r_overflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_overflow_lambda = np.clip(
                np.where(lagrange_r_overflow > 0,
                         self.reactor_overflow_lambda + self.lagrange_lr * lagrange_r_overflow,
                         self.reactor_overflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)

        r_underflow = np.maximum(
            0.0, (self.DRY_FLOOR - self.M) / self.DRY_FLOOR)
        underflow_viol_mask = r_underflow > 0
        if np.any(underflow_viol_mask):
            constraint_violated = True
            constraint_violated_local |= underflow_viol_mask
            violation_id = 2
            violated_constraints.add(2)
        lagrange_r_underflow = np.maximum(
            0.0, r_underflow - self.lagrange_target_slack)
        lagrange_penalty_local += self.reactor_underflow_lambda * lagrange_r_underflow
        lagrange_penalty_local += np.where(
            underflow_viol_mask, BASE_SPIKE * (1.0 + r_underflow), 0.0)
        if self.lagrange_updates_enabled:
            self.reactor_underflow_lambda = np.clip(
                np.where(lagrange_r_underflow > 0,
                         self.reactor_underflow_lambda + self.lagrange_lr * lagrange_r_underflow,
                         self.reactor_underflow_lambda * (1.0 - self.lagrange_decay)),
                0.0, self.lagrange_max)

        # Central Hub Boundaries
        # Storage overflow: normalized by S_max → proportional fractional excess.
        storage_overflow_violation = max(
            0.0, (self.S_mass - self.S_max) / self.S_max)
        if storage_overflow_violation > 0:
            constraint_violated = True
            violation_id = 3
            violated_constraints.add(3)

        lagrange_s_overflow = max(
            0.0, storage_overflow_violation - self.lagrange_target_slack)
        storage_lagrange_penalty_global += self.storage_overflow_lambda * lagrange_s_overflow
        if storage_overflow_violation > 0:
            storage_lagrange_penalty_global += BASE_SPIKE * \
                (1.0 + storage_overflow_violation)

        if self.lagrange_updates_enabled:
            if lagrange_s_overflow > 0.0:
                new_u = self.storage_overflow_lambda + self.lagrange_lr * lagrange_s_overflow
            else:
                new_u = self.storage_overflow_lambda * (
                    1.0 - self.lagrange_decay)
            self.storage_overflow_lambda = min(
                self.lagrange_max, max(0.0, new_u))

        # Storage underflow: normalized by S_min (5,000 kg) so the gradient signal
        # spans [0, 1] instead of being suppressed to max 0.0125 by S_max (400,000).
        # At S=2500, violation = (5000-2500)/5000 = 0.5 — a strong, actionable signal.
        storage_underflow_violation = max(
            0.0, (self.S_min - self.S_mass) / self.S_min)
        if storage_underflow_violation > 0:
            constraint_violated = True
            violation_id = 4
            violated_constraints.add(4)

        lagrange_s_underflow = max(
            0.0, storage_underflow_violation - self.lagrange_target_slack)
        storage_lagrange_penalty_global += self.storage_underflow_lambda * lagrange_s_underflow
        if storage_underflow_violation > 0:
            storage_lagrange_penalty_global += BASE_SPIKE * \
                (1.0 + storage_underflow_violation)

        if self.lagrange_updates_enabled:
            if lagrange_s_underflow > 0.0:
                new_u = self.storage_underflow_lambda + self.lagrange_lr * lagrange_s_underflow
            else:
                new_u = self.storage_underflow_lambda * (
                    1.0 - self.lagrange_decay)
            self.storage_underflow_lambda = min(
                self.lagrange_max, max(0.0, new_u))

        global_constraint_penalty = storage_lagrange_penalty_global / self.n_reactors
        constraint_penalty_arr = lagrange_penalty_local + global_constraint_penalty

        # Instability penalty, scaled per reactor by instability_severity (0
        # for a reactor that stayed stable; otherwise the normalized
        # concentration excess, or 1.0 for a NaN — see integrate_rk4).
        # Previously a numerical blow-up carried NO reward penalty at all,
        # only early episode termination: no gradient about how close to
        # blowing up a trajectory got, and a perverse-incentive risk — an
        # agent already deep in penalty territory could prefer to force
        # instability and end the episode rather than keep accruing ordinary
        # constraint penalties, since termination itself was free.
        INSTABILITY_PENALTY_COEF = 5.0
        if instability_detected:
            constraint_penalty_arr = constraint_penalty_arr + \
                INSTABILITY_PENALTY_COEF * (1.0 + instability_severity)

        total_lagrange_penalty = np.sum(
            lagrange_penalty_local) + storage_lagrange_penalty_global

        # Feedstock low-supply barrier (independent of Lagrangian, pure soft penalty)
        lf_ratio = self.total_lf / self.initial_lf
        c5_ratio = self.total_c5 / self.initial_c5
        feedstock_barrier = 0.0
        _feedstock_barrier_coef = 0.005
        if 0.0 < lf_ratio < 0.02:
            feedstock_barrier += _feedstock_barrier_coef * \
                (-np.log(lf_ratio + 0.001))
        if 0.0 < c5_ratio < 0.02:
            feedstock_barrier += _feedstock_barrier_coef * \
                (-np.log(c5_ratio + 0.001))
        if feedstock_barrier > 0.0:
            constraint_penalty_arr += feedstock_barrier / self.n_reactors

        # ── Storage underflow spike + log-barrier (λ-independent) ─────────
        # Two complementary mechanisms prevent the central storage tank from
        # running dry, providing gradient signal well before the hard S_min
        # constraint fires and escalating sharply once it does:
        #
        #   Log-barrier: activates when S < 16·S_min (≈20% of S_max, ~80,000 kg).
        #     Raising the activation threshold from 2×S_min to 16×S_min gives the
        #     policy ~50 extra steps of warning before hitting the hard limit, since
        #     storage drains at 1,500 kg/hr and reactors can only replenish during
        #     stage 3.  The -log(ratio) term steepens as S→0 so the signal escalates
        #     naturally without a separate decay schedule.
        #
        #   Hard spike: fires when S < S_min.  A flat 0.15 base plus a term
        #     proportional to the fractional shortfall ensures the penalty
        #     dominates the reward landscape once the hard limit is breached,
        #     making constraint violation unambiguously costly.
        _storage_barrier_coef = 0.001
        _storage_spike_base = 0.005
        s_ratio = self.S_mass / self.S_max
        s_barrier_threshold = (16.0 * self.S_min) / \
            self.S_max  # normalized (~0.20)
        storage_underflow_extra = 0.0
        if 0.0 < s_ratio < s_barrier_threshold:
            # Log-barrier: -log(s_ratio) rises steeply as storage empties.
            # The +0.001 prevents log(0); at s_ratio ≈ s_barrier_threshold
            # the penalty is near zero so it blends smoothly with normal ops.
            storage_underflow_extra += _storage_barrier_coef * \
                (-np.log(s_ratio / s_barrier_threshold + 0.001))
        if self.S_mass < self.S_min:
            # Hard spike: flat base + proportional depth below S_min.
            shortfall = (self.S_min - self.S_mass) / self.S_min
            storage_underflow_extra += _storage_spike_base * \
                (1.0 + shortfall)
        if storage_underflow_extra > 0.0:
            constraint_penalty_arr += storage_underflow_extra / self.n_reactors

        # trickle penalty during Stage 3 (Unload)
        eff_penalty_arr = np.where(self.current_stage == 3, 0.0005, 0.0)

        if is_first:
            current_flow_action = a_clipped.reshape(self.n_reactors, 4)[:, 1:4]
            prev_flow = self.prev_flow_action.reshape(self.n_reactors, 3)
            # localise smoothing
            smoothing_penalty_arr = self.smoothing_coef * \
                np.mean(np.square(current_flow_action - prev_flow), axis=1)
            self.prev_flow_action = current_flow_action.flatten().copy()
        else:
            smoothing_penalty_arr = np.zeros(self.n_reactors)

        # ==================== Raw Material Consumption Penalty ====================
        # Penalises wasteful feedstock use: deducted each step proportional to the
        # fraction of the total initial supply consumed this tick.  This discourages
        # the agent from flooding reactors with feed early and depleting the global
        # LF/C5 pool before downstream reactors reach their feed stages.
        #
        # The penalty is symmetric for LF and C5 so both resources are conserved
        # equally, and it is normalised per-reactor so the signal magnitude is
        # independent of n_reactors.
        lf_consumed_this_step = max(
            0.0, lf_ratio * self.initial_lf - self.total_lf)
        c5_consumed_this_step = max(
            0.0, c5_ratio * self.initial_c5 - self.total_c5)
        _raw_mat_penalty_coef = 10
        raw_mat_penalty_global = _raw_mat_penalty_coef * (
            lf_consumed_this_step / max(self.initial_lf, 1.0) +
            c5_consumed_this_step / max(self.initial_c5, 1.0)
        )
        raw_mat_penalty_arr = np.full(
            self.n_reactors, raw_mat_penalty_global / self.n_reactors)

        # Equal share of global product reward plus shaping signals (no dense ethanol bonus)
        reward_arr = (prod_reward / self.n_reactors) - (smoothing_penalty_arr + eff_penalty_arr +
                                                        constraint_penalty_arr + raw_mat_penalty_arr)

        self.time_step_count += 1

        # Episode completion logic
        done = (self.time_step_count >= self.max_steps or instability_detected)

        # Numerical instability in the ODE integrator terminates the episode (violation 5: instability)
        if instability_detected:
            violation_id = 5  # Instability
            violated_constraints.add(5)

        hard_violation = instability_detected or constraint_violated

        info = {
            "prod": prod_reward,
            "constraint": np.sum(constraint_penalty_arr),
            "smoothing": np.sum(smoothing_penalty_arr),
            "eff": np.sum(eff_penalty_arr),
            "eth_harvested": total_harvested_eth,
            "raw_mat_penalty": float(np.sum(raw_mat_penalty_arr)),
            "lf_remaining": self.total_lf,
            "c5_remaining": self.total_c5,
            "storage_overflow_violation": float(storage_overflow_violation),
            "storage_underflow_violation": float(storage_underflow_violation),
            "instability_severity": float(np.max(instability_severity)),
            "total_lagrange_penalty": float(total_lagrange_penalty),
            "hard_violation": hard_violation,
            "constraint_violated": constraint_violated,
            "violation_id": violation_id,
            "violated_constraints": list(violated_constraints),
            "actual_a_reshaped": a_reshaped.copy(),
        }
        return self.get_state_norm(), reward_arr, done, info

    def step(self, action):
        """Execute one outer environment step with action repeat.

        Outer step function that repeats the agent's action for
        control_action_interval inner steps.

        Args:
            action: Flattened action array in [-1, 1], shape [n_reactors * 4].

        Returns:
            A tuple of:
                - state: Next normalized state, shape [n_reactors, 28].
                - total_reward: Per-reactor accumulated reward over repeats.
                - done: Episode termination flag.
                - info_out: Aggregated diagnostics for the outer step.
        """
        total_prod = 0.0
        total_constraint = 0.0
        total_smoothing = 0.0
        total_eff = 0.0
        total_reward = np.zeros(self.n_reactors)

        total_harvested_eth_accum = 0.0

        max_storage_overflow = 0.0
        max_storage_underflow = 0.0

        lagrange_penalty_accum = 0.0
        raw_mat_penalty_accum = 0.0

        hard_violation_accum = False
        constraint_violated_accum = False
        violation_id_accum = 0
        violation_count_accum = 0
        violation_ids_accum = set()

        actual_a_reshaped_first = None
        lf_remaining_last = 0.0
        c5_remaining_last = 0.0

        # Track actual ticks executed (may be < interval if done fires early)
        steps_completed = 0
        for idx in range(self.control_action_interval):
            state, reward, done, info = self._step_internal(
                action, is_first=(idx == 0))
            steps_completed += 1

            total_prod += info.get("prod", 0.0)
            total_constraint += info.get("constraint", 0.0)
            total_smoothing += info.get("smoothing", 0.0)
            total_eff += info.get("eff", 0.0)
            total_reward += reward

            total_harvested_eth_accum += info.get("eth_harvested", 0.0)

            # Track worst-case violation magnitude seen across any sub-step
            max_storage_overflow = max(max_storage_overflow, info.get(
                "storage_overflow_violation", 0.0))
            max_storage_underflow = max(max_storage_underflow, info.get(
                "storage_underflow_violation", 0.0))

            # Accumulate penalties — will be normalised below
            lagrange_penalty_accum += info.get("total_lagrange_penalty", 0.0)
            raw_mat_penalty_accum += info.get("raw_mat_penalty", 0.0)

            hard_violation_accum = hard_violation_accum or info.get(
                "hard_violation", False)
            constraint_violated_accum = constraint_violated_accum or info.get(
                "constraint_violated", False)
            if info.get("violation_id", 0) > 0:
                violation_id_accum = max(
                    violation_id_accum, info["violation_id"])
            if "violated_constraints" in info:
                violation_ids_accum.update(info["violated_constraints"])
            if info.get("hard_violation", False) or info.get("constraint_violated", False):
                violation_count_accum += 1

            lf_remaining_last = info.get("lf_remaining", 0.0)
            c5_remaining_last = info.get("c5_remaining", 0.0)

            if idx == 0:
                actual_a_reshaped_first = info.get("actual_a_reshaped")

            if done:
                break

        norm = max(steps_completed, 1)
        info_out = {
            "prod": total_prod,
            "constraint": total_constraint,
            "smoothing": total_smoothing,
            "eff": total_eff,
            "eth_harvested": total_harvested_eth_accum,
            "raw_mat_penalty": raw_mat_penalty_accum,
            "lf_remaining": lf_remaining_last,
            "c5_remaining": c5_remaining_last,
            "storage_overflow_violation": float(max_storage_overflow),
            "storage_underflow_violation": float(max_storage_underflow),
            "hard_violation": hard_violation_accum,
            "constraint_violated": constraint_violated_accum,
            "violation_id": violation_id_accum,
            "violation_ids": list(violation_ids_accum),
            "violation_count": violation_count_accum,
            "actual_a_reshaped": actual_a_reshaped_first,
            # Divide by steps_completed so per-decision magnitude stays consistent
            # with the original 1-hour control regime regardless of interval length.
            "storage_lagrange_penalty": float(lagrange_penalty_accum / norm),
        }

        # End-of-episode mass balance check to catch long-horizon numerical drift.
        if done:
            current_total_mass = self._compute_total_tracked_mass()
            expected_total_mass = self.episode_initial_total_mass - self.episode_mass_discharged
            mass_balance_error = current_total_mass - expected_total_mass
            mass_balance_tol = max(
                self.mass_balance_abs_tol,
                self.mass_balance_rel_tol *
                max(self.episode_initial_total_mass, 1.0)
            )
            mass_balance_ok = abs(mass_balance_error) <= mass_balance_tol

            info_out["mass_balance_error"] = float(mass_balance_error)
            info_out["mass_balance_tol"] = float(mass_balance_tol)
            info_out["mass_balance_ok"] = bool(mass_balance_ok)

            if not mass_balance_ok:
                info_out["hard_violation"] = True
                info_out["constraint_violated"] = True
                info_out["violation_id"] = max(
                    int(info_out["violation_id"]), 6)
                violation_ids = set(info_out.get("violation_ids", []))
                violation_ids.add(6)
                info_out["violation_ids"] = list(violation_ids)
                info_out["violation_count"] = int(
                    info_out.get("violation_count", 0)) + 1

                # Severity-scaled penalty, added to this final step's reward.
                # Previously this branch only set diagnostic flags — zero
                # reward consequence — so the agent had no training signal at
                # all about mass-conservation drift, however large. Severity
                # is measured in multiples of the allowed tolerance (1.0 =
                # right at the boundary), so the penalty scales with how far
                # past it the drift actually was, consistent with every other
                # violation type in this environment.
                MASS_BALANCE_PENALTY_COEF = 1.0
                mb_severity = abs(mass_balance_error) / max(mass_balance_tol, 1e-6)
                mass_balance_penalty = MASS_BALANCE_PENALTY_COEF * mb_severity
                total_reward = total_reward - mass_balance_penalty / self.n_reactors
                info_out["mass_balance_penalty"] = float(mass_balance_penalty)

        return state, total_reward, done, info_out
