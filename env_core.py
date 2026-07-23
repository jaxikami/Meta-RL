"""Core kinetic and integration kernels for fermentation simulation.

This module contains Numba-jitted numerical routines used by the environment
to compute biochemical reaction rates and advance reactor states via RK4.
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
from numba import njit

# ─── Nominal pH and Kinetic-Modifier Bump Functions ──────────────────────────
# The three kinetic modifiers (overall hydrolysis eta, glucose-uptake and
# xylose-uptake pH factors) are pre-computed Gaussian "bump" functions of the
# operating pH.  Randomising pH therefore induces a physically coherent,
# CORRELATED shift across all three rate channels simultaneously (proposal
# §2.3, group 1) rather than perturbing them independently.
_PH = 5.39

# Bump-function parameters (amplitude, mean, sigma) for each channel.
_ETA_A, _ETA_MU, _ETA_SIG = 1.0 * 0.3, 5.178044612, 1.088854751
_PHG_MU, _PHG_SIG = 5.3887586428, 0.0096983961
_PHX_MU, _PHX_SIG = 5.3752378194, 0.0093149827


def kinetics_from_pH(pH):
    """Map an operating pH to the three correlated kinetic modifiers.

    Implements the pre-computed Gaussian bump functions used throughout the
    simulator.  Because all three channels share the single pH input, a single
    pH draw yields a coherent, correlated shift in (eta, gamma_G, gamma_X) —
    exactly the coupling described in proposal §2.3, group 1.

    Args:
        pH: Scalar operating pH.

    Returns:
        A tuple (eta, ph_g_factor, ph_x_factor) of float kinetic modifiers.
    """
    eta = _ETA_A * np.exp(-((pH - _ETA_MU) ** 2) / (2 * (_ETA_SIG ** 2)))
    ph_g = np.exp(-((pH - _PHG_MU) ** 2) / (2 * (_PHG_SIG ** 2)))
    ph_x = np.exp(-((pH - _PHX_MU) ** 2) / (2 * (_PHX_SIG ** 2)))
    return float(eta), float(ph_g), float(ph_x)


# Nominal kinetic modifiers evaluated at the nominal operating pH.
_ETA, _PH_G_FACTOR, _PH_X_FACTOR = kinetics_from_pH(_PH)

# ─── Nominal enzyme-adsorption parameters (proposal §2.3, group 3) ────────────
# Langmuir adsorption capacities (E_max,j) and affinities (K_j) for the three
# enzyme components: cellulase-on-cellulose, cellulase-on-lignin (nonproductive),
# xylanase-on-xylan.  Exposed so the task sampler can randomise lot-to-lot
# enzyme efficacy.
NOMINAL_MAX_ADS = np.array([0.015, 0.01, 0.01], dtype=np.float64)
NOMINAL_K_ADS = np.array([0.84, 0.1, 0.1], dtype=np.float64)

# ─── Nominal toxin half-saturation constants (proposal §2.3, group 2) ─────────
# Glucose- and xylose-channel inhibition constants for furfural (F), acetate
# (ACT) and HMF.  The task sampler multiplies these by per-toxin tolerance
# factors; the SAME factor scales the (biologically linked) G and X channels so
# strain tolerance is not decoupled across channels.
NOMINAL_TOX_G = np.array([0.75, 2.74, 2.0], dtype=np.float64)   # [F, ACT, HMF]
NOMINAL_TOX_X = np.array([0.35, 0.2, 10.0], dtype=np.float64)   # [F, ACT, HMF]

# Length of the kinetic parameter vector consumed by calculate_rates_numba.
#   [0]   eta                (overall hydrolysis modifier)
#   [1]   ph_g_factor        (glucose-uptake pH modifier)
#   [2]   ph_x_factor        (xylose-uptake pH modifier)
#   [3:6] max_ads[0:3]       (enzyme adsorption capacities E_max,j)
#   [6:9] k_ads[0:3]         (enzyme adsorption affinities K_j)
#   [9]   tox_F              (furfural tolerance multiplier)
#   [10]  tox_ACT            (acetate tolerance multiplier)
#   [11]  tox_HMF            (HMF tolerance multiplier)
PARAM_DIM = 12


@njit(fastmath=True, cache=True)
def calculate_rates_numba(C, params):
    """Compute biochemical reaction rates for each reactor.

    Args:
        C: Concentration matrix of shape [n_reactors, 15].
        params: Kinetic parameter vector of length ``PARAM_DIM`` (12).  Layout:
            [0]   eta                overall hydrolysis modifier
            [1]   ph_g_factor        glucose-uptake pH modifier
            [2]   ph_x_factor        xylose-uptake pH modifier
            [3:6] max_ads[0:3]       enzyme adsorption capacities (E_max,j)
            [6:9] k_ads[0:3]         enzyme adsorption affinities (K_j)
            [9]   tox_F              furfural tolerance multiplier
            [10]  tox_ACT            acetate tolerance multiplier
            [11]  tox_HMF            HMF tolerance multiplier

    Returns:
        Rate matrix of shape [n_reactors, 15] for tracked species.
    """
    i_CS, i_XS, i_LS, i_C, i_G, i_X, i_F, i_E, i_AC, i_Cell, i_Eth, i_CO2, i_ACT, i_HMF, i_Base = range(
        15)
    eps = 1e-8

    n_reactors = C.shape[0]
    R = np.zeros((n_reactors, 15))

    _ETA_param = params[0]
    _PH_G_param = params[1]
    _PH_X_param = params[2]

    # Enzyme adsorption capacities/affinities (randomisable, group 3).
    max_ads = np.array([params[3], params[4], params[5]])
    k_ads = np.array([params[6], params[7], params[8]])

    # Toxin tolerance multipliers (randomisable, group 2).  A factor > 1 widens
    # the half-saturation constant → weaker inhibition → more tolerant strain.
    tox_F = params[9]
    tox_ACT = params[10]
    tox_HMF = params[11]

    for i in range(n_reactors):
        C_E_total = C[i, i_E]
        C_solid = C[i, i_CS] + C[i, i_XS] + C[i, i_LS] + eps
        Ce = np.array([0.5 * C_E_total, 0.3 * C_E_total, 0.2 * C_E_total])

        Cef = np.zeros(3)
        Ceb = np.zeros(3)
        for j in range(3):
            a = k_ads[j]
            b = 1 + k_ads[j] * C_solid * max_ads[j] - k_ads[j] * Ce[j]
            c = -Ce[j]
            Cef[j] = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a + eps)
            Ceb[j] = Ce[j] - Cef[j]

        CebC1 = Ceb[0] * (C[i, i_CS] / C_solid)
        CebC2 = Ceb[1] * (C[i, i_CS] / C_solid)
        CebX3 = Ceb[2] * (C[i, i_XS] / C_solid)

        # Reaction progression rates
        r1 = (0.00034 * _ETA_param * CebC1 * C[i, i_CS]) / (1 + (C[i, i_C]/0.0014) + (
            C[i, i_X]/0.1007) + (C[i, i_G]/0.073) + (C[i, i_F]/10.0) + (C[i, i_Eth]/0.15) + eps)
        r2 = (0.0023 * _ETA_param * (CebC1 + CebC2) * C[i, i_CS]) / (
            1 + (C[i, i_C]/132.0) + (C[i, i_X]/0.029) + (C[i, i_G]/0.34) + (C[i, i_F]/10.0) + eps)
        r3 = (0.07 * _ETA_param * Cef[1] * C[i, i_C]) / (24.3 * (1 +
                                                                 (C[i, i_X]/201.0) + (C[i, i_G]/3.9) + (C[i, i_F]/10.0)) + C[i, i_C] + eps)
        r4 = (0.0087 * _ETA_param * CebX3 * C[i, i_XS]) / (1 + (C[i, i_C]/24.3) +
                                                           (C[i, i_X]/201.0) + (C[i, i_G]/2.34) + (C[i, i_F]/10.0) + eps)
        r5 = 0.5 * r4

        # Monod continuous biological limits
        monod_G = C[i, i_G] / (1.342 + C[i, i_G] + (C[i, i_G]**2)/4890.0 + eps)
        eth_inh_G = max(0.0, 1 - (C[i, i_Eth]/103.0)**1.42)
        # Toxin inhibition: multiply each half-saturation constant by its
        # tolerance factor (same factor for the linked G and X channels).
        kF_G, kACT_G, kHMF_G = 0.75 * tox_F, 2.74 * tox_ACT, 2.0 * tox_HMF
        tox_inh_G = (kF_G/(kF_G+C[i, i_F])) * \
            (kACT_G/(kACT_G+C[i, i_ACT])) * (kHMF_G/(kHMF_G+C[i, i_HMF]))
        qG = (1/0.47) * (0.000318 * _PH_G_param) * \
            C[i, i_Cell] * monod_G * eth_inh_G * tox_inh_G

        monod_X = C[i, i_X] / (3.4 + C[i, i_X] + (C[i, i_X]**2)/81.3 + eps)
        eth_inh_X = max(0.0, 1 - (C[i, i_Eth]/100.2)**0.608)
        kF_X, kACT_X, kHMF_X = 0.35 * tox_F, 0.2 * tox_ACT, 10.0 * tox_HMF
        tox_inh_X = (kF_X/(kF_X+C[i, i_F])) * \
            (kACT_X/(kACT_X+C[i, i_ACT])) * (kHMF_X/(kHMF_X+C[i, i_HMF]))
        qX = (1/0.4) * (0.00083444 * _PH_X_param) * \
            C[i, i_Cell] * monod_X * eth_inh_X * tox_inh_X

        # Toxin breakdown rates
        qF = 4.6706E-5 * C[i, i_Cell] * (C[i, i_F] / (0.05 + C[i, i_F] + eps))
        qHMF = (8.7576E-5 * C[i, i_Cell] * (C[i, i_HMF] /
                (C[i, i_HMF] + 0.5 + eps))) * (0.25 / (0.25 + C[i, i_F] + eps))
        qACT = 1.2292E-5 * C[i, i_Cell] * \
            (C[i, i_ACT] / (C[i, i_ACT] + 2.5 + eps))

        R[i, i_CS], R[i, i_XS], R[i, i_LS] = -r1-r2, -r4-r5, 0.0
        R[i, i_C], R[i, i_G], R[i, i_X] = r1-r3, r2+r3-qG, r4-qX
        R[i, i_F], R[i, i_Eth], R[i, i_HMF] = -qF, qG*0.47 + qX*0.4, -qHMF
        R[i, i_ACT], R[i, i_CO2], R[i, i_AC] = qHMF*0.23392 - \
            qACT, qG*0.47 + qX*0.4 + qACT*0.1, r5

        # Yeast cell reproduction/death
        sugar_sum = C[i, i_G] + C[i, i_X] + eps
        R[i, i_Cell] = ((C[i, i_G]/sugar_sum)*(qG - 2.6944E-5*C[i, i_Cell]) *
                        0.115 + (C[i, i_X]/sugar_sum)*(qX - 1.8611E-5*C[i, i_Cell])*0.162)
    return R


@njit(fastmath=True, cache=True)
def ode_system(C, M, F_in, F_out, Cin, params):
    """Evaluate reactor ODE derivatives for concentrations and mass.

    Applies lower bounds to concentrations and masses in intermediate RK4
    stages for numerical stability.

    Args:
        C: Concentration matrix [n_reactors, 15].
        M: Reactor mass vector [n_reactors].
        F_in: Inflow rates [n_reactors].
        F_out: Outflow rates [n_reactors].
        Cin: Inlet concentrations [n_reactors, 15].
        params: Kinetic modifier array [eta, ph_g_factor, ph_x_factor].

    Returns:
        A tuple of:
            - dCdt: Concentration derivatives [n_reactors, 15].
            - dMdt: Mass derivatives [n_reactors].
    """
    # Ensure physical bounds for intermediate RK4 step stability to prevent ZeroDivisionError
    n_reactors = C.shape[0]
    C_safe = np.empty_like(C)
    M_safe = np.empty_like(M)
    for i in range(n_reactors):
        M_safe[i] = M[i] if M[i] > 1.0 else 1.0
        for j in range(15):
            C_safe[i, j] = C[i, j] if C[i, j] > 1e-8 else 1e-8

    R = calculate_rates_numba(C_safe, params)
    dCdt = np.zeros_like(C_safe)
    dMdt = F_in - F_out
    for i in range(n_reactors):
        for j in range(15):
            dCdt[i, j] = (F_in[i] * (Cin[i, j] - C_safe[i, j])) / \
                (M_safe[i] + 1e-8) + R[i, j]
    return dCdt, dMdt


@njit(fastmath=True, cache=True)
def integrate_rk4(c_init, m_init, F_in, F_out, Cin, params, dt, n_steps):
    """Integrate coupled reactor dynamics with fixed-step RK4.

    Args:
        c_init: Initial concentrations [n_reactors, 15].
        m_init: Initial reactor masses [n_reactors].
        F_in: Inflow rates [n_reactors].
        F_out: Outflow rates [n_reactors].
        Cin: Inlet concentrations [n_reactors, 15].
        params: Kinetic modifier array [eta, ph_g_factor, ph_x_factor].
        dt: RK4 integration step size.
        n_steps: Number of RK4 sub-steps to execute.

    Returns:
        A tuple of:
            - c: Final concentrations [n_reactors, 15].
            - m: Final masses [n_reactors].
            - instability_detected: True if NaN or concentration overflow
              instability is detected.
            - severity: Per-reactor instability severity, shape [n_reactors].
              0 for a reactor that stayed stable. For a concentration
              blow-up, the normalized excess over the 3000 threshold
              ((max_c - 3000) / 3000); for a NaN, a fixed severity of 1.0
              (a NaN has no natural magnitude to scale by). Lets the caller
              penalize instability in proportion to how far past the
              numerical-stability boundary the integrator drifted, rather
              than only a flat pass/fail signal.
    """
    c = c_init.copy()
    m = m_init.copy()
    n_reactors = c.shape[0]
    severity = np.zeros(n_reactors)

    for _ in range(n_steps):
        dc1, dm1 = ode_system(c, m, F_in, F_out, Cin, params)
        dc2, dm2 = ode_system(c + 0.5 * dt * dc1, m +
                              0.5 * dt * dm1, F_in, F_out, Cin, params)
        dc3, dm3 = ode_system(c + 0.5 * dt * dc2, m +
                              0.5 * dt * dm2, F_in, F_out, Cin, params)
        dc4, dm4 = ode_system(c + dt * dc3, m + dt * dm3,
                              F_in, F_out, Cin, params)

        c += (dt / 6.0) * (dc1 + 2.0*dc2 + 2.0*dc3 + dc4)
        m += (dt / 6.0) * (dm1 + 2.0*dm2 + 2.0*dm3 + dm4)

        for i in range(n_reactors):
            for j in range(15):
                if c[i, j] < 1e-8:
                    c[i, j] = 1e-8
            if m[i] < 1.0:
                m[i] = 1.0

        unstable = False
        for i in range(n_reactors):
            row_has_nan = False
            row_max = 0.0
            for j in range(15):
                v = c[i, j]
                if np.isnan(v):
                    row_has_nan = True
                elif v > row_max:
                    row_max = v
            if row_has_nan:
                severity[i] = 1.0
                unstable = True
            elif row_max > 3000.0:
                s = (row_max - 3000.0) / 3000.0
                if s > severity[i]:
                    severity[i] = s
                unstable = True
        if unstable:
            return c, m, True, severity

    return c, m, False, severity
