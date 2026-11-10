"""Plant-model mismatch: kinetics with rate terms the modelled plant lacks.

The plant the agents train on runs ``metarl.env.kinetics`` (Fermentation-PPO's,
verbatim). A mismatched plant runs the same rate law with a constant
multiplier ``theta`` on individual rate terms, so the true process has
structure no training plant had: the faults and the task prior only move the
three lumped modifiers in ``params`` (hydrolysis eta, glucose and xylose
uptake), and the APN's KineticNet was trained on that same rate law.

``calculate_rates_mismatch`` is ``kinetics.calculate_rates_numba`` with
``theta`` inserted and nothing else changed; at ``theta = 1`` it reproduces
it (``tests/smoke_test.py``). The integrator is ``kinetics.integrate_rk4``
over these rates. Re-copy both when ``kinetics.py`` is re-copied.

Terms (``MISMATCH_TERMS``), each a multiplier on:

    r1     cellulose -> cellobiose hydrolysis
    r2     cellulose -> glucose hydrolysis
    r3     cellobiose -> glucose hydrolysis
    r4     xylan -> xylose hydrolysis (and r5 = 0.5 r4, its acetyl release)
    qG     glucose uptake (ethanol, CO2 and growth on glucose)
    qX     xylose uptake (ethanol, CO2 and growth on xylose)
    qF     furfural detoxification
    qHMF   HMF detoxification
    qACT   acetate consumption
    death  the yeast's death constants on both sugars
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
from numba import njit

MISMATCH_TERMS = ("r1", "r2", "r3", "r4", "qG", "qX", "qF", "qHMF", "qACT",
                  "death")
MISMATCH_INDEX = {k: i for i, k in enumerate(MISMATCH_TERMS)}
N_MISMATCH_TERMS = len(MISMATCH_TERMS)


def mismatch_matrix(spec, n_reactors):
    """The ``theta`` matrix for a mismatch spec, or None when it is nominal.

    Args:
        spec: None, or dict term -> multiplier (a scalar for every reactor,
            or one value per reactor).
        n_reactors: Number of reactors.

    Returns:
        Float64 array [n_reactors, N_MISMATCH_TERMS], or None when every
        multiplier is 1 (the plant then runs ``kinetics`` itself, so the
        nominal plant stays bit-identical to Fermentation-PPO's).

    Raises:
        ValueError: On an unknown term or a multiplier that is not positive
            and finite.
    """
    if not spec:
        return None
    theta = np.ones((n_reactors, N_MISMATCH_TERMS))
    for term, value in spec.items():
        if term not in MISMATCH_INDEX:
            raise ValueError(f"unknown mismatch term {term!r}; expected one "
                             f"of {MISMATCH_TERMS}")
        v = np.broadcast_to(np.asarray(value, dtype=np.float64), (n_reactors,))
        if not np.all(np.isfinite(v) & (v > 0.0)):
            raise ValueError(f"mismatch {term}={value!r} must be positive")
        theta[:, MISMATCH_INDEX[term]] = v
    return None if np.all(theta == 1.0) else theta


@njit(fastmath=True, cache=True)
def calculate_rates_mismatch(C, params, theta):
    """``kinetics.calculate_rates_numba`` with a multiplier per rate term.

    Args:
        C: Concentration matrix [n_reactors, 15].
        params: Kinetic modifiers [n_reactors, 3] (eta, ph_g, ph_x).
        theta: Term multipliers [n_reactors, N_MISMATCH_TERMS], in
            MISMATCH_TERMS order.

    Returns:
        Rate matrix [n_reactors, 15].
    """
    i_CS, i_XS, i_LS, i_C, i_G, i_X, i_F, i_E, i_AC, i_Cell, i_Eth, i_CO2, i_ACT, i_HMF, i_Base = range(
        15)
    eps = 1e-8

    n_reactors = C.shape[0]
    R = np.zeros((n_reactors, 15))

    for i in range(n_reactors):
        _ETA_param = params[i, 0]
        _PH_G_param = params[i, 1]
        _PH_X_param = params[i, 2]

        C_E_total = C[i, i_E]
        C_solid = C[i, i_CS] + C[i, i_XS] + C[i, i_LS] + eps
        Ce = np.array([0.5 * C_E_total, 0.3 * C_E_total, 0.2 * C_E_total])
        max_ads = np.array([0.015, 0.01, 0.01])
        k_ads = np.array([0.84, 0.1, 0.1])

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

        r1 = theta[i, 0] * (0.00034 * _ETA_param * CebC1 * C[i, i_CS]) / (1 + (C[i, i_C]/0.0014) + (
            C[i, i_X]/0.1007) + (C[i, i_G]/0.073) + (C[i, i_F]/10.0) + (C[i, i_Eth]/0.15) + eps)
        r2 = theta[i, 1] * (0.0023 * _ETA_param * (CebC1 + CebC2) * C[i, i_CS]) / (
            1 + (C[i, i_C]/132.0) + (C[i, i_X]/0.029) + (C[i, i_G]/0.34) + (C[i, i_F]/10.0) + eps)
        r3 = theta[i, 2] * (0.07 * _ETA_param * Cef[1] * C[i, i_C]) / (24.3 * (1 +
                                                                              (C[i, i_X]/201.0) + (C[i, i_G]/3.9) + (C[i, i_F]/10.0)) + C[i, i_C] + eps)
        r4 = theta[i, 3] * (0.0087 * _ETA_param * CebX3 * C[i, i_XS]) / (1 + (C[i, i_C]/24.3) +
                                                                        (C[i, i_X]/201.0) + (C[i, i_G]/2.34) + (C[i, i_F]/10.0) + eps)
        r5 = 0.5 * r4

        monod_G = C[i, i_G] / (1.342 + C[i, i_G] + (C[i, i_G]**2)/4890.0 + eps)
        eth_inh_G = max(0.0, 1 - (C[i, i_Eth]/103.0)**1.42)
        tox_inh_G = (0.75/(0.75+C[i, i_F])) * \
            (2.74/(2.74+C[i, i_ACT])) * (2.0/(2.0+C[i, i_HMF]))
        qG = theta[i, 4] * (1/0.47) * (0.000318 * _PH_G_param) * \
            C[i, i_Cell] * monod_G * eth_inh_G * tox_inh_G

        monod_X = C[i, i_X] / (3.4 + C[i, i_X] + (C[i, i_X]**2)/81.3 + eps)
        eth_inh_X = max(0.0, 1 - (C[i, i_Eth]/100.2)**0.608)
        tox_inh_X = (0.35/(0.35+C[i, i_F])) * \
            (0.2/(0.2+C[i, i_ACT])) * (10.0/(10.0+C[i, i_HMF]))
        qX = theta[i, 5] * (1/0.4) * (0.00083444 * _PH_X_param) * \
            C[i, i_Cell] * monod_X * eth_inh_X * tox_inh_X

        qF = theta[i, 6] * 4.6706E-5 * C[i, i_Cell] * (C[i, i_F] / (0.05 + C[i, i_F] + eps))
        qHMF = theta[i, 7] * (8.7576E-5 * C[i, i_Cell] * (C[i, i_HMF] /
                              (C[i, i_HMF] + 0.5 + eps))) * (0.25 / (0.25 + C[i, i_F] + eps))
        qACT = theta[i, 8] * 1.2292E-5 * C[i, i_Cell] * \
            (C[i, i_ACT] / (C[i, i_ACT] + 2.5 + eps))

        R[i, i_CS], R[i, i_XS], R[i, i_LS] = -r1-r2, -r4-r5, 0.0
        R[i, i_C], R[i, i_G], R[i, i_X] = r1-r3, r2+r3-qG, r4-qX
        R[i, i_F], R[i, i_Eth], R[i, i_HMF] = -qF, qG*0.47 + qX*0.4, -qHMF
        R[i, i_ACT], R[i, i_CO2], R[i, i_AC] = qHMF*0.23392 - \
            qACT, qG*0.47 + qX*0.4 + qACT*0.1, r5

        kd = theta[i, 9]
        sugar_sum = C[i, i_G] + C[i, i_X] + eps
        R[i, i_Cell] = ((C[i, i_G]/sugar_sum)*(qG - kd*2.6944E-5*C[i, i_Cell]) *
                        0.115 + (C[i, i_X]/sugar_sum)*(qX - kd*1.8611E-5*C[i, i_Cell])*0.162)
    return R


@njit(fastmath=True, cache=True)
def _ode_system_mismatch(C, M, F_in, F_out, Cin, params, theta):
    """``kinetics.ode_system`` over the mismatched rates."""
    n_reactors = C.shape[0]
    C_safe = np.empty_like(C)
    M_safe = np.empty_like(M)
    for i in range(n_reactors):
        M_safe[i] = M[i] if M[i] > 1.0 else 1.0
        for j in range(15):
            C_safe[i, j] = C[i, j] if C[i, j] > 1e-8 else 1e-8

    R = calculate_rates_mismatch(C_safe, params, theta)
    dCdt = np.zeros_like(C_safe)
    dMdt = F_in - F_out
    for i in range(n_reactors):
        for j in range(15):
            dCdt[i, j] = (F_in[i] * (Cin[i, j] - C_safe[i, j])) / \
                (M_safe[i] + 1e-8) + R[i, j]
    return dCdt, dMdt


@njit(fastmath=True, cache=True)
def integrate_rk4_mismatch(c_init, m_init, F_in, F_out, Cin, params, theta,
                           dt, n_steps):
    """``kinetics.integrate_rk4`` over the mismatched rates.

    Returns:
        Tuple (concentrations, masses, instability_detected).
    """
    c = c_init.copy()
    m = m_init.copy()
    n_reactors = c.shape[0]

    for _ in range(n_steps):
        dc1, dm1 = _ode_system_mismatch(c, m, F_in, F_out, Cin, params, theta)
        dc2, dm2 = _ode_system_mismatch(c + 0.5 * dt * dc1, m + 0.5 * dt * dm1,
                                        F_in, F_out, Cin, params, theta)
        dc3, dm3 = _ode_system_mismatch(c + 0.5 * dt * dc2, m + 0.5 * dt * dm2,
                                        F_in, F_out, Cin, params, theta)
        dc4, dm4 = _ode_system_mismatch(c + dt * dc3, m + dt * dm3,
                                        F_in, F_out, Cin, params, theta)

        c += (dt / 6.0) * (dc1 + 2.0*dc2 + 2.0*dc3 + dc4)
        m += (dt / 6.0) * (dm1 + 2.0*dm2 + 2.0*dm3 + dm4)

        for i in range(n_reactors):
            for j in range(15):
                if c[i, j] < 1e-8:
                    c[i, j] = 1e-8
            if m[i] < 1.0:
                m[i] = 1.0

        if np.any(np.isnan(c)) or np.max(c) > 3000.0:
            return c, m, True

    return c, m, False
