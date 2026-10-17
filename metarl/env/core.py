"""Numba-jitted fermentation kinetics and RK4 integration.

The rate kernel is the Fermentation-PPO project's, with its hardcoded kinetic
constants lifted into a parameter vector (``PARAM_DIM``), exactly as in the
lean single-reactor study (``MetaRL_Lean/env_core.py``). The kernels keep a
leading reactor axis, and the plant passes all three reactors at once.

``integrate_rk4`` is the allocation-free integrator the environment uses. It
takes one parameter row PER REACTOR (``params`` of shape [n_reactors,
PARAM_DIM]), so a biochemical fault can act on one vessel and leave the others
alone. ``calculate_rates_numba`` / ``ode_system`` / ``integrate_rk4_reference``
are the original array-based path with one shared parameter vector, kept as the
reference the fast path is checked against (``tests/smoke_test.py``, with
identical rows).
"""

# ─── Third-Party ──────────────────────────────────────────────────────────────
import numpy as np
from numba import njit

# ─── Nominal pH and Kinetic-Modifier Bump Functions ──────────────────────────
_PH = 5.39

# Gaussian bump parameters (amplitude, mean, sigma) for each pH modifier.
_ETA_A, _ETA_MU, _ETA_SIG = 1.0 * 0.3, 5.178044612, 1.088854751
_PHG_MU, _PHG_SIG = 5.3887586428, 0.0096983961
_PHX_MU, _PHX_SIG = 5.3752378194, 0.0093149827


def kinetics_from_pH(pH):
    """Map an operating pH to the three kinetic modifiers.

    All three are Gaussian bumps of the same pH, so a pH change shifts
    (eta, gamma_G, gamma_X) together rather than independently.

    Args:
        pH: Scalar operating pH.

    Returns:
        A tuple (eta, ph_g_factor, ph_x_factor) of float kinetic modifiers.
    """
    eta = _ETA_A * np.exp(-((pH - _ETA_MU) ** 2) / (2 * (_ETA_SIG ** 2)))
    ph_g = np.exp(-((pH - _PHG_MU) ** 2) / (2 * (_PHG_SIG ** 2)))
    ph_x = np.exp(-((pH - _PHX_MU) ** 2) / (2 * (_PHX_SIG ** 2)))
    return float(eta), float(ph_g), float(ph_x)


# ─── Nominal enzyme-adsorption parameters ─────────────────────────────────────
# Langmuir capacities (E_max,j) and affinities (K_j) for cellulase-on-cellulose,
# cellulase-on-lignin (nonproductive) and xylanase-on-xylan.
NOMINAL_MAX_ADS = np.array([0.015, 0.01, 0.01], dtype=np.float64)
NOMINAL_K_ADS = np.array([0.84, 0.1, 0.1], dtype=np.float64)

# Length of the kinetic parameter vector; layout in calculate_rates_numba.
PARAM_DIM = 12


@njit(fastmath=True, cache=True)
def calculate_rates_numba(C, params):
    """Compute biochemical reaction rates for each reactor (reference path).

    Args:
        C: Concentration matrix of shape [n_reactors, 15].
        params: Kinetic parameter vector of length ``PARAM_DIM`` (12), shared
            by every reactor. Layout:
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

    max_ads = np.array([params[3], params[4], params[5]])
    k_ads = np.array([params[6], params[7], params[8]])

    # A tolerance factor > 1 widens the half-saturation constant, i.e. weaker
    # inhibition.
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
    """Evaluate reactor ODE derivatives for concentrations and mass (reference).

    Concentrations and masses are floored before evaluation, because
    intermediate RK4 stages can overshoot below zero.

    Args:
        C: Concentration matrix [n_reactors, 15].
        M: Reactor mass vector [n_reactors].
        F_in: Inflow rates [n_reactors].
        F_out: Outflow rates [n_reactors].
        Cin: Inlet concentrations [n_reactors, 15].
        params: Kinetic parameter vector of length ``PARAM_DIM``.

    Returns:
        A tuple of:
            - dCdt: Concentration derivatives [n_reactors, 15].
            - dMdt: Mass derivatives [n_reactors].
    """
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
def integrate_rk4_reference(c_init, m_init, F_in, F_out, Cin, params, dt,
                            n_steps):
    """Reference RK4 built from ``ode_system``; allocates on every stage.

    Kept only to verify ``integrate_rk4`` (``smoke_test``), which must match
    it to floating-point rounding when every reactor's parameter row equals
    ``params``. Same returns as ``integrate_rk4``.

    Args:
        c_init: Initial concentrations [n_reactors, 15].
        m_init: Initial reactor masses [n_reactors].
        F_in: Inflow rates [n_reactors].
        F_out: Outflow rates [n_reactors].
        Cin: Inlet concentrations [n_reactors, 15].
        params: Kinetic parameter vector of length ``PARAM_DIM``, shared.
        dt: RK4 integration step size.
        n_steps: Number of RK4 sub-steps to execute.

    Returns:
        A tuple of:
            - c: Final concentrations [n_reactors, 15].
            - m: Final masses [n_reactors].
            - instability_detected: True on a NaN or a concentration above
              3000; integration stops at that sub-step.
            - severity: Per-reactor severity, shape [n_reactors]: 0 when
              stable, ``(max_c - 3000) / 3000`` for a blow-up, 1.0 for a NaN.
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


@njit(fastmath=True, cache=True)
def _free_enzyme(ce, k, emax, c_solid):
    """Free (unadsorbed) enzyme from the Langmuir quadratic, as in the kernel."""
    b = 1 + k * c_solid * emax - k * ce
    return (-b + np.sqrt(b**2 - 4*k*(-ce))) / (2*k + 1e-8)


@njit(fastmath=True, cache=True)
def _derivs(c, m, f_in, f_out, cin, params, cs, dc):
    """One reactor's derivatives, without allocating.

    The same arithmetic as ``ode_system`` / ``calculate_rates_numba``, written
    with scalars so the RK4 inner loop allocates nothing.

    Args:
        c: Concentrations, length 15.
        m: Reactor mass.
        f_in: Inflow rate.
        f_out: Outflow rate.
        cin: Inlet concentrations, length 15.
        params: This reactor's kinetic parameter vector, length ``PARAM_DIM``.
        cs: Scratch, length 15; receives the floored concentrations.
        dc: Output, length 15; receives dC/dt.

    Returns:
        dM/dt.
    """
    eps = 1e-8
    m_safe = m if m > 1.0 else 1.0
    for j in range(15):
        cs[j] = c[j] if c[j] > 1e-8 else 1e-8
    CS, XS, LS, Cb, G, X, Fu, E = cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7]
    Cell, Eth, ACT, HMF = cs[9], cs[10], cs[12], cs[13]
    eta, ph_g, ph_x = params[0], params[1], params[2]

    C_solid = CS + XS + LS + eps
    ce0, ce1, ce2 = 0.5 * E, 0.3 * E, 0.2 * E
    cef0 = _free_enzyme(ce0, params[6], params[3], C_solid)
    cef1 = _free_enzyme(ce1, params[7], params[4], C_solid)
    cef2 = _free_enzyme(ce2, params[8], params[5], C_solid)
    CebC1 = (ce0 - cef0) * (CS / C_solid)
    CebC2 = (ce1 - cef1) * (CS / C_solid)
    CebX3 = (ce2 - cef2) * (XS / C_solid)

    r1 = (0.00034 * eta * CebC1 * CS) / (1 + (Cb/0.0014) + (
        X/0.1007) + (G/0.073) + (Fu/10.0) + (Eth/0.15) + eps)
    r2 = (0.0023 * eta * (CebC1 + CebC2) * CS) / (
        1 + (Cb/132.0) + (X/0.029) + (G/0.34) + (Fu/10.0) + eps)
    r3 = (0.07 * eta * cef1 * Cb) / (24.3 * (1 +
                                             (X/201.0) + (G/3.9) + (Fu/10.0)) + Cb + eps)
    r4 = (0.0087 * eta * CebX3 * XS) / (1 + (Cb/24.3) +
                                        (X/201.0) + (G/2.34) + (Fu/10.0) + eps)
    r5 = 0.5 * r4

    monod_G = G / (1.342 + G + (G**2)/4890.0 + eps)
    eth_inh_G = max(0.0, 1 - (Eth/103.0)**1.42)
    kF_G, kACT_G, kHMF_G = 0.75 * params[9], 2.74 * params[10], 2.0 * params[11]
    tox_inh_G = (kF_G/(kF_G+Fu)) * \
        (kACT_G/(kACT_G+ACT)) * (kHMF_G/(kHMF_G+HMF))
    qG = (1/0.47) * (0.000318 * ph_g) * \
        Cell * monod_G * eth_inh_G * tox_inh_G

    monod_X = X / (3.4 + X + (X**2)/81.3 + eps)
    eth_inh_X = max(0.0, 1 - (Eth/100.2)**0.608)
    kF_X, kACT_X, kHMF_X = 0.35 * params[9], 0.2 * params[10], 10.0 * params[11]
    tox_inh_X = (kF_X/(kF_X+Fu)) * \
        (kACT_X/(kACT_X+ACT)) * (kHMF_X/(kHMF_X+HMF))
    qX = (1/0.4) * (0.00083444 * ph_x) * \
        Cell * monod_X * eth_inh_X * tox_inh_X

    qF = 4.6706E-5 * Cell * (Fu / (0.05 + Fu + eps))
    qHMF = (8.7576E-5 * Cell * (HMF /
            (HMF + 0.5 + eps))) * (0.25 / (0.25 + Fu + eps))
    qACT = 1.2292E-5 * Cell * \
        (ACT / (ACT + 2.5 + eps))
    sugar_sum = G + X + eps

    dc[0] = -r1-r2                                   # CS
    dc[1] = -r4-r5                                   # XS
    dc[2] = 0.0                                      # LS
    dc[3] = r1-r3                                    # C
    dc[4] = r2+r3-qG                                 # G
    dc[5] = r4-qX                                    # X
    dc[6] = -qF                                      # F
    dc[7] = 0.0                                      # E
    dc[8] = r5                                       # AC
    dc[9] = ((G/sugar_sum)*(qG - 2.6944E-5*Cell) *
             0.115 + (X/sugar_sum)*(qX - 1.8611E-5*Cell)*0.162)   # Cell
    dc[10] = qG*0.47 + qX*0.4                        # Eth
    dc[11] = qG*0.47 + qX*0.4 + qACT*0.1             # CO2
    dc[12] = qHMF*0.23392 - qACT                     # ACT
    dc[13] = -qHMF                                   # HMF
    dc[14] = 0.0                                     # Base
    for j in range(15):
        dc[j] = (f_in * (cin[j] - cs[j])) / (m_safe + 1e-8) + dc[j]
    return f_in - f_out


@njit(fastmath=True, cache=True)
def co2_formation_rates(C, params):
    """Each reactor's CO2 formation rate by reaction alone, per second.

    The ``dc[11]`` term of ``_derivs`` with no flow through the vessel, i.e.
    the reaction's CO2 formation rate (``calculate_rates_numba``'s CO2 column),
    evaluated with each reactor's own parameter row.

    Args:
        C: Concentrations [n_reactors, 15].
        params: Per-reactor kinetic parameters [n_reactors, PARAM_DIM].

    Returns:
        Array [n_reactors] of CO2 formation rates (concentration units per
        second).
    """
    n_reactors = C.shape[0]
    out = np.zeros(n_reactors)
    cs = np.empty(15)
    dc = np.empty(15)
    zero_in = np.zeros(15)
    for i in range(n_reactors):
        _derivs(C[i], 1.0e4, 0.0, 0.0, zero_in, params[i], cs, dc)
        out[i] = dc[11]
    return out


@njit(fastmath=True, cache=True)
def integrate_rk4(c_init, m_init, F_in, F_out, Cin, params, dt, n_steps):
    """Integrate reactor dynamics with fixed-step RK4, allocation-free.

    Numerically equivalent to ``integrate_rk4_reference`` when every row of
    ``params`` equals the reference's vector, but reuses scratch buffers
    instead of allocating ~50 small arrays per sub-step, and reads one
    parameter row per reactor.

    Args:
        c_init: Initial concentrations [n_reactors, 15].
        m_init: Initial reactor masses [n_reactors].
        F_in: Inflow rates [n_reactors].
        F_out: Outflow rates [n_reactors].
        Cin: Inlet concentrations [n_reactors, 15].
        params: Per-reactor kinetic parameters [n_reactors, PARAM_DIM].
        dt: RK4 integration step size.
        n_steps: Number of RK4 sub-steps to execute.

    Returns:
        A tuple of:
            - c: Final concentrations [n_reactors, 15].
            - m: Final masses [n_reactors].
            - instability_detected: True on a NaN or a concentration above
              3000; integration stops at that sub-step.
            - severity: Per-reactor severity, shape [n_reactors]: 0 when
              stable, ``(max_c - 3000) / 3000`` for a blow-up, 1.0 for a NaN.
    """
    c = c_init.copy()
    m = m_init.copy()
    n_reactors = c.shape[0]
    severity = np.zeros(n_reactors)
    cs = np.empty(15)
    tmp = np.empty(15)
    k1 = np.empty(15)
    k2 = np.empty(15)
    k3 = np.empty(15)
    k4 = np.empty(15)

    for _ in range(n_steps):
        for i in range(n_reactors):
            ci = c[i]
            cin = Cin[i]
            prm = params[i]
            fin = F_in[i]
            fout = F_out[i]
            mi = m[i]
            dm1 = _derivs(ci, mi, fin, fout, cin, prm, cs, k1)
            for j in range(15):
                tmp[j] = ci[j] + 0.5 * dt * k1[j]
            dm2 = _derivs(tmp, mi + 0.5 * dt * dm1, fin, fout, cin, prm, cs, k2)
            for j in range(15):
                tmp[j] = ci[j] + 0.5 * dt * k2[j]
            dm3 = _derivs(tmp, mi + 0.5 * dt * dm2, fin, fout, cin, prm, cs, k3)
            for j in range(15):
                tmp[j] = ci[j] + dt * k3[j]
            dm4 = _derivs(tmp, mi + dt * dm3, fin, fout, cin, prm, cs, k4)

            for j in range(15):
                ci[j] += (dt / 6.0) * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j])
                if ci[j] < 1e-8:
                    ci[j] = 1e-8
            mi += (dt / 6.0) * (dm1 + 2.0*dm2 + 2.0*dm3 + dm4)
            if mi < 1.0:
                mi = 1.0
            m[i] = mi

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
