from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.special import gammaln


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class Params:
    """Dimensionless refrigerator benchmark with hbar = k_B = nu = 1."""

    nu: float = 1.0
    omega0: float = 3.0
    eta: float = 0.05

    beta_h_times_omega0: float = 1.40
    beta_c_times_omega_minus: float = 1.00

    spectral_amplitude_h: float = 1.0e-5
    spectral_amplitude_c: float = 8.5e-5
    Gamma_h: float = 1.0e-3
    Gamma_c: float = 1.0e-3
    delta_h: float = 5.0e-3
    delta_c: float = 5.0e-3
    window_h: float = 1.0e-2
    window_c: float = 1.0e-2

    n0_initial: float = 6.0

    rate_scan_end: float = 5.0e3
    rate_time_points: int = 14001
    n_frequency_grid: int = 6001

    propagation_end: float = 735.0
    num_time_points: int = 801

    Np: int = 40
    reduced_rtol: float = 1.0e-10
    reduced_atol: float = 1.0e-13
    joint_rtol: float = 2.0e-9
    joint_atol: float = 2.0e-12
    joint_max_step: float = 2.0

    dpi_save: int = 600

    @property
    def omega_minus(self) -> float:
        return self.omega0 - self.nu

    @property
    def beta_h(self) -> float:
        return self.beta_h_times_omega0 / self.omega0

    @property
    def beta_c(self) -> float:
        return self.beta_c_times_omega_minus / self.omega_minus

    @property
    def sideband_prefactor(self) -> float:
        return 4.0 * self.eta**2

    @property
    def alpha0(self) -> float:
        return float(np.sqrt(self.n0_initial))

    @property
    def pe0(self) -> float:
        Gh = self.spectral_amplitude_h * self.Gamma_h**2 / (
            self.delta_h**2 + self.Gamma_h**2
        )
        Gc = self.spectral_amplitude_c * self.Gamma_c**2 / (
            self.delta_c**2 + self.Gamma_c**2
        )

        r_h_down = 2.0 * np.pi * Gh
        r_h_up = np.exp(-self.beta_h_times_omega0) * r_h_down
        r_c_down = self.sideband_prefactor * 2.0 * np.pi * Gc
        r_c_up = np.exp(-self.beta_c_times_omega_minus) * r_c_down
        n0 = self.n0_initial

        return float(
            (r_h_up + r_c_up * n0)
            / (r_h_down + r_h_up + r_c_up * n0 + r_c_down * (n0 + 1.0))
        )


P = Params()
OUTPUT_DIR = Path("refrigerator_figure_output")


# =============================================================================
# Plot style
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.8,
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.05,
        "figure.dpi": 150,
        "savefig.dpi": P.dpi_save,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "finite_reduced": "#1f77b4",
    "finite_joint": "#0b3c5d",
    "markov_reduced": "#d62728",
    "markov_joint": "#7f1d1d",
    "zero": "#6e6e6e",
}


# =============================================================================
# Reservoir spectra and finite-time rates
# =============================================================================

def support_info(
    bath: str, p: Params = P
) -> tuple[float, float, float, float, float, float]:
    if bath == "h":
        target = p.omega0
        peak = p.omega0 + p.delta_h
        left = target - p.window_h
        right = target + p.window_h
        amplitude = p.spectral_amplitude_h
        beta = p.beta_h
    elif bath == "c":
        target = p.omega_minus
        peak = p.omega_minus - p.delta_c
        left = target - p.window_c
        right = target + p.window_c
        amplitude = p.spectral_amplitude_c
        beta = p.beta_c
    else:
        raise ValueError("bath must be 'h' or 'c'")

    return target, peak, left, right, amplitude, beta


def positive_spectrum(xi: np.ndarray, bath: str, p: Params = P) -> np.ndarray:
    _, peak, left, right, amplitude, _ = support_info(bath, p)
    linewidth = p.Gamma_h if bath == "h" else p.Gamma_c
    xi = np.asarray(xi, dtype=float)
    lorentzian = amplitude * linewidth**2 / ((xi - peak) ** 2 + linewidth**2)
    return lorentzian * ((xi >= left) & (xi <= right))


def spectrum_value(omega: float, bath: str, p: Params = P) -> float:
    _, _, _, _, _, beta = support_info(bath, p)
    x = abs(float(omega))
    positive = float(positive_spectrum(np.array([x]), bath, p)[0])
    return positive if omega >= 0.0 else float(np.exp(-beta * x) * positive)


def markov_rate(omega: float, bath: str, p: Params = P) -> float:
    return float(2.0 * np.pi * spectrum_value(omega, bath, p))


def sin_over_delta(delta: np.ndarray, time: float) -> np.ndarray:
    return float(time) * np.sinc(np.asarray(delta) * float(time) / np.pi)


def instantaneous_rate(omega: float, time: float, bath: str, p: Params = P) -> float:
    _, _, left, right, _, beta = support_info(bath, p)
    xi = np.linspace(left, right, p.n_frequency_grid)
    positive = positive_spectrum(xi, bath, p)

    positive_branch = positive * sin_over_delta(omega - xi, time)
    negative_branch = np.exp(-beta * xi) * positive * sin_over_delta(omega + xi, time)

    return float(2.0 * np.trapezoid(positive_branch + negative_branch, xi))


@dataclass
class RateTable:
    time: np.ndarray
    h_down: PchipInterpolator
    h_up: PchipInterpolator
    c_down: PchipInterpolator
    c_up: PchipInterpolator

    def retained(self, time: float, p: Params = P) -> tuple[float, float, float, float]:
        return (
            float(self.h_down(time)),
            float(self.h_up(time)),
            p.sideband_prefactor * float(self.c_down(time)),
            p.sideband_prefactor * float(self.c_up(time)),
        )


def build_rate_table(p: Params = P) -> RateTable:
    time = np.linspace(0.0, p.rate_scan_end, p.rate_time_points)

    h_down = np.empty_like(time)
    h_up = np.empty_like(time)
    c_down = np.empty_like(time)
    c_up = np.empty_like(time)

    for i, t in enumerate(time):
        h_down[i] = instantaneous_rate(+p.omega0, t, "h", p)
        h_up[i] = instantaneous_rate(-p.omega0, t, "h", p)
        c_down[i] = instantaneous_rate(+p.omega_minus, t, "c", p)
        c_up[i] = instantaneous_rate(-p.omega_minus, t, "c", p)

    return RateTable(
        time=time,
        h_down=PchipInterpolator(time, h_down),
        h_up=PchipInterpolator(time, h_up),
        c_down=PchipInterpolator(time, c_down),
        c_up=PchipInterpolator(time, c_up),
    )


def markov_retained_rates(p: Params = P) -> tuple[float, float, float, float]:
    return (
        markov_rate(+p.omega0, "h", p),
        markov_rate(-p.omega0, "h", p),
        p.sideband_prefactor * markov_rate(+p.omega_minus, "c", p),
        p.sideband_prefactor * markov_rate(-p.omega_minus, "c", p),
    )


# =============================================================================
# Reduced refrigerator dynamics
# =============================================================================

def solve_reduced(
    case: str,
    times: np.ndarray,
    rate_table: RateTable,
    p: Params = P,
) -> pd.DataFrame:
    if case not in {"FT", "M"}:
        raise ValueError("case must be 'FT' or 'M'")

    markov_rates = markov_retained_rates(p)

    def rates_at(t: float) -> tuple[float, float, float, float]:
        return rate_table.retained(t, p) if case == "FT" else markov_rates

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        pe, n_p, alpha, _q_c = y
        pg = 1.0 - pe
        r_h_down, r_h_up, r_c_down, r_c_up = rates_at(t)

        cooling = r_c_up * pg * n_p
        reverse = r_c_down * pe * (n_p + 1.0)
        gamma_p = r_c_up * pg - r_c_down * pe

        dpe = -r_h_down * pe + r_h_up * pg + cooling - reverse
        dn_p = reverse - cooling
        dalpha = -0.5 * gamma_p * alpha
        j_c = p.omega_minus * (cooling - reverse)

        return np.array([dpe, dn_p, dalpha, j_c], dtype=float)

    y0 = np.array([p.pe0, p.n0_initial, p.alpha0, 0.0], dtype=float)
    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        y0,
        t_eval=times,
        method="DOP853",
        rtol=p.reduced_rtol,
        atol=p.reduced_atol,
    )

    pe, n_p, _alpha, q_c = solution.y
    j_c = np.empty_like(times)

    for i, t in enumerate(times):
        pg = 1.0 - pe[i]
        _, _, r_c_down, r_c_up = rates_at(float(t))
        cooling = r_c_up * pg * n_p[i]
        reverse = r_c_down * pe[i] * (n_p[i] + 1.0)
        j_c[i] = p.omega_minus * (cooling - reverse)

    return pd.DataFrame(
        {
            "t": times,
            "Jc": j_c,
            "Qc_quanta": q_c / p.omega_minus,
        }
    )


# =============================================================================
# Joint TLS-piston dynamics
# =============================================================================

def destroy(dimension: int) -> sparse.csr_matrix:
    data = np.sqrt(np.arange(1, dimension, dtype=float))
    return sparse.diags(data, offsets=1, shape=(dimension, dimension), format="csr")


def coherent_ket(alpha: complex, dimension: int) -> np.ndarray:
    n = np.arange(dimension)
    coefficients = np.exp(
        -abs(alpha) ** 2 / 2
        + n * np.log(alpha)
        - 0.5 * gammaln(n + 1)
    ).astype(complex)
    return coefficients / np.linalg.norm(coefficients)


def dissipator_superoperator(L: sparse.csr_matrix, dimension: int) -> sparse.csr_matrix:
    identity = sparse.identity(dimension, format="csr", dtype=complex)
    ldag_l = (L.getH() @ L).tocsr()
    return (
        sparse.kron(L.conjugate(), L, format="csr")
        - 0.5 * sparse.kron(identity, ldag_l, format="csr")
        - 0.5 * sparse.kron(ldag_l.transpose(), identity, format="csr")
    )


@dataclass
class JointOperators:
    D_h_down: sparse.csr_matrix
    D_h_up: sparse.csr_matrix
    D_c_down: sparse.csr_matrix
    D_c_up: sparse.csr_matrix
    L_c_down_dagL: np.ndarray
    L_c_up_dagL: np.ndarray


def build_joint_operators(dimension: int) -> JointOperators:
    identity_p = sparse.identity(dimension, format="csr", dtype=complex)
    a = destroy(dimension)
    adag = a.getH()

    sigma_minus = sparse.csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))
    sigma_plus = sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))

    L_h_down = sparse.kron(sigma_minus, identity_p, format="csr")
    L_h_up = sparse.kron(sigma_plus, identity_p, format="csr")
    L_c_down = sparse.kron(sigma_minus, adag, format="csr")
    L_c_up = sparse.kron(sigma_plus, a, format="csr")

    total_dimension = 2 * dimension
    return JointOperators(
        D_h_down=dissipator_superoperator(L_h_down, total_dimension),
        D_h_up=dissipator_superoperator(L_h_up, total_dimension),
        D_c_down=dissipator_superoperator(L_c_down, total_dimension),
        D_c_up=dissipator_superoperator(L_c_up, total_dimension),
        L_c_down_dagL=(L_c_down.getH() @ L_c_down).toarray(),
        L_c_up_dagL=(L_c_up.getH() @ L_c_up).toarray(),
    )


def initial_joint_state(dimension: int, p: Params = P) -> np.ndarray:
    rho_tls = np.array([[p.pe0, 0.0], [0.0, 1.0 - p.pe0]], dtype=complex)
    piston_ket = coherent_ket(p.alpha0, dimension)
    rho_piston = np.outer(piston_ket, piston_ket.conjugate())
    rho = np.kron(rho_tls, rho_piston)
    return rho.reshape(-1, order="F")


def solve_joint(
    case: str,
    times: np.ndarray,
    rate_table: RateTable,
    p: Params = P,
) -> pd.DataFrame:
    if case not in {"FT", "M"}:
        raise ValueError("case must be 'FT' or 'M'")

    operators = build_joint_operators(p.Np)
    markov_rates = markov_retained_rates(p)
    matrix_dimension = 2 * p.Np

    def rates_at(t: float) -> tuple[float, float, float, float]:
        return rate_table.retained(t, p) if case == "FT" else markov_rates

    def rhs(t: float, rho_vector: np.ndarray) -> np.ndarray:
        r_h_down, r_h_up, r_c_down, r_c_up = rates_at(t)
        return (
            r_h_down * (operators.D_h_down @ rho_vector)
            + r_h_up * (operators.D_h_up @ rho_vector)
            + r_c_down * (operators.D_c_down @ rho_vector)
            + r_c_up * (operators.D_c_up @ rho_vector)
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        initial_joint_state(p.Np, p),
        t_eval=times,
        method="DOP853",
        rtol=p.joint_rtol,
        atol=p.joint_atol,
        max_step=p.joint_max_step,
    )

    j_c = np.empty_like(times)
    q_c = np.zeros_like(times)

    for i, (t, rho_vector) in enumerate(zip(times, solution.y.T)):
        rho = rho_vector.reshape((matrix_dimension, matrix_dimension), order="F")
        rho = 0.5 * (rho + rho.conjugate().T)
        rho = rho / np.trace(rho)

        _, _, r_c_down, r_c_up = rates_at(float(t))
        cooling = r_c_up * float(np.real(np.trace(operators.L_c_up_dagL @ rho)))
        reverse = r_c_down * float(np.real(np.trace(operators.L_c_down_dagL @ rho)))
        j_c[i] = p.omega_minus * (cooling - reverse)

        if i > 0:
            q_c[i] = q_c[i - 1] + 0.5 * (j_c[i - 1] + j_c[i]) * (times[i] - times[i - 1])

    return pd.DataFrame(
        {
            "t": times,
            "Jc": j_c,
            "Qc_quanta": q_c / p.omega_minus,
        }
    )


# =============================================================================
# Figures
# =============================================================================

def style_axis(ax) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.0)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.5, width=0.8)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.20, linewidth=0.55)
    ax.grid(True, which="minor", alpha=0.08, linewidth=0.40)


def apply_scientific_formatter(ax) -> None:
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)


def add_panel_label(ax, label: str, x: float, y: float, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.5),
        zorder=20,
    )


def save_panel(fig, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    reduced_ft: pd.DataFrame,
    reduced_m: pd.DataFrame,
    joint_ft: pd.DataFrame,
    joint_m: pd.DataFrame,
    p: Params = P,
) -> None:
    time = reduced_ft["t"].to_numpy()
    marker_stride = max(1, len(joint_ft) // 35)

    current_scale = max(float(np.max(np.abs(reduced_m["Jc"]))), 1.0e-30)

    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    ax.plot(
        time,
        reduced_ft["Jc"] / current_scale,
        color=COLORS["finite_reduced"],
        label="finite time reduced",
    )
    ax.plot(
        time,
        reduced_m["Jc"] / current_scale,
        color=COLORS["markov_reduced"],
        linestyle="--",
        label="Markovian reduced",
    )
    ax.plot(
        joint_ft["t"][::marker_stride],
        joint_ft["Jc"][::marker_stride] / current_scale,
        "o",
        ms=3.0,
        color=COLORS["finite_joint"],
        label="finite time joint",
    )
    ax.plot(
        joint_m["t"][::marker_stride],
        joint_m["Jc"][::marker_stride] / current_scale,
        "s",
        ms=2.8,
        color=COLORS["markov_joint"],
        label="Markovian joint",
    )
    ax.axhline(0.0, linewidth=1.0, color=COLORS["zero"], linestyle=":")
    ax.set_xlabel(r"$\nu s$")
    ax.set_ylabel(r"$\mathcal{J}_c(s)/\max|\mathcal{J}_c^{\rm M}|$")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    add_panel_label(ax, "(a)", x=0.05, y=0.95)
    save_panel(fig, "refrigerator_panel_a_Jc.pdf")

    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    ax.plot(
        time,
        reduced_ft["Qc_quanta"],
        color=COLORS["finite_reduced"],
        label="finite time reduced",
    )
    ax.plot(
        time,
        reduced_m["Qc_quanta"],
        color=COLORS["markov_reduced"],
        linestyle="--",
        label="Markovian reduced",
    )
    ax.plot(
        joint_ft["t"][::marker_stride],
        joint_ft["Qc_quanta"][::marker_stride],
        "o",
        ms=3.0,
        color=COLORS["finite_joint"],
        label="finite time joint",
    )
    ax.plot(
        joint_m["t"][::marker_stride],
        joint_m["Qc_quanta"][::marker_stride],
        "s",
        ms=2.8,
        color=COLORS["markov_joint"],
        label="Markovian joint",
    )
    ax.set_xlabel(r"$\nu s$")
    ax.set_ylabel(r"$Q_c(s)/(\hbar\omega_-)$")
    apply_scientific_formatter(ax)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    add_panel_label(ax, "(b)", x=0.95, y=0.22, ha="right")
    save_panel(fig, "refrigerator_panel_b_Qc.pdf")


# =============================================================================
# Execution
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rate_table = build_rate_table(P)
    times = np.linspace(0.0, P.propagation_end, P.num_time_points)

    reduced_ft = solve_reduced("FT", times, rate_table, P)
    reduced_m = solve_reduced("M", times, rate_table, P)
    joint_ft = solve_joint("FT", times, rate_table, P)
    joint_m = solve_joint("M", times, rate_table, P)

    make_figures(reduced_ft, reduced_m, joint_ft, joint_m, P)


if __name__ == "__main__":
    main()
