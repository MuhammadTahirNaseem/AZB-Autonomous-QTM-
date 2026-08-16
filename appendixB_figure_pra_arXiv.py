from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, diags, eye, kron


OUTPUT_DIR = Path("appendixB_figure_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Params:
    """Dimensionless parameters with hbar = k_B = nu = 1."""

    nu: float = 1.0
    omega0: float = 3.0
    zeta: float = 0.095

    alpha0_abs2: float = 1.0
    fock_m: int = 1
    sinh2_r: float = 1.0
    nbar_thermal: float = 1.0

    beta_h_times_omega0: float = 0.35
    beta_c_times_omega_minus: float = 1.50

    Gamma_h: float = 1.60e-3
    Gamma_c: float = 1.10e-3
    delta_h: float = 5.00e-3
    delta_c: float = 3.80e-3
    window_h: float = 8.50e-3
    window_c: float = 7.50e-3
    spectral_amplitude: float = 1.0e-5

    tau_min_plot: float = 20.0
    tau_max: float = 4.0e4
    n_plot_points: int = 950

    n_frequency_grid: int = 4001
    rate_chunk_size: int = 192
    fock_cutoff: int = 80

    density_matrix_cutoff: int = 60
    density_matrix_markers: int = 24
    density_matrix_rtol: float = 1.0e-10
    density_matrix_atol: float = 1.0e-12

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
        return 4.0 * self.zeta**2


P = Params()


plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 10.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "lines.linewidth": 1.9,
    "axes.linewidth": 0.9,
    "figure.dpi": 180,
    "savefig.dpi": 900,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def trapz(y: np.ndarray, x: np.ndarray, axis: int = -1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def cumulative_trapezoid_from_zero(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(t))
    return out


def make_time_grids(p: Params = P) -> tuple[np.ndarray, np.ndarray]:
    tau_plot = np.geomspace(p.tau_min_plot, p.tau_max, p.n_plot_points)
    early = np.linspace(0.0, 50.0, 501)
    long = np.geomspace(50.0, p.tau_max, 3001)
    peak = np.linspace(400.0, 900.0, 1001)
    tau = np.unique(np.concatenate([early, long, peak, tau_plot]))
    tau.sort()
    return tau, tau_plot


def positive_filtered_lorentzian(
    x: np.ndarray,
    *,
    center: float,
    target: float,
    linewidth: float,
    half_width: float,
    p: Params = P,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lorentzian = (
        p.spectral_amplitude
        * linewidth**2
        / ((x - center) ** 2 + linewidth**2)
    )
    window = (np.abs(x - target) <= half_width).astype(float)
    return lorentzian * window


def spectrum_two_sided(
    omega: np.ndarray,
    bath: str,
    p: Params = P,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=float)
    x = np.abs(omega)

    if bath == "h":
        target = p.omega0
        center = p.omega0 + p.delta_h
        linewidth = p.Gamma_h
        half_width = p.window_h
        beta = p.beta_h
    elif bath == "c":
        target = p.omega_minus
        center = p.omega_minus - p.delta_c
        linewidth = p.Gamma_c
        half_width = p.window_c
        beta = p.beta_c
    else:
        raise ValueError("bath must be 'h' or 'c'")

    positive = positive_filtered_lorentzian(
        x,
        center=center,
        target=target,
        linewidth=linewidth,
        half_width=half_width,
        p=p,
    )
    return np.where(omega >= 0.0, positive, np.exp(-beta * x) * positive)


def frequency_grid_for_bath(
    bath: str,
    n_frequency_grid: int,
    p: Params = P,
) -> np.ndarray:
    if bath == "h":
        target, half_width = p.omega0, p.window_h
    elif bath == "c":
        target, half_width = p.omega_minus, p.window_c
    else:
        raise ValueError("bath must be 'h' or 'c'")

    return np.linspace(
        target - half_width,
        target + half_width,
        n_frequency_grid,
    )


def instantaneous_rate_array(
    omega: float,
    times: np.ndarray,
    bath: str,
    *,
    n_frequency_grid: int,
    p: Params = P,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    x = frequency_grid_for_bath(bath, n_frequency_grid, p)
    spectrum_positive = spectrum_two_sided(x, bath, p)
    spectrum_negative = spectrum_two_sided(-x, bath, p)

    delta_positive = omega - x
    delta_negative = omega + x
    rates = np.empty_like(times)

    for start in range(0, len(times), p.rate_chunk_size):
        stop = min(start + p.rate_chunk_size, len(times))
        tc = times[start:stop]

        phase_positive = tc[:, None] * delta_positive[None, :]
        phase_negative = tc[:, None] * delta_negative[None, :]

        kernel_positive = np.empty_like(phase_positive)
        kernel_negative = np.empty_like(phase_negative)

        small_positive = np.abs(delta_positive) < 1.0e-14
        small_negative = np.abs(delta_negative) < 1.0e-14

        kernel_positive[:, ~small_positive] = (
            np.sin(phase_positive[:, ~small_positive])
            / delta_positive[None, ~small_positive]
        )
        if np.any(small_positive):
            kernel_positive[:, small_positive] = tc[:, None]

        kernel_negative[:, ~small_negative] = (
            np.sin(phase_negative[:, ~small_negative])
            / delta_negative[None, ~small_negative]
        )
        if np.any(small_negative):
            kernel_negative[:, small_negative] = tc[:, None]

        integrand = (
            kernel_positive * spectrum_positive[None, :]
            + kernel_negative * spectrum_negative[None, :]
        )
        rates[start:stop] = 2.0 * trapz(integrand, x, axis=1)

    return rates


def markov_rate(omega: float, bath: str, p: Params = P) -> float:
    return float(
        2.0
        * np.pi
        * spectrum_two_sided(np.array([omega]), bath, p)[0]
    )


def build_channel(p: Params = P):
    tau, tau_plot = make_time_grids(p)

    gamma_h_down = instantaneous_rate_array(
        +p.omega0,
        tau,
        "h",
        n_frequency_grid=p.n_frequency_grid,
        p=p,
    )
    gamma_h_up = instantaneous_rate_array(
        -p.omega0,
        tau,
        "h",
        n_frequency_grid=p.n_frequency_grid,
        p=p,
    )
    gamma_c_down = p.sideband_prefactor * instantaneous_rate_array(
        +p.omega_minus,
        tau,
        "c",
        n_frequency_grid=p.n_frequency_grid,
        p=p,
    )
    gamma_c_up = p.sideband_prefactor * instantaneous_rate_array(
        -p.omega_minus,
        tau,
        "c",
        n_frequency_grid=p.n_frequency_grid,
        p=p,
    )

    gamma_h_down_m = markov_rate(+p.omega0, "h", p)
    gamma_h_up_m = markov_rate(-p.omega0, "h", p)
    gamma_c_down_m = p.sideband_prefactor * markov_rate(+p.omega_minus, "c", p)
    gamma_c_up_m = p.sideband_prefactor * markov_rate(-p.omega_minus, "c", p)

    pe_m = gamma_h_up_m / (gamma_h_down_m + gamma_h_up_m)
    pg_m = 1.0 - pe_m

    denominator = gamma_h_down + gamma_h_up
    pe_h = np.full_like(denominator, pe_m)
    mask = denominator > 0.0
    pe_h[mask] = gamma_h_up[mask] / denominator[mask]
    pg_h = 1.0 - pe_h

    D = gamma_c_down * pe_h
    B = gamma_c_up * pg_h
    Lambda = D - B

    D_m = gamma_c_down_m * pe_m
    B_m = gamma_c_up_m * pg_m
    Lambda_m = D_m - B_m

    K = cumulative_trapezoid_from_zero(Lambda, tau)
    G = np.exp(K)
    N = G * cumulative_trapezoid_from_zero(
        D / np.maximum(G, 1.0e-300),
        tau,
    )

    gain_factor = np.full_like(tau, np.nan)
    positive_time = tau > 0.0
    gain_factor[positive_time] = (
        K[positive_time] / tau[positive_time]
    ) / Lambda_m

    peak_mask = tau >= p.tau_min_plot
    peak_indices = np.where(peak_mask)[0]
    peak_index = peak_indices[np.nanargmax(gain_factor[peak_mask])]
    tau_star = float(tau[peak_index])

    plot_data = pd.DataFrame({
        "tau": tau_plot,
        "nu_tau": p.nu * tau_plot,
        "G": np.interp(tau_plot, tau, G),
        "N": np.interp(tau_plot, tau, N),
    })

    channel = {
        "tau": tau,
        "D": D,
        "B": B,
        "G": G,
        "N": N,
        "tau_star": tau_star,
        "nu_tau_star": p.nu * tau_star,
        "D_m": D_m,
        "B_m": B_m,
        "Lambda_m": Lambda_m,
    }
    return plot_data, channel


def gaussian_ergotropy(
    alpha_sq: np.ndarray | float,
    n_centered: np.ndarray | float,
    m_centered_abs: np.ndarray | float,
) -> np.ndarray:
    alpha_sq = np.asarray(alpha_sq, dtype=float)
    n_centered = np.asarray(n_centered, dtype=float)
    m_centered_abs = np.asarray(m_centered_abs, dtype=float)
    symplectic = np.sqrt(
        np.maximum((n_centered + 0.5) ** 2 - m_centered_abs**2, 0.0)
    )
    return alpha_sq + n_centered - symplectic + 0.5


def gaussian_state_curves(
    G: np.ndarray,
    N: np.ndarray,
    p: Params = P,
) -> dict[str, np.ndarray]:
    m0_abs = math.sqrt(p.sinh2_r * (p.sinh2_r + 1.0))

    return {
        "coherent": gaussian_ergotropy(
            p.alpha0_abs2 * G,
            N,
            np.zeros_like(G),
        ),
        "squeezed": gaussian_ergotropy(
            np.zeros_like(G),
            G * p.sinh2_r + N,
            G * m0_abs,
        ),
        "thermal": gaussian_ergotropy(
            np.zeros_like(G),
            G * p.nbar_thermal + N,
            np.zeros_like(G),
        ),
    }


def birth_death_rhs(
    probabilities: np.ndarray,
    gain_rate: float,
    loss_rate: float,
) -> np.ndarray:
    nmax = len(probabilities) - 1
    n = np.arange(nmax + 1, dtype=float)
    rhs = -(gain_rate * (n + 1.0) + loss_rate * n) * probabilities
    rhs[1:] += gain_rate * np.arange(1, nmax + 1, dtype=float) * probabilities[:-1]
    rhs[:-1] += loss_rate * np.arange(1, nmax + 1, dtype=float) * probabilities[1:]
    return rhs


def fock_ergotropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    n = np.arange(len(probabilities), dtype=float)
    energy = float(np.dot(n, probabilities))
    passive_energy = float(np.dot(n, np.sort(probabilities)[::-1]))
    return energy - passive_energy


def propagate_fock_time_dependent(
    tau_channel: np.ndarray,
    D_channel: np.ndarray,
    B_channel: np.ndarray,
    t_eval: np.ndarray,
    *,
    m: int,
    cutoff: int,
) -> np.ndarray:
    p0 = np.zeros(cutoff + 1)
    p0[m] = 1.0

    def rhs(t, y):
        D = float(np.interp(t, tau_channel, D_channel))
        B = float(np.interp(t, tau_channel, B_channel))
        return birth_death_rhs(y, D, B)

    solution = solve_ivp(
        rhs,
        (0.0, float(np.max(t_eval))),
        p0,
        method="DOP853",
        t_eval=np.asarray(t_eval, dtype=float),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    return np.array([fock_ergotropy(row) for row in solution.y.T])


def propagate_fock_markovian(
    t_eval: np.ndarray,
    *,
    D_m: float,
    B_m: float,
    m: int,
    cutoff: int,
) -> np.ndarray:
    p0 = np.zeros(cutoff + 1)
    p0[m] = 1.0

    def rhs(_t, y):
        return birth_death_rhs(y, D_m, B_m)

    solution = solve_ivp(
        rhs,
        (0.0, float(np.max(t_eval))),
        p0,
        method="DOP853",
        t_eval=np.asarray(t_eval, dtype=float),
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    return np.array([fock_ergotropy(row) for row in solution.y.T])


def oscillator_operators(cutoff: int):
    a = diags(
        np.sqrt(np.arange(1, cutoff, dtype=float)),
        offsets=1,
        shape=(cutoff, cutoff),
        dtype=complex,
        format="csc",
    )
    ident = eye(cutoff, dtype=complex, format="csc")
    return a, a.getH(), ident


def lindblad_superoperator(L: csc_matrix, ident: csc_matrix) -> csc_matrix:
    C = L.getH() @ L
    return (
        kron(L.conjugate(), L, format="csc")
        - 0.5 * kron(ident, C, format="csc")
        - 0.5 * kron(C.transpose(), ident, format="csc")
    )


def coherent_density(cutoff: int, alpha_abs2: float) -> np.ndarray:
    alpha = math.sqrt(alpha_abs2)
    coeff = np.empty(cutoff, dtype=complex)
    coeff[0] = math.exp(-0.5 * alpha_abs2)
    for n in range(1, cutoff):
        coeff[n] = coeff[n - 1] * alpha / math.sqrt(n)
    coeff /= np.linalg.norm(coeff)
    return np.outer(coeff, coeff.conjugate())


def squeezed_density(
    cutoff: int,
    sinh2_r: float,
    phi: float = 0.0,
) -> np.ndarray:
    r = math.asinh(math.sqrt(sinh2_r))
    t = math.tanh(r)
    phase = np.exp(1j * phi)

    coeff = np.zeros(cutoff, dtype=complex)
    coeff[0] = 1.0 / math.sqrt(math.cosh(r))

    k = 0
    while 2 * (k + 1) < cutoff:
        coeff[2 * (k + 1)] = (
            coeff[2 * k]
            * (-phase)
            * t
            * math.sqrt((2 * k + 2) * (2 * k + 1))
            / (2.0 * (k + 1))
        )
        k += 1

    coeff /= np.linalg.norm(coeff)
    return np.outer(coeff, coeff.conjugate())


def fock_density(cutoff: int, m: int) -> np.ndarray:
    rho = np.zeros((cutoff, cutoff), dtype=complex)
    rho[m, m] = 1.0
    return rho


def thermal_density(cutoff: int, nbar: float) -> np.ndarray:
    q = nbar / (1.0 + nbar)
    probabilities = (1.0 - q) * q ** np.arange(cutoff, dtype=float)
    probabilities /= np.sum(probabilities)
    return np.diag(probabilities.astype(complex))


def passive_ergotropy(rho: np.ndarray, nu: float) -> float:
    rho_h = 0.5 * (rho + rho.conjugate().T)
    energies = nu * np.arange(rho.shape[0], dtype=float)
    energy = float(np.real(np.dot(energies, np.diag(rho_h))))
    eigenvalues = np.linalg.eigvalsh(rho_h)
    passive_energy = float(np.dot(energies, np.sort(eigenvalues)[::-1]))
    return energy - passive_energy


def density_matrix_markers(
    tau_channel: np.ndarray,
    D_channel: np.ndarray,
    B_channel: np.ndarray,
    tau_plot: np.ndarray,
    p: Params = P,
) -> dict:
    cutoff = p.density_matrix_cutoff
    n2 = cutoff * cutoff

    marker_indices = np.unique(
        np.linspace(
            0,
            len(tau_plot) - 1,
            p.density_matrix_markers,
            dtype=int,
        )
    )
    marker_tau = np.asarray(tau_plot[marker_indices], dtype=float)
    t_eval = np.concatenate(([0.0], marker_tau))

    a, adag, ident = oscillator_operators(cutoff)
    L_gain = lindblad_superoperator(adag, ident)
    L_loss = lindblad_superoperator(a, ident)

    state_names = ("Coherent", "Squeezed", "Fock", "Thermal")
    rho0 = {
        "Coherent": coherent_density(cutoff, p.alpha0_abs2),
        "Squeezed": squeezed_density(cutoff, p.sinh2_r),
        "Fock": fock_density(cutoff, p.fock_m),
        "Thermal": thermal_density(cutoff, p.nbar_thermal),
    }

    Y0 = np.column_stack(
        [rho0[name].reshape(n2, order="F") for name in state_names]
    )
    y0 = Y0.reshape(-1, order="F")

    def rhs(t, y):
        Y = y.reshape((n2, len(state_names)), order="F")
        D = float(np.interp(t, tau_channel, D_channel))
        B = float(np.interp(t, tau_channel, B_channel))
        dY = D * (L_gain @ Y) + B * (L_loss @ Y)
        return np.asarray(dY).reshape(-1, order="F")

    solution = solve_ivp(
        rhs,
        (0.0, float(marker_tau[-1])),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=p.density_matrix_rtol,
        atol=p.density_matrix_atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    initial_ergotropy = {
        name: passive_ergotropy(rho0[name], p.nu)
        for name in state_names
    }
    delta_w = {name: np.empty(len(marker_tau)) for name in state_names}

    for j in range(1, len(t_eval)):
        Y = solution.y[:, j].reshape((n2, len(state_names)), order="F")
        for k, name in enumerate(state_names):
            rho = Y[:, k].reshape((cutoff, cutoff), order="F")
            delta_w[name][j - 1] = (
                passive_ergotropy(rho, p.nu) - initial_ergotropy[name]
            )

    return {
        "indices": marker_indices,
        "coherent": delta_w["Coherent"],
        "squeezed": delta_w["Squeezed"],
        "fock": delta_w["Fock"],
        "thermal": delta_w["Thermal"],
    }


def markovian_gain_noise(
    t: np.ndarray | float,
    *,
    D_m: float,
    Lambda_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    G = np.exp(Lambda_m * t)
    if abs(Lambda_m) > 1.0e-16:
        N = D_m * (G - 1.0) / Lambda_m
    else:
        N = D_m * t
    return G, N


def build_state_data(
    plot_data: pd.DataFrame,
    channel: dict,
    p: Params = P,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tau_plot = plot_data["tau"].to_numpy()
    G_plot = plot_data["G"].to_numpy()
    N_plot = plot_data["N"].to_numpy()

    gaussian = gaussian_state_curves(G_plot, N_plot, p)
    fock_curve = propagate_fock_time_dependent(
        channel["tau"],
        channel["D"],
        channel["B"],
        tau_plot,
        m=p.fock_m,
        cutoff=p.fock_cutoff,
    )

    initial = {
        "coherent": p.alpha0_abs2,
        "squeezed": p.sinh2_r,
        "fock": float(p.fock_m),
        "thermal": 0.0,
    }

    panel_b = pd.DataFrame({
        "tau": tau_plot,
        "nu_tau": p.nu * tau_plot,
        "dW_coh": gaussian["coherent"] - initial["coherent"],
        "dW_sqz": gaussian["squeezed"] - initial["squeezed"],
        "dW_fock": fock_curve - initial["fock"],
        "dW_thm": gaussian["thermal"] - initial["thermal"],
    })

    markers = density_matrix_markers(
        channel["tau"],
        channel["D"],
        channel["B"],
        tau_plot,
        p,
    )
    for column in ("dW_num_coh", "dW_num_sqz", "dW_num_fock", "dW_num_thm"):
        panel_b[column] = np.nan

    idx = markers["indices"]
    panel_b.loc[idx, "dW_num_coh"] = markers["coherent"]
    panel_b.loc[idx, "dW_num_sqz"] = markers["squeezed"]
    panel_b.loc[idx, "dW_num_fock"] = markers["fock"]
    panel_b.loc[idx, "dW_num_thm"] = markers["thermal"]

    tau_star = channel["tau_star"]
    G_star = float(np.interp(tau_star, channel["tau"], channel["G"]))
    N_star = float(np.interp(tau_star, channel["tau"], channel["N"]))

    finite_gaussian = gaussian_state_curves(
        np.array([G_star]),
        np.array([N_star]),
        p,
    )
    finite_fock = propagate_fock_time_dependent(
        channel["tau"],
        channel["D"],
        channel["B"],
        np.array([tau_star]),
        m=p.fock_m,
        cutoff=p.fock_cutoff,
    )

    G_m, N_m = markovian_gain_noise(
        np.array([tau_star]),
        D_m=channel["D_m"],
        Lambda_m=channel["Lambda_m"],
    )
    markov_gaussian = gaussian_state_curves(G_m, N_m, p)
    markov_fock = propagate_fock_markovian(
        np.array([tau_star]),
        D_m=channel["D_m"],
        B_m=channel["B_m"],
        m=p.fock_m,
        cutoff=p.fock_cutoff,
    )

    panel_a = pd.DataFrame({
        "state": ["Coherent", "Squeezed", "Fock", "Thermal"],
        "dW_finite_time": [
            finite_gaussian["coherent"][0] - initial["coherent"],
            finite_gaussian["squeezed"][0] - initial["squeezed"],
            finite_fock[0] - initial["fock"],
            finite_gaussian["thermal"][0] - initial["thermal"],
        ],
        "dW_markovian": [
            markov_gaussian["coherent"][0] - initial["coherent"],
            markov_gaussian["squeezed"][0] - initial["squeezed"],
            markov_fock[0] - initial["fock"],
            markov_gaussian["thermal"][0] - initial["thermal"],
        ],
    })

    return panel_a, panel_b


def style_axis(ax, *, logx: bool = False):
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.0,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.2,
    )
    if not logx:
        ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.16, linewidth=0.45)


def draw_panel_a(ax, panel_a: pd.DataFrame, tau_star: float):
    x = np.arange(len(panel_a))
    width = 0.36

    ax.bar(
        x - width / 2,
        panel_a["dW_finite_time"],
        width,
        label="Finite time",
    )
    ax.bar(
        x + width / 2,
        panel_a["dW_markovian"],
        width,
        label="Markovian",
    )

    ax.axhline(0.0, linewidth=0.9, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(panel_a["state"])
    ax.set_ylabel(r"$\Delta\mathcal{W}_P/(\hbar\nu)$")
    ax.set_title(rf"$\nu\tau_c^\ast={P.nu * tau_star:.1f}$")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(frameon=False, ncol=2, loc="best")
    style_axis(ax, logx=False)
    ax.text(
        -0.13,
        1.04,
        "(a)",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )


def draw_panel_b(ax, panel_b: pd.DataFrame, tau_star: float):
    x = panel_b["nu_tau"].to_numpy()

    line_coh = ax.plot(x, panel_b["dW_coh"], label="Coherent")[0]
    line_sqz = ax.plot(x, panel_b["dW_sqz"], linestyle="-.", label="Squeezed")[0]
    line_fock = ax.plot(x, panel_b["dW_fock"], linestyle="--", label="Fock")[0]
    line_thm = ax.plot(x, panel_b["dW_thm"], linestyle=":", label="Thermal")[0]

    numerical_specs = [
        ("dW_num_coh", line_coh, "o"),
        ("dW_num_sqz", line_sqz, "s"),
        ("dW_num_fock", line_fock, "^"),
        ("dW_num_thm", line_thm, "D"),
    ]
    for column, line, marker in numerical_specs:
        mask = np.isfinite(panel_b[column].to_numpy())
        ax.plot(
            x[mask],
            panel_b.loc[mask, column].to_numpy(),
            linestyle="none",
            marker=marker,
            markersize=3.1,
            markerfacecolor=line.get_color(),
            markeredgecolor=line.get_color(),
            markeredgewidth=0.55,
            label="_nolegend_",
            zorder=3,
        )

    ax.axhline(0.0, linewidth=0.9, alpha=0.65)
    ax.axvline(
        P.nu * tau_star,
        linestyle=":",
        linewidth=1.1,
        alpha=0.65,
    )
    ax.set_xscale("log")
    ax.set_xlim(P.tau_min_plot, P.tau_max)
    ax.set_xlabel(r"$\nu\tau_c$")
    ax.set_ylabel(r"$\Delta\mathcal{W}_P/(\hbar\nu)$")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(frameon=False, ncol=2, loc="best")
    style_axis(ax, logx=True)
    ax.text(
        -0.13,
        1.04,
        "(b)",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )


def make_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, tau_star: float):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 5.85),
        constrained_layout=True,
    )
    draw_panel_a(axes[0], panel_a, tau_star)
    draw_panel_b(axes[1], panel_b, tau_star)
    fig.savefig(
        OUTPUT_DIR / "appendixB_state_resolved_ergotropy.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def main():
    plot_data, channel = build_channel(P)
    panel_a, panel_b = build_state_data(plot_data, channel, P)
    make_figure(panel_a, panel_b, channel["tau_star"])


if __name__ == "__main__":
    main()
