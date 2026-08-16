"""Reproduce the three engine panels used in the manuscript.

The script evaluates the finite-time reservoir rates, constructs the analytical
engine curves, propagates the retained TLS--piston master equation at the marker
times, and saves the three figure panels as PDF files.

Units: hbar = k_B = nu = 1.
Dependencies: numpy, scipy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh, expm
from scipy.sparse import csc_matrix, identity, kron
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln


# =============================================================================
# Parameters
# =============================================================================


@dataclass(frozen=True)
class Parameters:
    """Dimensionless parameters for the engine benchmark."""

    nu: float = 1.0
    omega0: float = 3.0
    eta: float = 0.095
    alpha0: float = 1.0

    beta_h_times_omega0: float = 0.35
    beta_c_times_omega_minus: float = 1.50

    linewidth_h: float = 1.60e-3
    linewidth_c: float = 1.10e-3
    delta_h: float = 5.00e-3
    delta_c: float = 3.80e-3
    window_h: float = 8.50e-3
    window_c: float = 7.50e-3
    spectral_amplitude: float = 1.00e-5

    tau_min: float = 20.0
    tau_max: float = 4.00e4
    n_tau_points: int = 950
    n_frequency_grid: int = 1001

    piston_cutoff: int = 24
    marker_taus: tuple[float, ...] = (
        60.0,
        160.0,
        350.0,
        650.0,
        2000.0,
        8000.0,
        30000.0,
    )
    marker_time_steps: int = 150

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


P = Parameters()
OUTPUT_DIR = Path(".")


# =============================================================================
# Plot style
# =============================================================================


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 15,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.05,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "hot_down": "#1f77b4",
    "hot_up": "#17becf",
    "cold_down": "#d62728",
    "cold_up": "#9467bd",
    "gain": "#003f5c",
    "ergotropy": "#2f4b7c",
    "full": "#d62728",
    "reference": "#7f7f7f",
}


# =============================================================================
# Reservoir spectra and finite-time rates
# =============================================================================


def positive_filtered_lorentzian(
    frequency: np.ndarray,
    *,
    center: float,
    target: float,
    linewidth: float,
    half_width: float,
    p: Parameters = P,
) -> np.ndarray:
    """Positive-frequency filtered Lorentzian response."""

    frequency = np.asarray(frequency, dtype=float)
    window = (np.abs(frequency - target) <= half_width).astype(float)
    lorentzian = (
        p.spectral_amplitude
        * linewidth**2
        / ((frequency - center) ** 2 + linewidth**2)
    )
    return lorentzian * window


def two_sided_spectrum(
    frequency: np.ndarray,
    bath: str,
    p: Parameters = P,
) -> np.ndarray:
    """Two-sided thermal response spectrum."""

    frequency = np.asarray(frequency, dtype=float)
    abs_frequency = np.abs(frequency)

    if bath == "h":
        target = p.omega0
        center = p.omega0 + p.delta_h
        linewidth = p.linewidth_h
        half_width = p.window_h
        beta = p.beta_h
    elif bath == "c":
        target = p.omega_minus
        center = p.omega_minus - p.delta_c
        linewidth = p.linewidth_c
        half_width = p.window_c
        beta = p.beta_c
    else:
        raise ValueError("bath must be 'h' or 'c'")

    positive = positive_filtered_lorentzian(
        abs_frequency,
        center=center,
        target=target,
        linewidth=linewidth,
        half_width=half_width,
        p=p,
    )
    return np.where(
        frequency >= 0.0,
        positive,
        np.exp(-beta * abs_frequency) * positive,
    )


def frequency_grid(bath: str, p: Parameters = P) -> np.ndarray:
    """Positive-frequency quadrature grid over the selected spectral window."""

    if bath == "h":
        target, half_width = p.omega0, p.window_h
    elif bath == "c":
        target, half_width = p.omega_minus, p.window_c
    else:
        raise ValueError("bath must be 'h' or 'c'")

    return np.linspace(
        target - half_width,
        target + half_width,
        p.n_frequency_grid,
    )


def finite_time_kernel(
    transition_frequency: float,
    frequencies: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate sin[(omega-xi)t]/(omega-xi) on a time-frequency grid."""

    frequencies = np.asarray(frequencies, dtype=float)
    times = np.asarray(times, dtype=float)
    difference = transition_frequency - frequencies
    phase = times[:, None] * difference[None, :]

    near_zero = np.abs(difference) < 1.0e-13
    safe_difference = difference.copy()
    safe_difference[near_zero] = 1.0

    kernel = np.sin(phase) / safe_difference[None, :]
    if np.any(near_zero):
        kernel[:, near_zero] = times[:, None]

    return kernel


def finite_time_rate(
    transition_frequency: float,
    times: np.ndarray,
    bath: str,
    p: Parameters = P,
) -> np.ndarray:
    """Instantaneous finite-time rate gamma_j(omega,t)."""

    positive_frequencies = frequency_grid(bath, p)
    positive_spectrum = two_sided_spectrum(positive_frequencies, bath, p)
    negative_spectrum = two_sided_spectrum(-positive_frequencies, bath, p)

    positive_part = (
        finite_time_kernel(transition_frequency, positive_frequencies, times)
        * positive_spectrum[None, :]
    )
    negative_part = (
        finite_time_kernel(transition_frequency, -positive_frequencies, times)
        * negative_spectrum[None, :]
    )

    return 2.0 * np.trapezoid(
        positive_part + negative_part,
        positive_frequencies,
        axis=1,
    )


def finite_time_rate_scalar(
    transition_frequency: float,
    time: float,
    bath: str,
    p: Parameters = P,
) -> float:
    """Scalar finite-time rate."""

    return float(
        finite_time_rate(
            transition_frequency,
            np.array([time], dtype=float),
            bath,
            p,
        )[0]
    )


def markov_rate(
    transition_frequency: float,
    bath: str,
    p: Parameters = P,
) -> float:
    """Long-time rate 2*pi*G_j(omega)."""

    spectrum = two_sided_spectrum(
        np.array([transition_frequency]),
        bath,
        p,
    )[0]
    return float(2.0 * np.pi * spectrum)


def cumulative_integral_from_zero(
    values: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Cumulative trapezoidal integral from zero to each time."""

    full_times = np.concatenate(([0.0], np.asarray(times, dtype=float)))
    full_values = np.concatenate(([0.0], np.asarray(values, dtype=float)))
    integral = np.zeros_like(full_times)
    integral[1:] = np.cumsum(
        0.5
        * (full_values[1:] + full_values[:-1])
        * np.diff(full_times)
    )
    return integral[1:]


# =============================================================================
# Analytical engine curves
# =============================================================================


@dataclass(frozen=True)
class EngineCurves:
    tau: np.ndarray
    channel_hot_down: np.ndarray
    channel_hot_up: np.ndarray
    channel_cold_down: np.ndarray
    channel_cold_up: np.ndarray
    gain_factor: np.ndarray
    ergotropy_ratio: np.ndarray
    markov_gain: float



def build_engine_curves(p: Parameters = P) -> EngineCurves:
    """Construct the analytical curves shown in the three panels."""

    tau = np.geomspace(p.tau_min, p.tau_max, p.n_tau_points)

    hot_down = finite_time_rate(+p.omega0, tau, "h", p)
    hot_up = finite_time_rate(-p.omega0, tau, "h", p)
    cold_down_raw = finite_time_rate(+p.omega_minus, tau, "c", p)
    cold_up_raw = finite_time_rate(-p.omega_minus, tau, "c", p)

    cold_down = p.sideband_prefactor * cold_down_raw
    cold_up = p.sideband_prefactor * cold_up_raw

    hot_down_m = markov_rate(+p.omega0, "h", p)
    hot_up_m = markov_rate(-p.omega0, "h", p)
    cold_down_raw_m = markov_rate(+p.omega_minus, "c", p)
    cold_up_raw_m = markov_rate(-p.omega_minus, "c", p)

    cold_down_m = p.sideband_prefactor * cold_down_raw_m
    cold_up_m = p.sideband_prefactor * cold_up_raw_m

    excited_m = hot_up_m / (hot_down_m + hot_up_m)
    ground_m = 1.0 - excited_m
    markov_gain = cold_down_m * excited_m - cold_up_m * ground_m

    hot_total = hot_down + hot_up
    excited = np.where(
        hot_total > 0.0,
        hot_up / hot_total,
        excited_m,
    )
    ground = 1.0 - excited

    gain = cold_down * excited - cold_up * ground
    accumulated_gain = cumulative_integral_from_zero(gain, tau)
    gain_factor = (accumulated_gain / tau) / markov_gain

    markov_ergotropy_change = np.expm1(markov_gain * tau)
    finite_time_ergotropy_change = np.expm1(accumulated_gain)
    ergotropy_ratio = np.where(
        np.abs(markov_ergotropy_change) > 1.0e-300,
        finite_time_ergotropy_change / markov_ergotropy_change,
        gain_factor,
    )

    channel_hot_down = (
        cumulative_integral_from_zero(hot_down, tau) / tau
    ) / hot_down_m
    channel_hot_up = (
        cumulative_integral_from_zero(hot_up, tau) / tau
    ) / hot_up_m
    channel_cold_down = (
        cumulative_integral_from_zero(cold_down_raw, tau) / tau
    ) / cold_down_raw_m
    channel_cold_up = (
        cumulative_integral_from_zero(cold_up_raw, tau) / tau
    ) / cold_up_raw_m

    return EngineCurves(
        tau=tau,
        channel_hot_down=channel_hot_down,
        channel_hot_up=channel_hot_up,
        channel_cold_down=channel_cold_down,
        channel_cold_up=channel_cold_up,
        gain_factor=gain_factor,
        ergotropy_ratio=ergotropy_ratio,
        markov_gain=markov_gain,
    )


# =============================================================================
# Retained TLS--piston propagation for numerical markers
# =============================================================================


def destroy_sparse(dimension: int) -> csc_matrix:
    """Sparse harmonic-oscillator annihilation operator."""

    rows = np.arange(dimension - 1)
    cols = np.arange(1, dimension)
    values = np.sqrt(np.arange(1, dimension, dtype=float))
    return csc_matrix(
        (values, (rows, cols)),
        shape=(dimension, dimension),
        dtype=complex,
    )


def destroy_dense(dimension: int) -> np.ndarray:
    """Dense harmonic-oscillator annihilation operator."""

    operator = np.zeros((dimension, dimension), dtype=complex)
    indices = np.arange(1, dimension)
    operator[indices - 1, indices] = np.sqrt(indices)
    return operator


def coherent_state(alpha: complex, dimension: int) -> np.ndarray:
    """Truncated coherent-state ket."""

    number = np.arange(dimension)
    if abs(alpha) == 0.0:
        state = np.zeros(dimension, dtype=complex)
        state[0] = 1.0
        return state

    coefficients = np.exp(
        -abs(alpha) ** 2 / 2
        + number * np.log(alpha)
        - 0.5 * gammaln(number + 1)
    ).astype(complex)
    return coefficients / np.linalg.norm(coefficients)


def vectorize(matrix: np.ndarray) -> np.ndarray:
    """Column-major vectorization."""

    return np.asarray(matrix, dtype=complex).reshape((-1,), order="F")


def devectorize(vector: np.ndarray, dimension: int) -> np.ndarray:
    """Inverse column-major vectorization."""

    return np.asarray(vector, dtype=complex).reshape(
        (dimension, dimension),
        order="F",
    )


def dissipator_superoperator(operator: csc_matrix) -> csc_matrix:
    """Liouville-space Lindblad dissipator."""

    operator = csc_matrix(operator)
    dimension = operator.shape[0]
    unit = identity(dimension, format="csc", dtype=complex)
    adjoint = operator.getH()
    number = adjoint @ operator

    return (
        kron(operator.conjugate(), operator, format="csc")
        - 0.5 * kron(unit, number, format="csc")
        - 0.5 * kron(number.T, unit, format="csc")
    )


@dataclass(frozen=True)
class Superoperators:
    hot_down: csc_matrix
    hot_up: csc_matrix
    cold_down: csc_matrix
    cold_up: csc_matrix



def build_superoperators(p: Parameters = P) -> Superoperators:
    """Retained-channel dissipators in the joint TLS--piston space."""

    dimension = p.piston_cutoff
    piston_identity = identity(dimension, format="csc", dtype=complex)
    annihilation = destroy_sparse(dimension)
    creation = annihilation.getH()

    sigma_minus = csc_matrix(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    )
    sigma_plus = csc_matrix(
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    )

    return Superoperators(
        hot_down=dissipator_superoperator(
            kron(sigma_minus, piston_identity, format="csc")
        ),
        hot_up=dissipator_superoperator(
            kron(sigma_plus, piston_identity, format="csc")
        ),
        cold_down=dissipator_superoperator(
            kron(sigma_minus, creation, format="csc")
        ),
        cold_up=dissipator_superoperator(
            kron(sigma_plus, annihilation, format="csc")
        ),
    )


def initial_joint_state(
    excited_population: float,
    p: Parameters = P,
) -> np.ndarray:
    """Initial TLS state times a coherent piston state."""

    rho_tls = np.array(
        [
            [excited_population, 0.0],
            [0.0, 1.0 - excited_population],
        ],
        dtype=complex,
    )
    piston_ket = coherent_state(p.alpha0, p.piston_cutoff)
    rho_piston = np.outer(piston_ket, piston_ket.conj())
    return np.kron(rho_tls, rho_piston)


def reduced_piston_state(
    rho_joint: np.ndarray,
    piston_cutoff: int,
) -> np.ndarray:
    """Trace out the TLS."""

    reshaped = rho_joint.reshape(2, piston_cutoff, 2, piston_cutoff)
    return np.einsum("anam->nm", reshaped)


def ergotropy(rho_piston: np.ndarray, nu: float) -> float:
    """Ergotropy of a truncated piston state."""

    hermitian = 0.5 * (rho_piston + rho_piston.conj().T)
    dimension = rho_piston.shape[0]
    energies = nu * np.arange(dimension)

    mean_energy = float(
        np.real(np.sum(np.diag(hermitian) * energies))
    )
    eigenvalues = np.sort(np.real(eigvalsh(hermitian)))[::-1]
    passive_energy = float(np.sum(eigenvalues * energies))
    return mean_energy - passive_energy


def displacement_operator(
    alpha: complex,
    dimension: int,
) -> np.ndarray:
    """Truncated oscillator displacement operator."""

    annihilation = destroy_dense(dimension)
    creation = annihilation.conj().T
    return expm(alpha * creation - np.conj(alpha) * annihilation)


def bare_piston_state(
    rho_dressed_joint: np.ndarray,
    p: Parameters = P,
) -> np.ndarray:
    """Undo the polaron transformation and trace out the TLS."""

    dimension = p.piston_cutoff
    displacement_plus = displacement_operator(+p.eta, dimension)
    displacement_minus = displacement_operator(-p.eta, dimension)

    unitary = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    unitary_dagger = np.zeros_like(unitary)

    unitary[:dimension, :dimension] = displacement_plus
    unitary[dimension:, dimension:] = displacement_minus
    unitary_dagger[:dimension, :dimension] = displacement_minus
    unitary_dagger[dimension:, dimension:] = displacement_plus

    rho_bare_joint = unitary_dagger @ rho_dressed_joint @ unitary
    return reduced_piston_state(rho_bare_joint, dimension)


def retained_rates(
    time: float,
    p: Parameters = P,
) -> tuple[float, float, float, float]:
    """Instantaneous retained hot-carrier and cold-sideband rates."""

    return (
        finite_time_rate_scalar(+p.omega0, time, "h", p),
        finite_time_rate_scalar(-p.omega0, time, "h", p),
        p.sideband_prefactor
        * finite_time_rate_scalar(+p.omega_minus, time, "c", p),
        p.sideband_prefactor
        * finite_time_rate_scalar(-p.omega_minus, time, "c", p),
    )


def markovian_retained_rates(
    p: Parameters = P,
) -> tuple[float, float, float, float]:
    """Long-time retained-channel rates."""

    return (
        markov_rate(+p.omega0, "h", p),
        markov_rate(-p.omega0, "h", p),
        p.sideband_prefactor * markov_rate(+p.omega_minus, "c", p),
        p.sideband_prefactor * markov_rate(-p.omega_minus, "c", p),
    )


def liouvillian(
    rates: tuple[float, float, float, float],
    operators: Superoperators,
) -> csc_matrix:
    """Retained-channel Liouvillian for a given set of rates."""

    hot_down, hot_up, cold_down, cold_up = rates
    return (
        hot_down * operators.hot_down
        + hot_up * operators.hot_up
        + cold_down * operators.cold_down
        + cold_up * operators.cold_up
    )


def marker_time_grid(
    final_time: float,
    steps: int,
) -> np.ndarray:
    """Time grid used for the time-ordered marker propagation."""

    first_time = min(1.0e-3, final_time * 1.0e-7)
    return np.concatenate(
        ([0.0], np.geomspace(first_time, final_time, steps))
    )


def propagate_finite_time(
    final_time: float,
    rho0: np.ndarray,
    operators: Superoperators,
    p: Parameters = P,
) -> np.ndarray:
    """Propagate the retained master equation with instantaneous rates."""

    dimension = 2 * p.piston_cutoff
    state_vector = vectorize(rho0)
    grid = marker_time_grid(final_time, p.marker_time_steps)

    for time0, time1 in zip(grid[:-1], grid[1:]):
        step = time1 - time0
        midpoint = 0.5 * (time0 + time1)
        generator = liouvillian(retained_rates(midpoint, p), operators)
        state_vector = expm_multiply(generator * step, state_vector)

    rho = devectorize(state_vector, dimension)
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)


def propagate_markovian(
    final_time: float,
    rho0: np.ndarray,
    operators: Superoperators,
    p: Parameters = P,
) -> np.ndarray:
    """Propagate the retained master equation with long-time rates."""

    dimension = 2 * p.piston_cutoff
    generator = liouvillian(markovian_retained_rates(p), operators)
    state_vector = expm_multiply(
        generator * final_time,
        vectorize(rho0),
    )
    rho = devectorize(state_vector, dimension)
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho)


@dataclass(frozen=True)
class MarkerData:
    tau: np.ndarray
    gain_factor: np.ndarray
    bare_ergotropy_ratio: np.ndarray



def compute_markers(
    p: Parameters = P,
) -> MarkerData:
    """Compute the numerical markers used in panels (b) and (c)."""

    operators = build_superoperators(p)

    hot_down_m, hot_up_m, _, _ = markovian_retained_rates(p)
    excited_population = hot_up_m / (hot_down_m + hot_up_m)
    rho0 = initial_joint_state(excited_population, p)

    piston_annihilation = destroy_dense(p.piston_cutoff)
    rho_piston0 = reduced_piston_state(rho0, p.piston_cutoff)
    rho_bare0 = bare_piston_state(rho0, p)
    bare_ergotropy0 = ergotropy(rho_bare0, p.nu)
    alpha0_squared = abs(p.alpha0) ** 2

    gain_markers = []
    ergotropy_markers = []

    for tau in p.marker_taus:
        rho_finite = propagate_finite_time(tau, rho0, operators, p)
        rho_markov = propagate_markovian(tau, rho0, operators, p)

        piston_finite = reduced_piston_state(
            rho_finite,
            p.piston_cutoff,
        )
        piston_markov = reduced_piston_state(
            rho_markov,
            p.piston_cutoff,
        )

        alpha_finite = np.trace(piston_annihilation @ piston_finite)
        alpha_markov = np.trace(piston_annihilation @ piston_markov)

        accumulated_gain_finite = np.log(
            max(abs(alpha_finite) ** 2, 1.0e-300)
            / alpha0_squared
        )
        accumulated_gain_markov = np.log(
            max(abs(alpha_markov) ** 2, 1.0e-300)
            / alpha0_squared
        )
        gain_markers.append(
            accumulated_gain_finite / accumulated_gain_markov
        )

        bare_finite = bare_piston_state(rho_finite, p)
        bare_markov = bare_piston_state(rho_markov, p)

        delta_ergotropy_finite = (
            ergotropy(bare_finite, p.nu) - bare_ergotropy0
        )
        delta_ergotropy_markov = (
            ergotropy(bare_markov, p.nu) - bare_ergotropy0
        )
        ergotropy_markers.append(
            delta_ergotropy_finite / delta_ergotropy_markov
        )

    return MarkerData(
        tau=np.asarray(p.marker_taus, dtype=float),
        gain_factor=np.asarray(gain_markers, dtype=float),
        bare_ergotropy_ratio=np.asarray(ergotropy_markers, dtype=float),
    )


# =============================================================================
# Figure generation
# =============================================================================


def style_axis(axis: plt.Axes) -> None:
    """Apply the common axis style."""

    axis.set_xscale("log")
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=4.5,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.5,
    )
    axis.minorticks_on()
    axis.grid(True, which="major", alpha=0.20, linewidth=0.55)
    axis.grid(True, which="minor", alpha=0.09, linewidth=0.40)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    """Place a panel label at the upper-left of the axes."""

    axis.text(
        0.035,
        0.965,
        label,
        transform=axis.transAxes,
        fontsize=13.5,
        fontweight="normal",
        va="top",
        ha="left",
        zorder=10,
    )


def save_panel(figure: plt.Figure, filename: str) -> None:
    """Save one manuscript panel as a PDF."""

    figure.savefig(
        OUTPUT_DIR / filename,
        bbox_inches="tight",
        pad_inches=0.045,
    )
    plt.close(figure)


def plot_channel_factors(
    curves: EngineCurves,
    p: Parameters = P,
) -> None:
    """Generate panel (a)."""

    figure, axis = plt.subplots(
        figsize=(4.55, 3.35),
        constrained_layout=True,
    )

    axis.plot(
        p.nu * curves.tau,
        curves.channel_hot_down,
        color=COLORS["hot_down"],
        linestyle="-",
        label=r"$\overline{\mathcal{A}}_{h}^{\downarrow}$",
    )
    axis.plot(
        p.nu * curves.tau,
        curves.channel_hot_up,
        color=COLORS["hot_up"],
        linestyle="--",
        label=r"$\overline{\mathcal{A}}_{h}^{\uparrow}$",
    )
    axis.plot(
        p.nu * curves.tau,
        curves.channel_cold_down,
        color=COLORS["cold_down"],
        linestyle="-.",
        label=r"$\overline{\mathcal{A}}_{c}^{\downarrow}$",
    )
    axis.plot(
        p.nu * curves.tau,
        curves.channel_cold_up,
        color=COLORS["cold_up"],
        linestyle=":",
        label=r"$\overline{\mathcal{A}}_{c}^{\uparrow}$",
    )
    axis.axhline(
        1.0,
        color=COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.05,
    )

    axis.set_xlabel(r"$\nu\tau_c$", fontsize=15.5)
    axis.set_ylabel("Averaged channel factor")
    axis.set_xlim(p.tau_min, p.tau_max)
    axis.set_ylim(0.45, 3.0)
    add_panel_label(axis, "(a)")
    style_axis(axis)
    axis.legend(
        frameon=False,
        loc="upper right",
        handlelength=2.5,
        borderaxespad=0.25,
        fontsize=10.5,
    )

    save_panel(figure, "engine_panel_a_channel_factors.pdf")


def plot_gain_factor(
    curves: EngineCurves,
    markers: MarkerData,
    p: Parameters = P,
) -> None:
    """Generate panel (b)."""

    peak_index = int(np.nanargmax(curves.gain_factor))
    peak_tau = curves.tau[peak_index]
    peak_gain = curves.gain_factor[peak_index]

    figure, axis = plt.subplots(
        figsize=(4.55, 3.35),
        constrained_layout=True,
    )

    axis.plot(
        p.nu * curves.tau,
        curves.gain_factor,
        color=COLORS["gain"],
        label="Closure",
    )
    axis.plot(
        p.nu * markers.tau,
        markers.gain_factor,
        "o",
        color=COLORS["full"],
        markersize=5.2,
        label="Full TLS+piston",
    )
    axis.axhline(
        1.0,
        color=COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.05,
    )
    axis.scatter(
        [p.nu * peak_tau],
        [peak_gain],
        s=26,
        color=COLORS["gain"],
        zorder=5,
    )
    axis.annotate(
        rf"max $\overline{{\mathcal{{A}}}}_\Lambda={peak_gain:.2f}$",
        xy=(p.nu * peak_tau, peak_gain),
        xytext=(p.nu * peak_tau * 1.85, peak_gain - 0.52),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.85,
            "color": "0.25",
        },
        fontsize=10.2,
    )
    axis.text(28, 0.55, "Zeno", fontsize=8.8, color="0.25")
    axis.text(
        p.nu * peak_tau * 0.52,
        peak_gain + 0.13,
        "anti-Zeno",
        fontsize=8.8,
        color="0.25",
    )
    axis.text(
        p.nu * p.tau_max / 4.0,
        1.15,
        "Markovian",
        fontsize=8.8,
        color="0.25",
    )

    axis.set_xlabel(r"$\nu\tau_c$", fontsize=15.5)
    axis.set_ylabel(
        r"$\overline{\mathcal{A}}_\Lambda(\tau_c)$",
        fontsize=15.5,
    )
    axis.set_xlim(p.tau_min, p.tau_max)
    axis.set_ylim(0.45, 3.0)
    add_panel_label(axis, "(b)")
    style_axis(axis)
    axis.legend(
        frameon=False,
        loc="upper right",
        borderaxespad=0.25,
        fontsize=10.5,
    )

    save_panel(figure, "engine_panel_b_gain_factor.pdf")


def plot_ergotropy_ratio(
    curves: EngineCurves,
    markers: MarkerData,
    p: Parameters = P,
) -> None:
    """Generate panel (c)."""

    peak_index = int(np.nanargmax(curves.ergotropy_ratio))
    peak_tau = curves.tau[peak_index]
    peak_ratio = curves.ergotropy_ratio[peak_index]

    figure, axis = plt.subplots(
        figsize=(4.55, 3.35),
        constrained_layout=True,
    )

    axis.plot(
        p.nu * curves.tau,
        curves.ergotropy_ratio,
        color=COLORS["ergotropy"],
        label="Closure",
    )
    axis.plot(
        p.nu * markers.tau,
        markers.bare_ergotropy_ratio,
        "o",
        color=COLORS["full"],
        markersize=5.2,
        label="Full bare",
    )
    axis.axhline(
        1.0,
        color=COLORS["reference"],
        linestyle=(0, (4, 3)),
        linewidth=1.05,
    )
    axis.scatter(
        [p.nu * peak_tau],
        [peak_ratio],
        s=26,
        color=COLORS["ergotropy"],
        zorder=5,
    )
    axis.annotate(
        rf"max $R={peak_ratio:.2f}$",
        xy=(p.nu * peak_tau, peak_ratio),
        xytext=(p.nu * peak_tau * 1.85, peak_ratio - 0.52),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.85,
            "color": "0.25",
        },
        fontsize=10.2,
    )

    axis.set_xlabel(r"$\nu\tau_c$", fontsize=15.5)
    axis.set_ylabel(
        r"$R_{\Delta\mathcal{W}_P}^{\rm coh}(\tau_c)$",
        fontsize=15.5,
    )
    axis.set_xlim(p.tau_min, p.tau_max)
    axis.set_ylim(0.45, 3.0)
    add_panel_label(axis, "(c)")
    style_axis(axis)
    axis.legend(
        frameon=False,
        loc="upper right",
        borderaxespad=0.25,
        fontsize=10.5,
    )

    save_panel(figure, "engine_panel_c_ergotropy_ratio.pdf")


def main() -> None:
    curves = build_engine_curves(P)
    markers = compute_markers(P)

    plot_channel_factors(curves, P)
    plot_gain_factor(curves, markers, P)
    plot_ergotropy_ratio(curves, markers, P)


if __name__ == "__main__":
    main()
