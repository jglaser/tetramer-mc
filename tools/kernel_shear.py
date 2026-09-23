"""Frozen RBF coupling charts and an exact extended Euclidean involution.

This standalone prototype does not change any protein sampler or guide format.
The RBF expansion is a *vector displacement*, not a probability density. Its
coefficients need not be positive. The additive triangular shear has determinant
one, so pushing a standard Gaussian through it produces a normalized density.

Chart convention: x = mean + L S(z). All densities here are with respect to
Euclidean Lebesgue measure. A pose implementation still needs its rotational
coordinate/Haar Jacobian, seam/support handling, and physical acceptance factor.
Charts must stay frozen during a move. The pair law must be fixed and symmetric;
otherwise its forward/reverse probability ratio is an additional obligation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np


def _vector(value, dimension, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of length {dimension}')
    return result


def _matrix(value, rows, columns, name):
    result = np.asarray(value, dtype=float)
    if rows == 0 and result.size == 0:
        result = result.reshape(0, columns)
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite {rows} by {columns} matrix')
    return result


def _indices(value, name):
    result = tuple(value)
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in result):
        raise ValueError(f'{name} must contain integer indices')
    return tuple(int(i) for i in result)


def _tuples(matrix):
    return tuple(tuple(float(v) for v in row) for row in matrix)


@dataclass(frozen=True, slots=True)
class KernelShear:
    """S(z)_A=z_A, S(z)_B=z_B+sum_j a_j exp(-|z_A-c_j|²/(2h²)).

    A and B are ordered, disjoint coordinate lists covering 0..dimension-1.
    Empty A gives a constant displacement; empty B or no centers gives identity.
    Inputs are copied into immutable tuples, including nested coefficient rows.
    """
    dimension: int
    conditioning: tuple[int, ...]
    shifted: tuple[int, ...]
    centers: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    bandwidth: float = 1.0

    def __post_init__(self):
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, Integral) or self.dimension < 1:
            raise ValueError('dimension must be a positive integer')
        dimension = int(self.dimension)
        a, b = _indices(self.conditioning, 'conditioning'), _indices(self.shifted, 'shifted')
        if sorted(a + b) != list(range(dimension)):
            raise ValueError('conditioning and shifted must be a disjoint complete partition')
        bandwidth = float(self.bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('bandwidth must be finite and positive')
        centers = _matrix(self.centers, len(self.centers), len(a), 'centers')
        coefficients = _matrix(self.coefficients, len(centers), len(b), 'coefficients')
        for name, value in [('dimension', dimension), ('conditioning', a), ('shifted', b),
                            ('centers', _tuples(centers)), ('coefficients', _tuples(coefficients)),
                            ('bandwidth', bandwidth)]:
            object.__setattr__(self, name, value)

    def displacement(self, conditioning_values):
        a = _vector(conditioning_values, len(self.conditioning), 'conditioning values')
        if not self.centers or not self.shifted:
            return np.zeros(len(self.shifted))
        centers = np.asarray(self.centers).reshape(len(self.centers), len(self.conditioning))
        # Infinite distance is a representable zero RBF, even for finite inputs
        # whose subtraction or division overflows floating-point range.
        with np.errstate(over='ignore', under='ignore'):
            distance = (a - centers) / self.bandwidth
            kernel = np.exp(-0.5 * np.sum(distance * distance, axis=1))
            result = kernel @ np.asarray(self.coefficients)
        if not np.isfinite(result).all():
            raise ValueError('kernel displacement is not representable')
        return result

    def _apply(self, value, sign):
        result = _vector(value, self.dimension, 'shear coordinates').copy()
        with np.errstate(over='ignore', invalid='ignore'):
            result[list(self.shifted)] += sign * self.displacement(result[list(self.conditioning)])
        if not np.isfinite(result).all():
            raise ValueError('sheared coordinates are not representable')
        return result

    def forward(self, z):
        return self._apply(z, 1.0)

    def encode(self, z):
        """Alias for forward shear; chart encode below instead means x -> z."""
        return self.forward(z)

    def inverse(self, w):
        return self._apply(w, -1.0)


@dataclass(frozen=True, slots=True)
class WarpedGaussianChart:
    """Normalized pushforward of N(0,I) under x=mean+L*S(z)."""
    mean: tuple[float, ...]
    lower: tuple[tuple[float, ...], ...]
    shear: KernelShear

    def __post_init__(self):
        if not isinstance(self.shear, KernelShear):
            raise ValueError('shear must be a frozen KernelShear')
        d = self.shear.dimension
        mean = _vector(self.mean, d, 'mean')
        lower = _matrix(self.lower, d, d, 'lower')
        if np.any(np.triu(lower, 1) != 0) or np.any(np.diag(lower) <= 0):
            raise ValueError('lower must be triangular with strictly positive diagonal')
        object.__setattr__(self, 'mean', tuple(float(v) for v in mean))
        object.__setattr__(self, 'lower', _tuples(lower))

    @property
    def dimension(self):
        return self.shear.dimension

    @property
    def log_abs_determinant(self):
        return sum(math.log(self.lower[i][i]) for i in range(self.dimension))

    def decode(self, z):
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.asarray(self.mean) + np.asarray(self.lower) @ self.shear.forward(z)
        if not np.isfinite(result).all():
            raise ValueError('decoded coordinates are not representable')
        return result

    def encode(self, x):
        value = _vector(x, self.dimension, 'chart coordinates')
        with np.errstate(over='ignore', invalid='ignore'):
            centered = value - self.mean
        if not np.isfinite(centered).all():
            raise ValueError('centered coordinates are not representable')
        return self.shear.inverse(np.linalg.solve(np.asarray(self.lower), centered))

    def log_density(self, x):
        z = self.encode(x)
        with np.errstate(over='ignore'):
            squared_norm = float(z @ z)
        if not math.isfinite(squared_norm):
            raise ValueError('log density is not representable')
        return -0.5 * (self.dimension * math.log(2 * math.pi) + squared_norm) - self.log_abs_determinant


@dataclass(frozen=True, slots=True)
class PairTrace:
    source: int
    target: int
    noise: tuple[float, ...]

    def __post_init__(self):
        for name in ('source', 'target'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError('chart indices must be nonnegative integers')
            object.__setattr__(self, name, int(value))
        noise = np.asarray(self.noise, dtype=float)
        if noise.ndim != 1 or len(noise) == 0 or not np.isfinite(noise).all():
            raise ValueError('noise must be a finite nonempty vector')
        object.__setattr__(self, 'noise', tuple(float(v) for v in noise))


@dataclass(frozen=True, slots=True)
class PairStep:
    position: tuple[float, ...]
    inverse_trace: PairTrace
    source_latent: tuple[float, ...]
    target_latent: tuple[float, ...]
    log_extended_jacobian: float
    log_auxiliary_ratio: float
    log_correction: float


def pair_move(charts: Sequence[WarpedGaussianChart], x, trace: PairTrace, correlation: float):
    """Apply an extended involution; caller supplies independent N(0,I) noise.

    The reverse trace swaps charts and retains s*z-c*noise. At c=0 the physical
    fixed-noise derivative is singular, but the FULL (x,noise) Jacobian remains
    |det L_target|/|det L_source|. c=+/-1 is supported without special densities.
    This function performs no acceptance decision or chart-pair selection.
    """
    if not isinstance(trace, PairTrace) or max(trace.source, trace.target) >= len(charts):
        raise ValueError('trace must select available charts')
    a, b = charts[trace.source], charts[trace.target]
    if not isinstance(a, WarpedGaussianChart) or not isinstance(b, WarpedGaussianChart) or a.dimension != b.dimension:
        raise ValueError('selected charts must have the same Euclidean dimension')
    c = float(correlation)
    if not math.isfinite(c) or not -1 <= c <= 1:
        raise ValueError('correlation must lie in [-1,1]')
    noise = _vector(trace.noise, a.dimension, 'auxiliary noise')
    z = a.encode(x)
    sine = math.sqrt((1 - c) * (1 + c))
    with np.errstate(over='ignore', invalid='ignore'):
        target = c * z + sine * noise
        reverse_noise = sine * z - c * noise
        auxiliary = 0.5 * float(noise @ noise - reverse_noise @ reverse_noise)
    if not np.isfinite(target).all() or not np.isfinite(reverse_noise).all() or not math.isfinite(auxiliary):
        raise ValueError('extended move is not representable')
    y = b.decode(target)
    jacobian = b.log_abs_determinant - a.log_abs_determinant
    return PairStep(tuple(float(v) for v in y),
                    PairTrace(trace.target, trace.source, tuple(reverse_noise)),
                    tuple(float(v) for v in z), tuple(float(v) for v in target),
                    jacobian, auxiliary, jacobian + auxiliary)
