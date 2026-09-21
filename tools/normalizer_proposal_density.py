"""Independent pose density using the frozen sampler's effective Gaussian factors.

Scalar binary64 Cholesky and LAPACK are both backward stable yet can define
measurably different normalized laws for ill-conditioned covariance inputs.
The sampler draws and evaluates with the same scalar factor. Reconstruct that
factor, certify its covariance residual independently, and retain SciPy pose,
inverse-pose, Cayley and triangular-solve density calculations.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from prepare_smc_normalizer_atlas import Density, unwrap_proposal_model

ALGORITHM = 'rust-symmetric-scalar-cholesky-binary64-v1'
SOURCE_EXCERPT_SHA256 = '09ddf40beec7203b00ef5e837bfcb359bd8bd3a265034644ba9d03875d8887ea'
UNIT_ROUNDOFF = Fraction(1, 2**53)
HALF_SUBNORMAL_ULP = Fraction(1, 2**1075)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def symmetric_covariance(covariance):
    c = np.asarray(covariance, dtype=float)
    require(c.shape == (6, 6) and np.isfinite(c).all(), 'Finite 6x6 covariance required')
    scale = float(np.max(abs(c)))
    require(scale > 0 and np.all(abs(c-c.T) <= 1e-12*(scale+abs(c.T))), 'Covariance symmetry precondition failed')
    symmetric = np.empty((6, 6))
    for i in range(6):
        for j in range(6):symmetric[i,j] = .5*(float(c[i,j])+float(c[j,i]))
    require(np.isfinite(symmetric).all(), 'Unrepresentable symmetrized covariance')
    return c, symmetric


def scalar_cholesky(covariance):
    """The source-bound ordered multiply/subtract factorization, with no jitter."""
    _, c = symmetric_covariance(covariance)
    lower = np.zeros((6,6))
    for i in range(6):
        for j in range(i+1):
            remainder = float(c[i,j])
            for k in range(j):remainder -= float(lower[i,k])*float(lower[j,k])
            if i == j:
                require(math.isfinite(remainder) and remainder > 0, 'Covariance is not strictly positive definite')
                lower[i,j] = math.sqrt(remainder)
            else:
                lower[i,j] = remainder/float(lower[j,j])
                require(math.isfinite(lower[i,j]), 'Unrepresentable Cholesky factor')
    return lower


def certify_covariance(covariance, lower):
    """Exact-rational componentwise residual against a derived operation bound.

    For each multiply/subtract, |fl(x)-x| <= u|x|+eta, where u=2^-53
    and eta=2^-1075 includes gradual underflow. For off-diagonal division,
    |fl(r/l)*l-r| <= u|r|+eta|l|. For diagonal sqrt, the squared error is
    <= (2u+u²)|r|+2eta(1+u)max(1,|r|)+eta². Sum these bounds along the
    actual unblocked recurrence. The independently formed exact rational
    residual sum_k L_ik L_jk-S_ij must satisfy the bound for EVERY entry.
    No matrix-norm tolerance or covariance-condition multiplier is used.
    """
    raw, c = symmetric_covariance(covariance)
    lower = np.asarray(lower, dtype=float)
    require(lower.shape == (6,6) and np.isfinite(lower).all() and np.all(np.diag(lower)>0)
            and np.count_nonzero(np.triu(lower,1)) == 0, 'Invalid lower factor')
    f = lambda x: Fraction.from_float(float(x))
    u, eta = UNIT_ROUNDOFF, HALF_SUBNORMAL_ULP
    l = [[f(v) for v in row] for row in lower]
    residuals, bounds, ratios = [], [], []
    for i in range(6):
        for j in range(i+1):
            remainder = float(c[i,j]);bound = Fraction(0)
            for k in range(j):
                exact_product = l[i][k]*l[j][k]
                product = float(lower[i,k])*float(lower[j,k])
                require(math.isfinite(product), 'Factor recurrence overflow')
                exact_subtraction = f(remainder)-f(product)
                bound += u*abs(exact_product)+eta + u*abs(exact_subtraction)+eta
                remainder -= product
                require(math.isfinite(remainder), 'Factor recurrence overflow')
            r = abs(f(remainder))
            if i == j:
                require(remainder > 0, 'Factor recurrence lacks positive pivot')
                bound += (2*u+u*u)*r+2*eta*(1+u)*max(Fraction(1),r)+eta*eta
            else:
                bound += u*r+eta*abs(l[j][j])
            residual = sum(l[i][k]*l[j][k] for k in range(j+1))-f(c[i,j])
            require(abs(residual) <= bound, f'Covariance backward-error certificate failed at ({i},{j})')
            residuals.append(float(abs(residual)));bounds.append(float(bound));ratios.append(float(abs(residual)/bound))
    return dict(componentwise_entries_checked=21,maximum_componentwise_bound_ratio=max(ratios),
        maximum_absolute_covariance_residual=max(residuals),maximum_derived_roundoff_bound=max(bounds),
        maximum_input_covariance_asymmetry=float(np.max(abs(raw-raw.T))),
        target='Binary64 symmetric matrix S_ij=0.5*(C_ij+C_ji)',
        certificate='Exact rational L L^T residual at each lower-triangle entry against accumulated IEEE multiply/subtract/divide/sqrt rounding bounds')


def bind_source_bundle(path):
    raw = Path(path).read_bytes();bundle = json.loads(raw)
    entry = bundle['files']['src/proposal.rs']
    source = entry['text']
    require(hashlib.sha256(source.encode()).hexdigest() == entry['sha256'], 'Proposal source entry hash mismatch')
    start = source.index('fn prepare_cholesky(');end = source.index('/// The finite Cayley coordinate',start)
    excerpt = source[start:end]
    require(hashlib.sha256(excerpt.encode()).hexdigest() == SOURCE_EXCERPT_SHA256,
            'Unknown sampler Cholesky/draw/density implementation; review a new algorithm binding')
    return dict(source_bundle_sha256=hashlib.sha256(raw).hexdigest(),proposal_source_sha256=entry['sha256'],
                factor_draw_density_excerpt_sha256=SOURCE_EXCERPT_SHA256)


class NormalizerProposalDensity(Density):
    """Active reciprocal normalizer density with certified actual factors.

    This single factorization matches the active-reciprocal, scale-one direct
    model path only. The legacy/all-false normalizer exports L L^T and refactors
    after scaling; it is deliberately rejected here.
    """
    def __init__(self, model, *, source_bundle=None):
        _,flags=unwrap_proposal_model(model)
        require(any(flags), 'Effective-factor helper requires the active reciprocal scale-one direct-model path')
        super().__init__(model)
        self.algorithm_metadata = dict(algorithm=ALGORITHM,source_excerpt_sha256=SOURCE_EXCERPT_SHA256,
            source_binding=bind_source_bundle(source_bundle) if source_bundle is not None else None,
            geometry='Unchanged independent SciPy matrix/pose inverse and Cayley/Haar density',
            scope='Reconstruct coefficients used by both the frozen Gaussian draws and their density; no proposal refit, jitter, weight correction or audit-tolerance change')
        factors=[];self.factor_diagnostics=[]
        for k,c in enumerate(self.model['covariances']):
            lower=scalar_cholesky(c);certificate=certify_covariance(c,lower)
            _,symmetric=symmetric_covariance(c)
            lapack=np.linalg.cholesky(symmetric)
            a=solve_triangular(lapack,lower,lower=True)
            singular=np.linalg.svd(a,compute_uv=False)
            certificate.update(component_index=k,covariance_condition=float(np.linalg.cond(symmetric)),
                effective_logdet_half=float(np.log(np.diag(lower)).sum()),
                lapack_logdet_half=float(np.log(np.diag(lapack)).sum()),
                scalar_minus_lapack_logdet_half=float(np.log(np.diag(lower)).sum()-np.log(np.diag(lapack)).sum()),
                lapack_whitened_factor_singular_values=singular.tolist(),
                lapack_whitened_covariance_maximum_eigenvalue_deviation=float(np.max(abs(singular**2-1))))
            self.factor_diagnostics.append(certificate);factors.append(lower)
        self.lower=np.asarray(factors)[self.base_indices]
        self.logdet=np.log(np.diagonal(self.lower,axis1=1,axis2=2)).sum(axis=1)
        # Rust accumulates left-to-right, then normalizes stored base weights.
        # Python's built-in sum may use compensated summation on newer versions.
        total=0.
        for w in self.model['weights']:total+=float(w)
        require(math.isfinite(total) and total>0 and abs(total-1)<=2e-12, 'Invalid normalized base weights')
        self.weights=np.asarray([(float(self.model['weights'][k])/total)/(2 if self.reciprocal_components[k] else 1)
                                 for k in self.base_indices])
