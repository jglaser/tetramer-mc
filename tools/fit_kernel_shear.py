"""Fit frozen nonlinear contact charts without modifying a physical sampler.

The complete proposal is alpha*Uniform(original u ball) plus an untruncated
mixture of the existing invertible KernelShear charts. A fixed nonsingular
linear transform y=T*u permits angular-first conditional whitening. Its Jacobian
is included exactly. The original integration region is never redefined in the
new component coordinates, and no valid-only renormalization is performed.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from kernel_shear import KernelShear, WarpedGaussianChart


def _require(condition, message):
    if not condition: raise ValueError(message)


def _points(value, dimension):
    value = np.asarray(value, float)
    _require(value.ndim == 2 and value.shape[1] == dimension and np.isfinite(value).all(),
             'Coordinates must be a finite N by dimension matrix')
    return value


def _tuples(value):
    return tuple(tuple(float(x) for x in row) for row in value)


def _whiten(chart, transformed):
    return solve_triangular(np.asarray(chart.lower), (transformed-np.asarray(chart.mean)).T,
                            lower=True, check_finite=False).T


def features(points, centers, bandwidth):
    points = np.asarray(points, float); centers = np.asarray(centers, float)
    _require(points.ndim == centers.ndim == 2 and points.shape[1] == centers.shape[1]
        and np.isfinite(points).all() and np.isfinite(centers).all()
        and math.isfinite(bandwidth) and bandwidth > 0, 'Invalid RBF geometry')
    with np.errstate(over='ignore', under='ignore'):
        delta = (points[:,None,:]-centers[None,:,:])/bandwidth
        return np.exp(-.5*np.sum(delta*delta,axis=2))


def chart_log_density(chart, transformed, block=8192):
    """Batch version of the already checked scalar chart inverse and density."""
    values = _points(transformed, chart.dimension)
    result = np.empty(len(values)); a = list(chart.shear.conditioning); b = list(chart.shear.shifted)
    centers = np.asarray(chart.shear.centers).reshape(-1,len(a))
    coefficients = np.asarray(chart.shear.coefficients).reshape(len(centers),len(b))
    for first in range(0,len(values),block):
        v = _whiten(chart,values[first:first+block])
        if len(centers):
            v[:,b] -= features(v[:,a],centers,chart.shear.bandwidth) @ coefficients
        result[first:first+len(v)] = -.5*(chart.dimension*math.log(2*math.pi)+np.sum(v*v,axis=1))-chart.log_abs_determinant
    return result


@dataclass(frozen=True, slots=True)
class KernelMixture:
    alpha: float
    radius: float
    transform: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]
    charts: tuple[WarpedGaussianChart, ...]

    def __post_init__(self):
        charts = tuple(self.charts)
        _require(charts and all(isinstance(c,WarpedGaussianChart) for c in charts), 'At least one immutable chart required')
        d = charts[0].dimension
        _require(d >= 2 and all(c.dimension == d for c in charts), 'Chart dimensions differ')
        transform = np.asarray(self.transform,float); weights = np.asarray(self.weights,float)
        _require(transform.shape == (d,d) and np.isfinite(transform).all(), 'Invalid coordinate transform')
        sign,logdet = np.linalg.slogdet(transform)
        _require(sign != 0 and math.isfinite(logdet), 'Coordinate transform must be nonsingular')
        _require(weights.shape == (len(charts),) and np.isfinite(weights).all()
            and np.all(weights>=0) and np.max(weights)>0, 'Invalid mixture weights')
        _require(math.isfinite(self.alpha) and 0<=self.alpha<=1
            and math.isfinite(self.radius) and self.radius>0, 'Invalid defensive component')
        object.__setattr__(self,'alpha',float(self.alpha)); object.__setattr__(self,'radius',float(self.radius))
        object.__setattr__(self,'transform',_tuples(transform))
        scaled=weights/np.max(weights)
        object.__setattr__(self,'weights',tuple(float(w) for w in scaled/scaled.sum()))
        object.__setattr__(self,'charts',charts)

    @property
    def dimension(self): return self.charts[0].dimension

    @property
    def log_abs_transform_determinant(self): return float(np.linalg.slogdet(self.transform)[1])

    @property
    def log_ball_volume(self):
        d=self.dimension
        return d*math.log(self.radius)+.5*d*math.log(math.pi)-math.lgamma(.5*d+1)

    def transformed(self, points):
        return _points(points,self.dimension) @ np.asarray(self.transform).T

    def log_density(self, points):
        points = _points(points,self.dimension); y=self.transformed(points)
        uniform = math.log(self.alpha)-self.log_ball_volume if self.alpha>0 else -math.inf
        result = np.where(np.sum(points*points,axis=1)<=self.radius**2,uniform,-math.inf)
        if self.alpha < 1:
            for weight,chart in zip(self.weights,self.charts):
                if weight > 0:
                    term = math.log1p(-self.alpha)+math.log(weight)+self.log_abs_transform_determinant
                    result = np.logaddexp(result,term+chart_log_density(chart,y))
        return result

    @classmethod
    def from_guide(cls, guide, transform, radius=4.):
        _require(guide.get('schema')=='defensive-latent-shell-guide-v1', 'Unknown baseline guide schema')
        transform=np.asarray(transform,float); components=guide['gaussian_components']; charts=[]
        _require(components, 'Baseline guide needs Gaussian components')
        d=len(components[0]['mean']); a=tuple(range(d//2)); b=tuple(range(d//2,d))
        _require(transform.shape==(d,d) and np.isfinite(transform).all(), 'Invalid coordinate transform')
        for item in components:
            mean=np.asarray(item['mean'],float); covariance=np.asarray(item['covariance'],float)
            _require(mean.shape==(d,) and covariance.shape==(d,d) and np.isfinite(mean).all()
                and np.isfinite(covariance).all() and np.allclose(covariance,covariance.T,rtol=0,atol=1e-10),
                'Invalid Gaussian component')
            cov=transform@covariance@transform.T
            lower=np.linalg.cholesky((cov+cov.T)/2)
            charts.append(WarpedGaussianChart(tuple(transform@mean),_tuples(lower),KernelShear(d,a,b,(),())))
        return cls(guide['defensive_uniform_shell_probability'],radius,_tuples(transform),
                   tuple(c['weight'] for c in components),tuple(charts))

    def to_dict(self):
        return dict(schema='frozen-kernel-shear-mixture-v1',alpha=self.alpha,radius=self.radius,
            transform=[list(row) for row in self.transform],weights=list(self.weights),
            charts=[dict(mean=list(c.mean),lower=[list(row) for row in c.lower],
                shear=dict(dimension=c.dimension,conditioning=list(c.shear.conditioning),
                    shifted=list(c.shear.shifted),centers=[list(row) for row in c.shear.centers],
                    coefficients=[list(row) for row in c.shear.coefficients],bandwidth=c.shear.bandwidth))
                for c in self.charts])

    @classmethod
    def from_dict(cls,value):
        _require(value.get('schema')=='frozen-kernel-shear-mixture-v1', 'Unknown frozen nonlinear model')
        charts=tuple(WarpedGaussianChart(c['mean'],c['lower'],KernelShear(**c['shear'])) for c in value['charts'])
        return cls(value['alpha'],value['radius'],value['transform'],value['weights'],charts)


def _centers(points, weights, count, rng):
    """Bounded weighted kmeans++ seeding; fitting randomness only, no physics."""
    selected=[]; nearest=np.full(len(points),math.inf)
    probability=weights.copy()
    for _ in range(count):
        total=float(probability.sum())
        if total<=0: break  # All positive-weight conditioning positions coincide.
        index=min(int(np.searchsorted(np.cumsum(probability),rng.random()*total,side='right')),len(points)-1)
        selected.append(points[index].copy())
        nearest=np.minimum(nearest,np.sum((points-points[index])**2,axis=1))
        probability=weights*nearest
    return np.asarray(selected).reshape(len(selected),points.shape[1])


def fit_shears(model, points, log_weights, centers=16, bandwidth=1., ridge=.01,
               minimum_ess=64., seed=48020260924):
    """One frozen-responsibility fit, using TRAINING data only.

    Caller supplies the desired training measure, e.g. three equally weighted
    contact classes with original positive linear importance weights. Each
    component's regression is normalized by its responsibility mass. Thus ridge
    is a conditional average penalty; the overall mixture surrogate penalty is
    mass_k * ridge * ||coefficients_k||^2 / 2. No covariance or weight is fitted.
    """
    _require(isinstance(model,KernelMixture), 'Frozen baseline model required')
    _require(type(centers) is int and centers>0 and math.isfinite(bandwidth) and bandwidth>0
        and math.isfinite(ridge) and ridge>0 and math.isfinite(minimum_ess) and minimum_ess>0,
        'Invalid fixed regression settings')
    _require(type(seed) is int and seed>=0, 'Nonnegative integer fitting seed required')
    points=_points(points,model.dimension); log_weights=np.asarray(log_weights,float)
    _require(log_weights.shape==(len(points),) and not np.isnan(log_weights).any()
        and not np.isposinf(log_weights).any() and np.isfinite(log_weights).any(), 'Invalid training weights')
    _require(all(not c.shear.centers for c in model.charts), 'Fit starts from affine charts only')
    shifted=log_weights-float(np.max(log_weights))
    normalized=shifted-float(logsumexp(shifted)); w=np.exp(normalized)
    y=model.transformed(points); baseline=model.log_density(points)
    _require(np.isfinite(baseline[w>0]).all(), 'Training measure lies outside proposal support')
    results=[]; fitted=[]
    for index,(chart,mix_weight) in enumerate(zip(model.charts,model.weights)):
        if model.alpha==1 or mix_weight==0:
            fitted.append(chart);results.append(dict(index=index,responsibility_mass=0.,effective_samples=0.,
                fitted=False,reason='Zero Gaussian proposal mass',centers=0));continue
        component_log=math.log1p(-model.alpha)+math.log(mix_weight)+model.log_abs_transform_determinant+chart_log_density(chart,y)
        logw=normalized+component_log-baseline
        logmass=float(logsumexp(logw)); component_weights=np.exp(logw-logmass)
        mass=math.exp(logmass); ess=float(1/np.sum(component_weights**2))
        row=dict(index=index,responsibility_mass=mass,effective_samples=ess,fitted=False,centers=0)
        if ess<minimum_ess:
            row['reason']='Training responsibility ESS below fixed threshold'
            results.append(row); fitted.append(chart);continue
        v=_whiten(chart,y); a=list(chart.shear.conditioning); b=list(chart.shear.shifted)
        dictionary=_centers(v[:,a],component_weights,centers,np.random.default_rng(seed+index))
        gram=np.zeros((len(dictionary),len(dictionary))); rhs=np.zeros((len(dictionary),len(b)))
        for first in range(0,len(points),8192):
            k=features(v[first:first+8192,a],dictionary,bandwidth); ww=component_weights[first:first+8192]
            gram+=k.T@(ww[:,None]*k)
            rhs+=k.T@(ww[:,None]*v[first:first+8192,b])
        system=gram+ridge*np.eye(len(dictionary)); coefficients=np.linalg.solve(system,rhs)
        objective_before=float(.5*np.sum(component_weights[:,None]*v[:,b]**2))
        error=0.
        for first in range(0,len(points),8192):
            residual=v[first:first+8192,b]-features(v[first:first+8192,a],dictionary,bandwidth)@coefficients
            error+=float(.5*np.sum(component_weights[first:first+8192,None]*residual**2))
        penalty=float(.5*ridge*np.sum(coefficients**2)); residual=float(np.max(np.abs(system@coefficients-rhs)))
        _require(error+penalty<=objective_before+1e-10, 'Regression did not improve its fixed surrogate')
        shear=KernelShear(model.dimension,a,b,dictionary,coefficients,bandwidth)
        fitted.append(WarpedGaussianChart(chart.mean,chart.lower,shear))
        row.update(fitted=True,centers=len(dictionary),conditional_objective_before=objective_before,
            conditional_objective_after=error,coefficient_penalty=penalty,
            normal_equation_max_residual=residual)
        results.append(row)
    result=KernelMixture(model.alpha,model.radius,model.transform,model.weights,tuple(fitted))
    positive=w>0
    proposal_gain=float(np.sum(w[positive]*(result.log_density(points[positive])-baseline[positive])))
    _require(proposal_gain>=-1e-9, 'Fixed-responsibility fit reduced training mixture likelihood')
    return result,dict(components=results,settings=dict(centers=centers,bandwidth=bandwidth,ridge=ridge,
        minimum_ess=minimum_ess,seed=seed),training_rows=len(points),
        training_positive_weight_rows=int(np.sum(w>0)),training_weight_ESS=float(1/np.sum(w*w)),
        training_weighted_log_density_gain=proposal_gain,
        conditioning='First half of transformed angular-first coordinates; conditional translation whitening.',
        penalty='Per-component conditional average; global surrogate coefficient penalty weighted by responsibility mass.',
        physical_draws=0,production_kernels_changed=False)
