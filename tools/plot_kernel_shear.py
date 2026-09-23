#!/usr/bin/env python3
"""Illustrate an exact toy transfer; no protein sampling or learned-fit claim."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss

from kernel_shear import KernelShear, PairTrace, WarpedGaussianChart, pair_move


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve(shear, x):
    centers = np.asarray(shear.centers)[:, 0]
    coefficients = np.asarray(shear.coefficients)[:, 0]
    return np.exp(-.5 * ((np.asarray(x)[..., None] - centers) / shear.bandwidth) ** 2) @ coefficients


def moments(chart):
    # Gaussian quadrature for the known two-dimensional triangular model.
    nodes, weights = hermgauss(96)
    nodes, weights = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
    f = curve(chart.shear, nodes)
    mean_f = weights @ f
    covariance = np.array([[1., weights @ (nodes * f)],
                           [weights @ (nodes * f), 1. + weights @ (f * f) - mean_f * mean_f]])
    lower = np.asarray(chart.lower)
    return np.asarray(chart.mean) + lower @ [0., mean_f], lower @ covariance @ lower.T


def density(chart, grid):
    latent = np.linalg.solve(np.asarray(chart.lower), (grid - chart.mean).T).T
    latent[:, 1] -= curve(chart.shear, latent[:, 0])
    return np.exp(-.5 * np.sum(latent * latent, axis=1) - np.log(2 * np.pi)
                  - chart.log_abs_determinant)


def run(out):
    if out.exists():
        raise ValueError("Fresh figure directory required")
    first = KernelShear(2, (0,), (1,), ((0.,),), ((11.,),), bandwidth=.85)
    second = KernelShear(2, (0,), (1,), ((-1.,), (1.,)), ((9.,), (-9.,)), bandwidth=.65)
    charts = (WarpedGaussianChart((0., 0.), ((1., 0.), (0., .15)), first),
              WarpedGaussianChart((0., 0.), ((1., 0.), (0., .15)), second))
    latent = np.random.default_rng(1940723).normal(size=(700, 2))
    source = np.array([charts[0].decode(z) for z in latent])
    mean_a, cov_a = moments(charts[0])
    mean_b, cov_b = moments(charts[1])
    affine = mean_b + (np.linalg.cholesky(cov_b) @
                       np.linalg.solve(np.linalg.cholesky(cov_a), (source - mean_a).T)).T
    steps = [pair_move(charts, x, PairTrace(0, 1, (0., 0.)), 1.) for x in source]
    warped = np.array([step.position for step in steps])
    direct = np.array([charts[1].decode(z) for z in latent])
    residual = float(np.max(np.abs(warped - direct)))
    reverse = np.array([pair_move(charts, step.position, step.inverse_trace, 1.).position for step in steps])
    inverse_error = float(np.max(np.abs(reverse - source)))
    if max(residual, inverse_error) > 1e-12:
        raise ValueError("Toy transfer inverse check failed")
    x_axis = np.linspace(-3.3, 3.3, 170)
    y_axis = np.linspace(-2.1, 2.4, 160)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for chart in charts:
        select = grid[::997]
        direct_log = np.array([chart.log_density(x) for x in select])
        reconstructed = np.log(density(chart, select))
        if np.max(np.abs(direct_log - reconstructed)) > 1e-10:
            raise ValueError("Independent toy density reconstruction failed")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), sharex=True, sharey=True)
    subtitles = ("Source environment A", "Covariance-only transfer to B", "Kernel-warp transfer to B")
    colors = ("#2670a0", "#cf7b2a", "#2a846c")
    markers = [47, 65, 186, 350, 512]
    marker_colors = plt.cm.plasma(np.linspace(.05, .85, len(markers)))
    for ax, points, chart, title, color in zip(axes, (source, affine, warped),
                                             (charts[0], charts[1], charts[1]), subtitles, colors):
        den = density(chart, grid).reshape(xx.shape)
        peak = 1 / (2 * np.pi * np.exp(chart.log_abs_determinant))
        ax.contourf(xx, yy, den, levels=np.linspace(.01 * peak, peak, 12), cmap="Greys", alpha=.25)
        ax.contour(xx, yy, den, levels=peak * np.array([.05, .5]), colors="#444444", linewidths=.7)
        ax.scatter(points[:, 0], points[:, 1], s=5, color=color, alpha=.45, linewidths=0)
        ax.scatter(points[markers, 0], points[markers, 1], c=marker_colors, s=55, edgecolors="white", linewidths=.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Orientation-like coordinate")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.1, 2.4)
    axes[0].set_ylabel("Translation-like coordinate")
    fig.suptitle("Matching covariance does not match a curved contact relationship", fontsize=14)
    fig.text(.5, .045, "Schematic: known normalized 2D distributions; colored markers identify the same source samples.\n"
             "Gray contours show each destination density. Points are proposals before any physical acceptance.\n"
             "The nonlinear transfer has an explicit inverse and unit shear Jacobian. No protein speedup is inferred.",
             ha="center", fontsize=9.5)
    fig.subplots_adjust(left=.065, right=.99, bottom=.27, top=.81, wspace=.12)
    out.mkdir(parents=True)
    for extension in ("png", "svg"):
        fig.savefig(out / f"covariance-versus-kernel-transfer.{extension}", dpi=180)
    plt.close(fig)
    metadata = {
        "schema": "kernel-shear-toy-figure-v1", "complete": True,
        "scope": "Known normalized toy distributions; illustrative, not a fitted protein model or efficiency measurement.",
        "physical_draws": 0, "toy_draws": len(latent), "seed": 1940723,
        "transfer_reconstruction_error": residual, "inverse_error": inverse_error,
        "covariance_baseline": "96-point Gaussian quadrature gives full source and target means/covariances.",
        "code_sha256": {str(p): sha(p) for p in (Path(__file__).resolve(), Path(__file__).with_name("kernel_shear.py").resolve())},
        "files": {p.name: sha(p) for p in out.iterdir()},
    }
    (out / "figure.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out.resolve())
