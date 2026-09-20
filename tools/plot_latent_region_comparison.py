#!/usr/bin/env python3
"""Compare independent estimators of the same predeclared physical region."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
direct=ROOT/"runs/latent-region-uniform-ball-16384-l64"
read=lambda path:json.loads(path.read_text())
d=read(direct/"assessment/analysis.json")
records=[]
for scale in [1,2,4]:
    root=ROOT/f"runs/basin-normalizer-deep-far-std{scale}-8192-l64"
    region=read(root/"assessment/fixed-region.json")
    assert region["region_sha256"]==d["region_sha256"]
    full=read(root/"assessment/analysis.json")["groups"]["1.0"]
    e=region["estimates"]["discovered_ellipsoid"]
    records.append(dict(label=f"Gaussian\nwidth {scale}",estimate=e,cpu=full["sampler_cpu_seconds"],
        population_logs=e["independent_population_estimates"]["logQ_values"]))
records.append(dict(label="Uniform\n6-ball",estimate=d["estimate"],cpu=d["sampler_cpu_seconds"],
    population_logs=d["estimate"]["independent_populations"]["logQ_values"]))
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={"width_ratios":[1.5,1]})
colors=["#8295a8"]*3+["#087f8c"]
x=np.arange(4)
for i,r in enumerate(records):
    e=r["estimate"];mean=e["logQ"];se=e["relative_se"]
    axes[0].errorbar(i,mean,yerr=[[-np.log1p(-se)],[np.log1p(se)]],fmt="o",color=colors[i],capsize=4,markersize=7)
    axes[0].scatter(i+np.linspace(-.12,.12,4),r["population_logs"],s=23,facecolors="none",edgecolors=colors[i],alpha=.85)
axes[0].axhline(d["estimate"]["logQ"],color=colors[-1],alpha=.45,lw=1)
axes[0].set_ylabel("log physical mass of the fixed region")
axes[0].set_title("Same ellipsoid, independent fresh draws")
rate=[r["estimate"]["ess"]/r["cpu"] for r in records]
axes[1].bar(x,rate,color=colors)
for i,v in enumerate(rate):axes[1].text(i,v+.013,f"{v:.3f}",ha="center",fontsize=9)
axes[1].set_ylim(0,max(rate)*1.2)
axes[1].set_ylabel("Observed regional ESS / CPU second")
axes[1].set_title("Regional integration efficiency")
for ax in axes:
    ax.set_xticks(x,[r["label"] for r in records]);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
fig.suptitle("Direct integration removes Gaussian tail weighting",fontsize=13,y=.99)
fig.text(.02,.015,"Bars: observed ±1 SE in linear mass, mapped to log. Open circles: four independent populations.\nRegion only: original q ≥ 5 and fixed width-one Mahalanobis radius ≤ 3. No global convergence claim.",fontsize=8)
fig.tight_layout(rect=[0,.10,1,.94])
for ext in ["png","svg","pdf"]:fig.savefig(direct/f"assessment/method-comparison.{ext}",dpi=180)
(direct/"assessment/method-comparison.json").write_text(json.dumps(dict(region_sha256=d["region_sha256"],records=records),indent=2)+"\n")
print(direct/"assessment/method-comparison.png")
