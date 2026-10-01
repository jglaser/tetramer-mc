#!/usr/bin/env python3
"""Decimal references for near-tangent circle/sphere overlap angles."""
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path


def atan(x):
    term=x;total=x;n=1
    while True:
        term*=-x*x;addition=term/(2*n+1);new=total+addition
        if new==total:return total
        total=new;n+=1


def asin(x):
    term=x;total=x;n=1
    while True:
        term*=x*x*Decimal((2*n-1)**2)/Decimal(2*n*(2*n+1));new=total+term
        if new==total:return total
        total=new;n+=1


def references():
    cases=[]
    with localcontext() as context:
        context.prec=100;pi=16*atan(Decimal(1)/5)-4*atan(Decimal(1)/239)
        for tag,rho,offset in [('integer',1.,[2.,0.,0.]),('dyadic',1.25,[2.,0.,0.]),('oblique',1.3,[2.1,0.,.4])]:
            p=math.hypot(offset[0],offset[1]);minimum=math.hypot(p-rho,offset[2]);maximum=math.hypot(p+rho,offset[2])
            coords=list(map(Decimal.from_float,offset));q=(coords[0]**2+coords[1]**2).sqrt();r=Decimal.from_float(rho);z=coords[2]
            min2=(q-r)**2+z*z;max2=(q+r)**2+z*z
            for boundary,value in [('min',minimum),('max',maximum)]:
                for direction,R in [('below',math.nextafter(value,-math.inf)),('exact_fp64',value),('above',math.nextafter(value,math.inf))]:
                    s=Decimal.from_float(R);s2=s*s
                    if s2<=min2:topology='none';allowed=2*pi
                    elif s2>max2:topology='all';allowed=Decimal(0)
                    elif s2==max2:topology='all_except_tangent_point';allowed=Decimal(0)
                    else:
                        topology='partial'
                        if (s2-min2)<(max2-s2):allowed=2*pi-4*asin(((s2-min2)/(4*r*q)).sqrt())
                        else:allowed=4*asin(((max2-s2)/(4*r*q)).sqrt())
                    fp64='none' if R<=minimum else ('all' if R>maximum else ('all_except_tangent_point' if R==maximum else 'partial'))
                    cases.append(dict(name=f'{tag}_{boundary}_{direction}',circle_radius=rho,fixed_offset=offset,radius_sum=R,
                        fp64_minimum_distance=minimum,fp64_maximum_distance=maximum,fp64_topology=fp64,analytic_topology=topology,
                        classification_ambiguous=fp64!=topology,allowed_length=float(allowed),allowed_length_decimal=str(allowed)))
    return dict(schema='contact-circle-tangent-reference-v1',method='100-digit Decimal, exact binary64 inputs; Machin pi and convergent asin series near the appropriate boundary; no double-precision acos or clamping.',cases=cases)


if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'tests/data/contact_circle_tangent_reference.json'
    out.write_text(json.dumps(references(),indent=2,allow_nan=False)+'\n');print(out)
