#!/usr/bin/env python3
"""Exact-binary64 tiny-triangle area references, evaluated with Decimal."""
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path


def references():
    result=[];bounds=[[2.,2.1],[2.,2.1]]
    # Explicit binary64 distances bracket the predeclared area floor, including
    # adjacent representable values around the nominal square-root threshold.
    center=float(4.2-math.sqrt(2e-16))
    cases=[('half_floor',float(4.2-math.sqrt(1e-16))),
        ('twice_floor',float(4.2-math.sqrt(4e-16))),
        ('near_floor_below_distance',math.nextafter(center,-math.inf)),
        ('near_floor_center',center),('near_floor_above_distance',math.nextafter(center,math.inf)),
        ('tangent',4.2),('beyond_tangent',math.nextafter(4.2,math.inf))]
    with localcontext() as context:
        context.prec=100
        for name,D in cases:
            delta=Decimal.from_float(bounds[0][1])+Decimal.from_float(bounds[1][1])-Decimal.from_float(D)
            area=max(delta,Decimal(0))**2/2
            result.append(dict(name=name,bounds=bounds,distance=D,area=float(area),area_decimal=str(area),
                fallback_at_1e_16=area<=Decimal.from_float(1e-16)))
    return dict(schema='contact-distance-polygon-reference-v1',method='100-digit Decimal, exact binary64 upper endpoints and center distances; right-isosceles radius triangle area=max(hi1+hi2-D,0)^2/2.',cases=result)


if __name__=='__main__':
    destination=Path(__file__).resolve().parents[1]/'tests/data/contact_distance_polygon_reference.json'
    destination.write_text(json.dumps(references(),indent=2,allow_nan=False)+'\n')
    print(destination)
