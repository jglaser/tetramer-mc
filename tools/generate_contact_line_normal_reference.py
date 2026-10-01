#!/usr/bin/env python3
"""Independent Decimal reference integrals at exact binary64 endpoints."""
from decimal import Decimal as D, localcontext
import argparse
import json
from pathlib import Path


def atan_inverse(n):
    x=D(1)/n;x2=x*x;power=x;total=x;sign=-1;k=3
    while True:
        power*=x2;term=power/k;old=total;total+=sign*term
        if total==old:return total
        sign=-sign;k+=2


def mass(lo,hi,precision=110):
    with localcontext() as context:
        context.prec=precision
        a,b=D.from_float(lo),D.from_float(hi)
        if a==b:return D(0)
        middle=(a+b)/2;half=(b-a)/2
        # exp(-middle*t-t*t/2)=sum c_n t^n. Only even powers
        # survive integration on the symmetric interval [-half,+half].
        previous=D(1);current=-middle;power=half;total=2*half
        small=0
        for n in range(2,3000):
            previous,current=current,(-middle*current-previous)/n
            power*=half
            if n%2==0:
                term=2*current*power*half/(n+1);total+=term
                small=small+1 if abs(term)<abs(total)*D(10)**(-precision+15) else 0
                if small>=8:break
        else:raise ArithmeticError('Taylor integral did not converge')
        pi=16*atan_inverse(D(5))-4*atan_inverse(D(239))
        return +((-(middle*middle)/2).exp()*total/(2*pi).sqrt())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    bounds=[('zero',0.,0.),('central_tiny',-5e-13,5e-13),('central_positive',0.,1e-12),
        ('central_shifted',.01,.01+1e-12),('central_small',-.01,.02),('central_wide',-1.,1.),
        ('below_floor',0.,2.506628274631e-12),('above_floor',0.,2.506628274632e-12)]
    for z in (8.,20.):
        for w in (1e-12,1e-6,.01,1.):
            bounds.append((f'tail_pos_{z:g}_width_{w:g}',z,z+w))
            bounds.append((f'tail_neg_{z:g}_width_{w:g}',-z-w,-z))
    cases=[]
    for name,lo,hi in bounds:
        value=mass(lo,hi);check=mass(lo,hi,140)
        with localcontext() as context:
            context.prec=150
            assert value==check==0 or abs(value-check)/check<D('1e-85'),name
        cases.append(dict(name=name,lo=lo,hi=hi,mass=float(value),mass_decimal=str(value)))
    result=dict(schema='contact-line-normal-reference-v1',precision_decimal_digits=110,
        endpoint_convention='Integral at the exact binary64 values stored as lo/hi, using Decimal.from_float.',
        method='Independent midpoint Taylor integral of exp(-x²/2), 110 decimal digits; Machin-formula pi. Repeated at140 digits; relative agreement better than1e-85.',cases=cases)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(cases)} reference cases -> {args.out}')


if __name__=='__main__':main()
