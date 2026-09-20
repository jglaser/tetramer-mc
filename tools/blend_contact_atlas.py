#!/usr/bin/env python3
"""Add explicit native charts to a geometry-only atlas, preserving both laws."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-weight', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.native_weight < 1:
        parser.error('native weight must lie strictly between zero and one')
    raw = [p.read_bytes() for p in (args.geometry, args.native)]
    models = [json.loads(r) for r in raw]
    first = models[0]
    if any(m['shape_sha256'] != first['shape_sha256'] or
           m['coordinate_convention'] != 'anchor-body-relative' for m in models):
        parser.error('models must share the physical shape and relative-pose convention')
    result = {k: first[k] for k in ('shape_sha256', 'coordinate_convention', 'angular_length')}
    for key in ('anchors', 'means', 'covariances', 'weights', 'dfs'):
        result[key] = []
    for model, mass in zip(models, (1 - args.native_weight, args.native_weight)):
        n = len(model['weights'])
        if any(x is not None for x in model.get('dfs', [None] * n)):
            parser.error('only Gaussian components are supported')
        scale = [1.] * 3 + [result['angular_length'] / model['angular_length']] * 3
        result['anchors'].extend(copy.deepcopy(model['anchors']))
        result['means'].extend([[x * s for x, s in zip(mu, scale)] for mu in model['means']])
        result['covariances'].extend([[[v * scale[i] * scale[j] for j, v in enumerate(row)]
                                      for i, row in enumerate(cov)] for cov in model['covariances']])
        normalization = sum(model['weights'])
        result['weights'].extend(mass * w / normalization for w in model['weights'])
        result['dfs'].extend([None] * n)
    result['construction'] = {
        'kind': 'geometry-plus-explicit-native-atlas-v1',
        'native_weight': args.native_weight,
        'geometry_components': len(first['weights']),
        'native_components': len(models[1]['weights']),
        'inputs': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(r).hexdigest()}
                   for p, r in zip((args.geometry, args.native), raw)],
        'angular_rescaling': 'means D*mu and covariances D*C*D preserve physical chart densities',
        'scope': 'Proposal information only; unchanged physical depletion target.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'components': len(result['weights'])}))


if __name__ == '__main__':
    main()
