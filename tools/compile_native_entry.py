#!/usr/bin/env python3
"""Materialize the complete frozen native-entry predicate for the Rust probe.

This exports geometry and residue patches, not an approximation to the native
label. It performs no physical sampling, fitting or normalizer calculation.
The original definition and every archived input are verified by the independent
Python classifier before export. Caller-side hard/capture/wall checks are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from native_contact_regions import NativeContactRegions, CRITERIA


def compile_definition(definition):
    model = NativeContactRegions(definition)
    result = dict(
        schema='native-entry-compiled-v1',
        source_definition_sha256=model.definition_sha256,
        source_input_sha256=model.definition['input_sha256'],
        criteria=dict(CRITERIA), fixed_poses=model.fixed_poses,
        members=[dict(position=p.tolist(), rotation=r.tolist())
                 for p, r in zip(model.member_positions, model.member_rotations)],
        monomer_atoms=[dict(center=p.tolist(), radius=float(r), residue=int(s))
                       for p, r, s in zip(model.atoms, model.radii, model.residues)],
        residue_count=model.residue_count,
        references=[dict(label=label, family=ref['family'],
                         position=ref['position'].tolist(), rotation=ref['rotation'].tolist(),
                         native_residue_pairs=sorted(ref['native_residue_pairs']))
                    for label, ref in model.references.items()],
        motifs=[dict(id=m['id'], position=model.motif_positions[i].tolist(),
                     rotation=model.motif_rotations[i].tolist(),
                     member_contacts=[{k: contact[k] for k in
                                       ('member_i', 'member_j', 'directed_class')}
                                      for contact in m['member_contacts']])
                for i, m in enumerate(model.motifs)])
    return result, model


def export(definition, out):
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh compiled-definition directory required')
    compiled, model = compile_definition(Path(definition).resolve())
    text = json.dumps(compiled, indent=2, allow_nan=False) + '\n'
    out.mkdir(parents=True)
    (out / 'compiled.json').write_text(text)
    receipt = dict(schema='native-entry-compilation-v1', complete=True,
        compiled_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_definition_sha256=model.definition_sha256,
        source_input_sha256=compiled['source_input_sha256'],
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        counts=dict(members=len(compiled['members']), atoms=len(compiled['monomer_atoms']),
                    residues=model.residue_count, references=len(compiled['references']),
                    motifs=len(compiled['motifs'])),
        scope='Complete instantaneous native-entry geometry. Hard validity is a caller precondition; '
              'capture, vessel, R4, depletion, and catalogue cycles are not part of native_any.',
        physical_draws=0, old_classifier_passes_replayed=0,
        rust_parity_established=False, smc_integration_established=False)
    (out / 'compilation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--definition', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.definition, args.out), indent=2))
