#!/usr/bin/env python3
"""Admit completed sensitivity blocks, then compare saved independent summaries.

Run from the frozen campaign code directory after its 121-stage controller has
completed and drained. This entry point launches no child and never redraws.
Admission may hash scientific files as bytes; comparison reads metadata only.
"""
import argparse
import json
from pathlib import Path
import native_class_physical_followup as followup


def run(protocol_path, digest, phase):
    bindings = followup.stage.Inputs()
    protocol, execution = followup.stage.load_protocol(protocol_path, digest, bindings)
    followup.protocol_inventory(protocol)
    root = Path(protocol['root']); output = followup.stage.paths(root)
    if phase == 'admission':
        followup.require(not output['admission_plan'].exists() and not output['admission'].exists(),
                         'Follow-up admission must be new')
        plan = followup.stage.materialize_admission(protocol, execution, bindings)
        result = followup.join(plan)
        followup.write(output['admission'], result)
    else:
        destination = root/'analysis/external-primary-comparison.json'
        followup.require(not destination.exists(), 'Follow-up comparison must be new')
        result = followup.compare(dict(path=str(Path(protocol_path).resolve()), sha256=digest),
                                  followup.base.bound(output['admission']))
        followup.write(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--phase', choices=('admission', 'comparison'), required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.protocol_sha256, args.phase)
    print(json.dumps(dict(complete=result['complete'], phase=args.phase)))
