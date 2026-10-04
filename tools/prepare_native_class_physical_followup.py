#!/usr/bin/env python3
"""Prepare three fresh class-guide sensitivity blocks, without launching them."""
import argparse
import json
from pathlib import Path
import native_class_physical_followup as followup

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(followup.prepare(args.out, args.inputs, args.review), indent=2))
