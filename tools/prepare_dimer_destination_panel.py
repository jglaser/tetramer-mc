#!/usr/bin/env python3
"""Freeze a reset-event dimer panel without inspecting proposal outcomes.

The full eligible table and source event streams accompany the selected panel.
This is a conditional saved-snapshot diagnostic, not a population sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'runs/contact-anchor-benchmark-20260926'
EXPECTED = {
    'frame7400.json': 'e69e6c4696eb7d8fece5c2ef025364620dc21d7db7a5d92ff5ff6a896ed59b2e',
    'contact-anchor.json': 'f310ec8cf3ff4aa4b915be7dec16643b36d59e9293acf504eeb9f2dfbec1ddea',
    'freeze-manifest.json': '0561abfd14eb5f5134907490ebbb8195dee8b281d25cca965d377b32607097fb',
    'events-p0.jsonl': '581c29ae2b72a892f6d4357ab4adec863a0ca416faef800e066d64a8db80cc68',
    'events-p1.jsonl': 'd128d01d8a1b1868fa4ffd19562be52d23b48289039578e73f487ae061dae61a',
    'events-p2.jsonl': 'f24df487a349ed102f40cacbe9df54de592f306965387bb66f74e9f299537289',
    'events-p3.jsonl': '1b5369e605705dfecd32fc834d5b7f2212e46a1c48f5b6616822f7f949853032',
}
ATLAS_SOURCES = [
    ('blind_memory_allslot64', ROOT / 'runs/native-blind-memory-proposal-preparation-20260921/model.json',
     'b6d06b0a076d7f3cd4f591d116a77799b2dc4caa3ea6c21081ad5da8a45b3e9b'),
    ('blind_fft512slots', ROOT / 'examples/frozen-blind-contact-mixture-512.json',
     'c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08'),
    ('native_informed178', ROOT / 'examples/frozen-coverage-reciprocal-mixture.json',
     'feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e'),
]
SELECTION_RULE = {
    'eligible': 'Only event==1, exactly two distinct members, with old_poses exactly equal to the frozen frame at those labels. No outcome, acceptance, hard-validity, contact change, seed or native-label field is inspected.',
    'pair_table': 'Deduplicate sorted member pairs across all four event streams; require consistent parent-component and whole-component fields; retain every eligible witness.',
    'selected_pairs': 'In order whole_component=true then false, take the lexicographically first two eligible pairs belonging to distinct archived parent components within that stratum.',
    'root': 'The smaller selected label; child is the larger label.',
    'anchors': 'First the nearest nonmember by squared center distance to root, ties by label; then the nearest body outside the archived parent component, distinct from the first anchor, with the same ordering.',
    'source_reset': 'Every attempted proposal starts from the same immutable sweep-7400 frame.',
    'uniform_half_width': 160.0,
    'scope': 'Purposive geometric strata within one saved snapshot; neither equilibrium nor a native-blind source-state claim.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_sum(values):
    """Keep the original Python 3.9 panel arithmetic under newer Python too."""
    total = 0.
    for value in values:
        total += float(value)
    return total


def collect_pairs(state, streams):
    """streams: (name, list of parsed rows). Never consult an outcome field."""
    pairs, first_events = {}, []
    for name, rows in streams:
        phases = set()
        for line, row in enumerate(rows, 1):
            if row.get('event') != 1:
                continue
            phase = row['phase']
            require(phase not in phases, f'Duplicate first event: {name}/{phase}')
            phases.add(phase)
            members = row['members']
            witness = dict(stream=name, line=line, phase=phase, event=1)
            require(len(members) == len(set(members)), 'Repeated member')
            require(all(type(i) is int and 0 <= i < len(state) for i in members), 'Invalid member')
            require(row['old_poses'] == [state[i] for i in members], f'Non-reset first event: {witness}')
            first_events.append(dict(**witness, members=members, eligible=len(members) == 2))
            if len(members) != 2:
                continue
            pair = tuple(sorted(members))
            context = row['subset_context']
            parent = sorted(context['parent_component_members'])
            whole = context['whole_component']
            require(type(whole) is bool and len(parent) == len(set(parent)) and set(pair) <= set(parent), 'Invalid parent component')
            require(whole == (list(pair) == parent), 'Whole-component flag inconsistent')
            value = pairs.setdefault(pair, dict(members=list(pair), parent_component_members=parent,
                                               whole_component=whole, witnesses=[]))
            require(value['parent_component_members'] == parent and value['whole_component'] == whole,
                    f'Inconsistent reset context for {pair}')
            value['witnesses'].append(witness)
    return [pairs[p] for p in sorted(pairs)], first_events


def select_pairs(table, per_stratum=2):
    selected = []
    for whole in (True, False):
        parents = set()
        group = []
        for entry in sorted(table, key=lambda e: e['members']):
            parent = tuple(entry['parent_component_members'])
            if entry['whole_component'] == whole and parent not in parents:
                group.append(entry)
                parents.add(parent)
                if len(group) == per_stratum:
                    break
        require(len(group) == per_stratum, f'Insufficient distinct parent components: {whole}')
        selected.extend(group)
    return selected


def relative_translation(fixed, moving):
    # The inverse unit-quaternion rotation, independent of the Rust pose helpers.
    w, x, y, z = fixed['orientation']
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = (v/norm for v in (w, x, y, z))
    r = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    delta = [a-b for a,b in zip(moving['position'], fixed['position'])]
    return [ordered_sum(r[j][i]*delta[j] for j in range(3)) for i in range(3)]


def make_cases(state, selected, half_width=160.):
    cases, details = [], []
    for entry in selected:
        root, child = entry['members']
        order = sorted((i for i in range(len(state)) if i not in (root, child)),
                       key=lambda i: (ordered_sum((a-b)**2 for a,b in zip(state[root]['position'], state[i]['position'])), i))
        first = order[0]
        second = next(i for i in order if i not in entry['parent_component_members'] and i != first)
        for kind, anchor in [('nearest_nonmember', first), ('nearest_external_distinct', second)]:
            name = f"{'whole' if entry['whole_component'] else 'embedded'}_{root}_{child}_{kind}"
            cases.append(dict(name=name, root=root, child=child, anchor=anchor))
            edges = [relative_translation(state[anchor], state[root]), relative_translation(state[root], state[child])]
            extent = max(abs(v) for edge in edges for v in edge)
            require(extent <= half_width, f'Old edge outside fixed cube: {name}')
            details.append(dict(name=name, parent_component_members=entry['parent_component_members'],
                                old_edge_translations=edges, maximum_absolute_old_edge_coordinate=extent,
                                witness=entry['witnesses'][0]))
    return cases, details


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inputs = destination / 'panel-inputs'
    inputs.mkdir()  # Refuse to replace any prior preparation.
    bindings = {}
    paths = [(name, ARCHIVE/'input'/name) for name in ('frame7400.json', 'contact-anchor.json', 'freeze-manifest.json')]
    paths += [(f'events-p{i}.jsonl', ARCHIVE/f'contact-anchor-p{i}/events.jsonl') for i in range(4)]
    paths += [('prepare_dimer_destination_panel.py', Path(__file__).resolve())]
    for name, path in paths:
        digest = sha(path)
        if name in EXPECTED:
            require(digest == EXPECTED[name], f'Changed archived input: {path}')
        target = inputs/name
        shutil.copyfile(path, target)
        bindings[name] = dict(source_path=str(path), path=str(target), sha256=digest, bytes=target.stat().st_size)
    frame = json.loads((inputs/'frame7400.json').read_text())
    config = json.loads((inputs/'contact-anchor.json').read_text())
    require(frame['sweep'] == 7400 and frame['poses'] == config['initial_poses'], 'Saved source mismatch')
    streams = [(f'events-p{i}.jsonl', [json.loads(line) for line in (inputs/f'events-p{i}.jsonl').read_text().splitlines()]) for i in range(4)]
    table, first_events = collect_pairs(frame['poses'], streams)
    selected = select_pairs(table)
    cases, details = make_cases(frame['poses'], selected)
    atlas_bindings = []
    for name, path, digest in ATLAS_SOURCES:
        require(sha(path) == digest, f'Changed atlas: {path}')
        target = inputs/f'{name}.json'
        shutil.copyfile(path, target)
        model = json.loads(target.read_text()); base = model.get('base_model', model)
        atlas_bindings.append(dict(name=name, model=dict(path=str(target), sha256=digest),
                                   source_path=str(path), stored_gaussian_components=len(base['weights']),
                                   virtual_branches=len(base['weights'])+sum(model.get('reciprocal_components', []))))
    panel = dict(schema='dimer-destination-panel-v1', selection_rule=SELECTION_RULE,
                 source_bindings=bindings, atlases=atlas_bindings, sweep=frame['sweep'], bodies=len(frame['poses']),
                 uniform_half_width=160., cases=cases, case_details=details, selected_pairs=selected,
                 complete_eligible_pair_table=table, complete_first_event_table=first_events,
                 maximum_absolute_old_edge_coordinate=max(d['maximum_absolute_old_edge_coordinate'] for d in details))
    target = destination/'panel.json'
    with target.open('x') as out:
        json.dump(panel, out, indent=2, allow_nan=False);out.write('\n')
    return dict(path=str(target), sha256=sha(target), cases=len(cases), eligible_pairs=len(table),
                first_events=len(first_events), maximum_absolute_old_edge_coordinate=panel['maximum_absolute_old_edge_coordinate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
