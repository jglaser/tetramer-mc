"""Exact, bounded component embeddings for integer crystallographic operators.

This module consumes an already authenticated catalogue; it does not load or
certify one. An Affine acts as x -> R*x + t/d, where R is a proper signed
permutation and every operator in an embedding shares the positive denominator
``d``. Composition is left-after-right. No floating-point conversion, coordinate
rounding, lattice reduction, or numerical transform matching is performed.

Each native key (i, j, motif) has i < j and constrains G_j = G_i * M_motif.
Labels on one pair are alternatives. Every body occupies the monomer sites
G_i * S_member. Site identity is the FULL affine operation, appropriate only
for an asymmetric monomer with trivial stabilizer. Membership of the supplied
operators in the native space group and their alignment with the rigid body
are separate caller obligations.

Connected components are independently gauged at their smallest body index.
A successful certificate says nothing about relative placement or packing of
different components, physical pose fit, contact probability, or stability.
Components flagged as non-lifted periodic components remain unresolved; no
periodic quotient-space embedding is attempted.

The search budget counts attempted non-root body assignments, including failed
cycle/site tests. Gauge roots, input validation, candidate construction, and
witness serialization are not search nodes. Exhaustion preserves the work
counters and returns None, never a proof of impossibility. There is no cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=None):
    _require(type(value) is int, name + ' must be an integer (not bool)')
    _require(minimum is None or value >= minimum, name + ' is below its minimum')
    return value


def _sequence(value, name):
    # Do not consume arbitrary iterators: the admitted input is a finite record.
    _require(type(value) in (tuple, list), name + ' must be a finite tuple or list')
    return value


def _denominator(value):
    return _integer(value, 'denominator', 1)


@dataclass(frozen=True)
class Affine:
    """A canonical exact operator; translation stores integer numerators only."""

    rotation: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    translation_numerator: tuple[int, int, int]

    def __post_init__(self):
        rows = _sequence(self.rotation, 'rotation')
        _require(len(rows) == 3, 'rotation must have three rows')
        matrix = []
        for row in rows:
            row = _sequence(row, 'rotation row')
            _require(len(row) == 3, 'rotation rows must have three entries')
            matrix.append(tuple(_integer(x, 'rotation entry') for x in row))
        matrix = tuple(matrix)
        _require(all(x in (-1, 0, 1) for row in matrix for x in row),
                 'rotation entries must be -1, 0, or 1')
        _require(all(sum(x*x for x in row) == 1 for row in matrix)
                 and all(sum(matrix[i][j]**2 for i in range(3)) == 1 for j in range(3)),
                 'rotation must be a signed permutation')
        a, b, c = matrix
        determinant = (a[0]*(b[1]*c[2]-b[2]*c[1])
                       - a[1]*(b[0]*c[2]-b[2]*c[0])
                       + a[2]*(b[0]*c[1]-b[1]*c[0]))
        _require(determinant == 1, 'rotation must be proper (determinant +1)')
        translation = _sequence(self.translation_numerator, 'translation_numerator')
        _require(len(translation) == 3, 'translation_numerator must have three entries')
        translation = tuple(_integer(x, 'translation numerator') for x in translation)
        object.__setattr__(self, 'rotation', matrix)
        object.__setattr__(self, 'translation_numerator', translation)

    @classmethod
    def from_record(cls, record, *, denominator):
        """Read one strict JSON-style record and verify its common denominator."""
        denominator = _denominator(denominator)
        _require(type(record) is dict and set(record) ==
                 {'rotation', 'translation_numerator', 'denominator'},
                 'operator record requires exactly rotation, translation_numerator, denominator')
        _require(_denominator(record['denominator']) == denominator,
                 'operator record denominator differs from the declared common denominator')
        return cls(record['rotation'], record['translation_numerator'])

    def to_record(self, *, denominator):
        return dict(rotation=[list(row) for row in self.rotation],
                    translation_numerator=list(self.translation_numerator),
                    denominator=_denominator(denominator))


def _operator(value):
    _require(type(value) is Affine, 'expected an Affine operator')
    return value


def compose(left: Affine, right: Affine) -> Affine:
    """Return left after right; both translation vectors share one denominator."""
    left, right = _operator(left), _operator(right)
    a, b = left.rotation, right.rotation
    rotation = tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3))
                     for i in range(3))
    translation = tuple(left.translation_numerator[i]
                        + sum(a[i][k]*right.translation_numerator[k] for k in range(3))
                        for i in range(3))
    return Affine(rotation, translation)


def inverse(operator: Affine) -> Affine:
    operator = _operator(operator)
    rotation = tuple(tuple(operator.rotation[j][i] for j in range(3)) for i in range(3))
    translation = tuple(-sum(rotation[i][k]*operator.translation_numerator[k] for k in range(3))
                        for i in range(3))
    return Affine(rotation, translation)


IDENTITY = Affine(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
SCHEMA = 'exact-native-component-embedding-v1'
SCOPE = ('Exact catalogue-label and injective monomer-site feasibility, independently gauged per component. '
         'Full operator identity assumes an asymmetric monomer with trivial stabilizer. Catalogue native-group '
         'membership and rigid-body/member alignment are caller obligations. No measured-pose fit, '
         'cross-component packing, periodic quotient embedding, equilibrium, or stability claim.')


@dataclass(frozen=True, init=False)
class ExactNativeEmbedding:
    motif_operators: Mapping[int, Affine]
    member_operators: tuple[Affine, ...]
    denominator: int

    def __init__(self, motif_operators, member_operators, *, denominator):
        denominator = _denominator(denominator)
        _require(type(motif_operators) is dict, 'motif_operators must be a finite dictionary')
        motifs = {}
        for label, operator in motif_operators.items():
            motifs[_integer(label, 'motif ID', 0)] = _operator(operator)
        members = tuple(_operator(x) for x in _sequence(member_operators, 'member_operators'))
        _require(bool(members), 'at least one member operator is required')
        _require(len(set(members)) == len(members), 'duplicate member operator occupies the same monomer site')
        object.__setattr__(self, 'motif_operators', MappingProxyType(dict(sorted(motifs.items()))))
        object.__setattr__(self, 'member_operators', members)
        object.__setattr__(self, 'denominator', denominator)

    def check(self, body_count, native_keys, *, max_search_nodes, non_lifted_components=()):
        """Return a JSON-friendly certificate, with None for unresolved components.

        ``native_keys`` is a finite list/tuple of canonical (i,j,motif) triples.
        Exact duplicate triples are removed. ``non_lifted_components`` lists
        complete connected-component body sets, not individual winding edges.
        A zero budget is valid: isolated bodies are still certifiable without
        a search, while a component needing a candidate assignment is unknown.
        """
        body_count = _integer(body_count, 'body_count', 1)
        cap = _integer(max_search_nodes, 'max_search_nodes', 0)
        keys = []
        for key in _sequence(native_keys, 'native_keys'):
            key = _sequence(key, 'native key')
            _require(len(key) == 3, 'native key must contain i, j, motif ID')
            i, j, label = (_integer(x, 'native key entry', 0) for x in key)
            _require(i < j < body_count, 'native pair must satisfy 0 <= i < j < body_count')
            _require(label in self.motif_operators, 'native key names an unknown motif')
            keys.append((i, j, label))
        canonical = tuple(sorted(set(keys)))
        pairs = {}
        adjacency = [set() for _ in range(body_count)]
        for i, j, label in canonical:
            pairs.setdefault((i, j), []).append(label)
            adjacency[i].add(j)
            adjacency[j].add(i)
        components, unseen = [], set(range(body_count))
        while unseen:
            root = min(unseen)
            pending, reached = [root], set()
            while pending:
                body = pending.pop()
                if body in reached:
                    continue
                reached.add(body)
                pending.extend(sorted(adjacency[body]-reached, reverse=True))
            components.append(tuple(sorted(reached)))
            unseen.difference_update(reached)
        non_lifted = set()
        for component in _sequence(non_lifted_components, 'non_lifted_components'):
            indices = tuple(_integer(i, 'non-lifted body index', 0)
                            for i in _sequence(component, 'non-lifted component'))
            _require(indices and len(set(indices)) == len(indices),
                     'non-lifted component must contain distinct bodies')
            indices = tuple(sorted(indices))
            _require(indices in components, 'non-lifted set is not a complete native component')
            _require(indices not in non_lifted, 'duplicate non-lifted component')
            non_lifted.add(indices)
        used, results = 0, []
        for component in components:
            component_pairs = {pair: tuple(labels) for pair, labels in pairs.items() if pair[0] in component}
            result, used = self._component(component, component_pairs, used, cap, component in non_lifted)
            results.append(result)
        states = [component['embeddable'] for component in results]
        componentwise = False if False in states else (None if None in states else True)
        return dict(schema=SCHEMA, denominator=self.denominator, body_count=body_count,
                    member_count=len(self.member_operators), input_key_count=len(keys),
                    unique_native_keys=[list(key) for key in canonical], duplicate_key_count=len(keys)-len(canonical),
                    max_search_nodes=cap, search_nodes=used,
                    budget_exhausted=any(c['status'] == 'search_budget_exhausted' for c in results),
                    componentwise_embeddable=componentwise, components=results, scope=SCOPE)

    def _component(self, bodies, pairs, used, cap, non_lifted):
        start = used
        diagnostics = dict(cycle_constraint_rejections=0, site_collision_rejections=0,
                           first_cycle_rejection=None, first_site_collision=None)
        result = dict(bodies=list(bodies), root_body=bodies[0], pair_count=len(pairs),
                      label_alternatives=[dict(pair=list(pair), motif_ids=list(labels)) for pair, labels in pairs.items()],
                      embeddable=None, status=None, search_node_start=start, search_node_end=start,
                      search_nodes=0, diagnostics=diagnostics, witness=None)

        def finish(status, embeddable, witness=None):
            result.update(status=status, embeddable=embeddable, search_node_end=used,
                          search_nodes=used-start, witness=witness)
            return result, used

        if non_lifted:
            return finish('non_lifted_periodic_component', None)
        assigned = {bodies[0]: IDENTITY}
        occupied = {compose(IDENTITY, member): (bodies[0], k)
                    for k, member in enumerate(self.member_operators)}
        # The stack holds one finite candidate iterator per current frontier.
        # All assignments/collisions below use exact Affine keys. No recursion
        # limit or retained search cache can change a bounded search verdict.
        stack = []

        def frame():
            pair = next(pair for pair in pairs if (pair[0] in assigned) != (pair[1] in assigned))
            i, j = pair
            target = j if i in assigned else i
            candidates, seen = [], set()
            for label in pairs[pair]:
                motif = self.motif_operators[label]
                operator = (compose(assigned[i], motif) if target == j
                            else compose(assigned[j], inverse(motif)))
                # Distinct aliases yielding the same operator need only one
                # branch. The final witness independently chooses a valid label.
                if operator not in seen:
                    seen.add(operator)
                    candidates.append((label, operator))
            return dict(body=target, pair=pair, candidates=candidates, next_index=0, occupied=None)

        while True:
            if len(assigned) == len(bodies):
                chosen = []
                for (i, j), labels in pairs.items():
                    relative = compose(inverse(assigned[i]), assigned[j])
                    chosen.append([i, j, next(label for label in labels if self.motif_operators[label] == relative)])
                witness = dict(body_operators=[dict(body=i, operator=assigned[i].to_record(denominator=self.denominator))
                                               for i in sorted(assigned)],
                               chosen_labels=chosen,
                               occupied_sites=[dict(body=i, member_index=k,
                                                    operator=compose(assigned[i], member).to_record(denominator=self.denominator))
                                               for i in sorted(assigned)
                                               for k, member in enumerate(self.member_operators)])
                return finish('embedding_found', True, witness)
            stack.append(frame())
            # Advance/backtrack until a candidate survives. Budget is checked
            # only when there really is a further candidate to attempt.
            while stack:
                current = stack[-1]
                body = current['body']
                if current['occupied'] is not None:
                    for site in current['occupied']:
                        del occupied[site]
                    del assigned[body]
                    current['occupied'] = None
                if current['next_index'] == len(current['candidates']):
                    stack.pop()
                    continue
                if used == cap:
                    return finish('search_budget_exhausted', None)
                label, operator = current['candidates'][current['next_index']]
                current['next_index'] += 1
                used += 1
                assigned[body] = operator
                failed_pair = None
                for pair, labels in pairs.items():
                    i, j = pair
                    if body in pair and i in assigned and j in assigned:
                        relative = compose(inverse(assigned[i]), assigned[j])
                        if not any(self.motif_operators[m] == relative for m in labels):
                            failed_pair = pair
                            break
                if failed_pair is not None:
                    diagnostics['cycle_constraint_rejections'] += 1
                    if diagnostics['first_cycle_rejection'] is None:
                        diagnostics['first_cycle_rejection'] = dict(candidate_body=body,
                            generating_pair=list(current['pair']), generating_motif_id=label,
                            conflicting_pair=list(failed_pair),
                            candidate_operator=operator.to_record(denominator=self.denominator))
                    del assigned[body]
                    continue
                sites = [compose(operator, member) for member in self.member_operators]
                collision = next(((k, site, occupied[site]) for k, site in enumerate(sites) if site in occupied), None)
                if collision is not None:
                    diagnostics['site_collision_rejections'] += 1
                    if diagnostics['first_site_collision'] is None:
                        member, site, existing = collision
                        diagnostics['first_site_collision'] = dict(candidate_body=body, candidate_member_index=member,
                            generating_pair=list(current['pair']), generating_motif_id=label,
                            occupied_by_body=existing[0], occupied_by_member_index=existing[1],
                            operator=site.to_record(denominator=self.denominator))
                    del assigned[body]
                    continue
                for k, site in enumerate(sites):
                    occupied[site] = (body, k)
                current['occupied'] = sites
                break
            else:
                return finish('no_admissible_embedding', False)
