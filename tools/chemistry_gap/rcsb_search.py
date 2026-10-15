"""Search the RCSB for hen egg-white lysozyme in the tetragonal form (P 43 21 2): high-resolution
X-ray (<= 1.2 A) and neutron (or joint X-ray/neutron) entries, with resolution, method,
and the non-water ligands present (ions)."""
import json
import urllib.request


def search(extra):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entity_source_organism.ncbi_scientific_name", "operator": "exact_match",
            "value": "Gallus gallus"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity.pdbx_description", "operator": "contains_phrase", "value": "lysozyme"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "symmetry.space_group_name_H_M", "operator": "exact_match", "value": "P 43 21 2"}}] + extra},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 40},
                            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]}}
    req = urllib.request.Request('https://search.rcsb.org/rcsbsearch/v2/query', data=json.dumps(q).encode(),
                                 headers={'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d.get('total_count', 0), [r['identifier'] for r in d.get('result_set', [])]


def info(pid):
    d = json.load(urllib.request.urlopen(f'https://data.rcsb.org/rest/v1/core/entry/{pid}', timeout=60))
    e = d.get('rcsb_entry_info', {})
    methods = [x.get('method') for x in d.get('exptl', [])]
    ligs = e.get('nonpolymer_bound_components') or []
    return dict(id=pid, res=e.get('resolution_combined'), methods=methods, ligands=ligs,
                title=d.get('struct', {}).get('title', '')[:90])


xray = search([{"type": "terminal", "service": "text", "parameters": {
    "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 1.2}}])
neutron = search([{"type": "terminal", "service": "text", "parameters": {
    "attribute": "exptl.method", "operator": "exact_match", "value": "NEUTRON DIFFRACTION"}}])
print('high-resolution X-ray (<= 1.2 A):', xray[0])
for pid in xray[1][:15]:
    print('  ', info(pid))
print('neutron:', neutron[0])
for pid in neutron[1][:15]:
    print('  ', info(pid))
