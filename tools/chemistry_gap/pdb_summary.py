"""Summary of downloaded lysozyme structures (pdb-hewl/*.pdb, read as data): space group, cell,
resolution, protein atom count (incl. H/D), alternate conformations, HETATM residues (waters,
ions), and whether waters carry hydrogens/deuteriums. Usage: pdb_summary.py FILE..."""
import sys
from collections import Counter

for f in sys.argv[1:]:
    lines = open(f).read().splitlines()
    cryst = next((l for l in lines if l.startswith('CRYST1')), '')
    res = next((l for l in lines if l.startswith('REMARK   2 RESOLUTION')), '').strip()
    atoms = [l for l in lines if l.startswith('ATOM')]
    het = [l for l in lines if l.startswith('HETATM')]
    elem = Counter(l[76:78].strip() for l in atoms)
    alt = sum(1 for l in atoms if l[16] not in ' A')
    hres = Counter(l[17:20] for l in het)
    welem = Counter(l[76:78].strip() for l in het if l[17:20] in ('HOH', 'DOD'))
    nwat = len({(l[17:20], l[21], l[22:27]) for l in het if l[17:20] in ('HOH', 'DOD')})
    print(f"{f.split('/')[-1]}: {cryst[55:66].strip()} cell {cryst[6:33].split()} | {res}")
    print(f"   protein atoms {len(atoms)} {dict(elem)}; altloc atoms {alt}; waters {nwat} (atoms {dict(welem)})")
    print(f"   HETATM residues {dict(hres)}")
