import copy
import json
from pathlib import Path
import tempfile
import unittest
from bind_native_class_line_probe import validate_evidence
from run_native_class_line_probe import sha


class BindingTest(unittest.TestCase):
    def fixture(self, root):
        executable = root/'inert-binary'; executable.write_text('never executed')
        rust_source = root/'lib.rs'; rust_source.write_text('toy source')
        names = ['native_class_line_reference.py', 'hard_free_line_reference.py',
                 'analyze_contact_line_audit.py', 'native_contact_regions.py']
        references = {}
        for name in names:
            path = root/name; path.write_text('# never imported\n'+name)
            references[str(path)] = sha(path)
        data = dict(rust=dict(complete=True, passed=True, executable_sha256=sha(executable),
            source_closure_before={'lib.rs':sha(rust_source)}, source_closure_after={'lib.rs':sha(rust_source)},
            test_count=1, exit_code=0), python=dict(complete=True, passed=True, source_sha256=references),
            cli=dict(complete=True, passed=True, executable_sha256=sha(executable), reference_source_sha256=references))
        paths = {}
        for role, value in data.items():
            paths[role] = root/(role+'.json'); paths[role].write_text(json.dumps(value))
        return executable, paths, data

    def test_distinct_roles_source_and_binary_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); executable, paths, data=self.fixture(root)
            self.assertEqual(len(validate_evidence(executable, paths, root)), 3)
            duplicate = dict(paths, python=paths['rust'], cli=paths['rust'])
            with self.assertRaisesRegex(ValueError, 'distinct receipts'): validate_evidence(executable, duplicate, root)
            changed=copy.deepcopy(data['cli']); changed['executable_sha256']='0'*64
            paths['cli'].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'different executable'): validate_evidence(executable, paths, root)

    def test_changed_source_and_incomplete_closure_rejected(self):
        for kind in ('rust', 'python', 'missing'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root=Path(d); executable, paths, data=self.fixture(root)
                if kind=='rust': (root/'lib.rs').write_text('changed')
                elif kind=='python': (root/'native_class_line_reference.py').write_text('changed')
                else:
                    del data['cli']['reference_source_sha256'][str(root/'native_contact_regions.py')]
                    paths['cli'].write_text(json.dumps(data['cli']))
                with self.assertRaises(ValueError): validate_evidence(executable, paths, root)


if __name__=='__main__': unittest.main()
