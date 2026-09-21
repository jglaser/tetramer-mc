"""Archive and dispatch checks with a fake executable, never physical sampling."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FrozenGuideCampaignTests(unittest.TestCase):
    def run_campaign(self, guided):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            shape=root/'shape.json';shape.write_text('{"atoms":[]}\n')
            config=root/'input.json';config.write_text(json.dumps(dict(shape='shape.json',reservoir_density=.035)))
            region=root/'region.json';region.write_text('{"fixture":"fixed region"}\n')
            guide=root/'source-guide.json';guide.write_text('{"fixture":"immutable guide"}\n')
            # The mock records invocation and exits; it does not produce estimates.
            binary=root/'fake-normalizer'
            binary.write_text('#!'+sys.executable+'\n'+
                'import json, pathlib, sys\n'+
                'args=sys.argv[1:]; out=pathlib.Path(args[args.index("--out")+1])\n'+
                'out.mkdir(parents=True); (out/"argv.json").write_text(json.dumps(args))\n')
            binary.chmod(0o755)
            out=root/'campaign'
            command=[sys.executable,'-B',str(Path(__file__).with_name('run_latent_region_campaign.py')),
                '--out',str(out),'--config',str(config),'--region',str(region),'--binary',str(binary),
                '--samples','11','--replicates','2','--workers','2','--seed','9191001']
            if guided:command+=['--importance-guide',str(guide)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((out/'manifest.json').read_text())
            archive=out/'provenance'
            self.assertEqual((archive/'region.json').read_bytes(),region.read_bytes())
            self.assertEqual(json.loads((archive/'config.json').read_text())['shape'],str(archive/'shape.json'))
            self.assertEqual([j['seed'] for j in manifest['jobs']],[9191001,9192010])
            for job in manifest['jobs']:
                argv=json.loads((Path(job['directory'])/'argv.json').read_text())
                self.assertEqual(argv[argv.index('--samples')+1],'11')
                self.assertEqual(argv[argv.index('--region')+1],str(archive/'region.json'))
                self.assertEqual(json.loads((out/(job['id']+'-status.json')).read_text())['returncode'],0)
                if guided:self.assertEqual(argv[argv.index('--importance-guide')+1],str(archive/'importance-guide.json'))
                else:self.assertNotIn('--importance-guide',argv)
            if guided:
                self.assertEqual(manifest['schema'],'importance-latent-region-campaign-v1')
                self.assertEqual((archive/'importance-guide.json').read_bytes(),guide.read_bytes())
                digest=hashlib.sha256(guide.read_bytes()).hexdigest()
                self.assertEqual(manifest['importance_guide_sha256'],digest)
                self.assertEqual(manifest['archive_sha256']['importance-guide.json'],digest)
                self.assertEqual(manifest['source_inputs']['importance-guide.json'],str(guide))
            else:
                self.assertEqual(manifest['schema'],'uniform-latent-region-campaign-v1')
                self.assertNotIn('importance_guide_sha256',manifest)
                self.assertNotIn('importance-guide.json',manifest['archive_sha256'])
                self.assertFalse((archive/'importance-guide.json').exists())

    def test_frozen_guide_is_archived_bound_and_dispatched(self):
        self.run_campaign(True)

    def test_no_guide_keeps_legacy_manifest_and_command(self):
        self.run_campaign(False)


if __name__=='__main__':unittest.main()
