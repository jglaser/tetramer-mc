#!/usr/bin/env python3
"""Archive selected original AB rows and descriptive two-cloud Poisson diagnostics.

No new samples are drawn and no weights are changed. Point selection biases these
diagnostics: plug-in overlap estimates are not independent confirmation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    source=root/'AB/assessment-streaming.json';assessment=json.loads(source.read_text())
    chosen=assessment['top_weights'][:2];records=[]
    for ranked in chosen:
        pop=root/'AB/runs'/ranked['replicate']
        manifest=json.loads((pop/'manifest.json').read_text())
        z=manifest['activity'];rate=z*manifest['lambda_ratio']
        sample_path=pop/'samples.jsonl';selected=None
        with sample_path.open() as handle:
            for line_index,line in enumerate(handle,1):
                row=json.loads(line)
                if row['draw']==ranked['draw']:
                    selected=row;break
        assert selected is not None
        for key in ['cloud_log_weights','cloud_overlap_counts','pose','q','log_boltzmann_mean']:
            assert selected[key]==ranked[key]
        counts=selected['cloud_overlap_counts'];n=len(counts);total=sum(counts)
        lower=selected['lower_volume'];overlap=lower+total/(n*rate)
        overlap_se=math.sqrt(total)/(n*rate)
        zc=z*overlap
        cloud_cv2=math.expm1(z*z*(overlap-lower)/rate)
        records.append({
            'original_row':selected,'replicate':ranked['replicate'],
            'sample_path':str(sample_path),'sample_line_1_based':line_index,
            'sample_sha256':sha(sample_path),
            'manifest_path':str(pop/'manifest.json'),'manifest_sha256':sha(pop/'manifest.json'),
            'original_mass_fraction':math.exp(selected['log_importance_weight']-
                assessment['regions']['native']['log_normalizer']-math.log(assessment['samples'])),
            'diagnostic_selected_cloud_overlap_A3':overlap,
            'diagnostic_plugin_Poisson_SE_overlap_A3':overlap_se,
            'diagnostic_selected_cloud_zC':zc,
            'diagnostic_plugin_Poisson_SE_zC':z*overlap_se,
            'original_logmean_minus_plugin_zC':selected['log_boltzmann_mean']-zc,
            'diagnostic_plugin_two_cloud_weight_relative_SD':math.sqrt(cloud_cv2/n),
            'diagnostic_cloud_count_difference_over_sqrt_sum':
                (counts[0]-counts[1])/math.sqrt(total) if total and n==2 else None})
    result={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'no_resampling_or_weight_replacement':True,
        'scope':'Selected original clouds only; plug-in Poisson diagnostics are descriptive and selection-biased, not independent fixed-pose confirmation or confidence intervals.',
        'formula':'K_i~Poisson(lambda*(C-L)); C_hat=L+sum(K)/(m*lambda); plugin SE(C_hat)=sqrt(sum(K))/(m*lambda); CV2(mean W)=expm1(z*z*(C_hat-L)/lambda)/m.',
        'assessment_sha256':sha(source),'records':records,
        'analysis_script_sha256':sha(Path(__file__))}
    (root/'dominant-AB-original-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
