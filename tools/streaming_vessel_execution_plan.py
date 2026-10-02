"""Pure scheduling adapter for the already frozen streaming-vessel allocation.

This module neither reads result files nor launches processes or writes files.
Its hashes are declared bindings to be authenticated by a future dispatcher,
not certificates that the referenced bytes or scientific prerequisites passed.
The original preparation remains inert and dispatch_ready=False. Existing
streaming preparation and population/stage readers own their validation laws;
this module only arranges unchanged job commands and the pinned stage reader.
"""
from __future__ import annotations
import copy
from pathlib import Path
import re

from analyze_r4_smc_control import require
from prepare_streaming_vessel_comparison import validate_contract, STAGES, THREADS

SCHEMA = 'inert-streaming-vessel-execution-plan-v1'
READER = 'analyze_streaming_vessel_stage.py'
GROUP_WORKERS = {'physical':8, 'audit':4, 'partition':4, 'aggregate':1}
BUDGETS = dict(maximum_physical_workers=8, maximum_audit_workers=4, maximum_all_workers=32)


def _digest(value, label):
    require(isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None,
            'Explicit SHA-256 required for '+label)
    return value


def _paths(out, preparation):
    out,preparation = Path(out).resolve(),Path(preparation).resolve()
    require(out != preparation and not out.is_relative_to(preparation)
            and not preparation.is_relative_to(out),
            'Execution-plan directory must be separate from the frozen preparation')
    return out,preparation


def stages_for(out, preparation, plan, preparation_sha256, reader_sha256):
    """Return two fixed stages, each with physical/audit/partition/reader groups.

    Every existing job command and destination is copied verbatim. The reader
    is to be frozen separately in out/common, with its entire source closure;
    this declaration pins its entry point but does not archive or authenticate
    it. The future dispatcher must use the existing stage reader's validators.
    """
    out,preparation = _paths(out,preparation)
    _digest(preparation_sha256,'preparation'); _digest(reader_sha256,'frozen reader')
    validate_contract(plan,preparation)
    python = plan['runtime']['python']
    require(isinstance(python,str) and Path(python).is_absolute(),'Pinned Python executable must be absolute')
    require(all(job['audit_command'][0] == python and job['partition_command'][0] == python
                for job in plan['jobs']), 'Reader and population Python executables differ')
    stages = []
    for stage,_ in STAGES:
        jobs = [job for job in plan['jobs'] if job['stage'] == stage]
        groups = []
        for kind in ('physical','audit','partition'):
            steps = []
            for job in jobs:
                steps.append(dict(id=job['id']+'-'+kind,job_id=job['id'],kind=kind,
                    directory=job['directory' if kind=='physical' else kind+'_directory'],
                    command=copy.deepcopy(job['command' if kind=='physical' else kind+'_command']),
                    log=job['log'] if kind=='physical' else str(out/'logs'/(job['id']+'-'+kind+'.log'))))
            groups.append(dict(kind=kind,workers=GROUP_WORKERS[kind],steps=steps))
        reader = out/'common'/READER
        destination = out/(stage+'-comparison')
        step = dict(id=stage+'-aggregate',kind='aggregate',directory=str(destination),
            log=str(out/'logs'/(stage+'-aggregate.log')),reader_sha256=reader_sha256,
            command=[python,'-B',str(reader),'--preparation',str(preparation),
                     '--preparation-sha256',preparation_sha256,'--stage',stage,'--out',str(destination)])
        groups.append(dict(kind='aggregate',workers=GROUP_WORKERS['aggregate'],steps=[step]))
        stages.append(dict(name=stage,groups=groups))
    return stages


def execution_plan(out, preparation, plan, preparation_sha256, reader_sha256):
    """Build declarations only; no artifact authentication or gate authority."""
    out,preparation = _paths(out,preparation)
    stages = stages_for(out,preparation,plan,preparation_sha256,reader_sha256)
    return dict(schema=SCHEMA,preparation=dict(path=str(preparation),sha256=preparation_sha256),
        reader=dict(path=str(out/'common'/READER),sha256=reader_sha256),
        budgets=copy.deepcopy(BUDGETS),thread_environment=copy.deepcopy(THREADS),
        stages=stages,physical_jobs=16,total_unconditional_draws=plan['total_unconditional_draws'],
        dispatch_ready=False,gate_authority=False,launch_entry_point=False,
        stage_policy='Complete standard physics, audits, partitions and aggregate before large. '
            'Retain both fixed allocations regardless of standard statistical diagnostics; no pooling or adaptation.',
        scope='Scheduling declaration only. Hashes require future authentication; regional prerequisites remain '
            'unresolved. No processes, files, geometry, classification or physical estimates are produced.')


def validate(value, out, preparation, plan, preparation_sha256, reader_sha256):
    """Reject any scheduling, command, budget, source-pin or authority mutation."""
    expected = execution_plan(out,preparation,plan,preparation_sha256,reader_sha256)
    require(value == expected,'Streaming execution plan differs from the fixed scheduling contract')
    return value
