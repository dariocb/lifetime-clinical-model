"""Execute disposable fast-mode notebook copies on matched simple/rich cohorts.

No source notebook or existing training dataset is overwritten. Figures retain
normal notebook settings; only dataset paths, FAST_MODE and artifact paths change.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
from nbclient import NotebookClient
from lifetime_clinical.data import load_dataset


def prepare_subset(source, destination, splits):
    dataset=load_dataset(source)
    ids=set(sum(splits.values(),[]))
    for name in ('patients','events','pre_study_events','measurements','dynamic_covariates'):
        frame=getattr(dataset,name)
        setattr(dataset,name,frame[frame.patient_id.isin(ids)].copy().reset_index(drop=True))
    dataset.save(destination)
    (destination/'splits.json').write_text(json.dumps(splits,indent=2))


def metric_cell(relative, destination):
    prefix='''import json
from pathlib import Path
benchmark_metrics={'groups':{}}
'''
    if relative.name=='00_baselines.ipynb':
        body='''benchmark_metrics['groups']['baseline_timing']=baseline_nll_comparison.iloc[:,0].to_dict()
benchmark_metrics['groups']['baseline_marked']=baseline_nll_comparison.iloc[:,1].to_dict()
benchmark_metrics['heldout_events']=sum(int(np.sum(s.times/YEAR_DAYS>cutoffs[s.patient_id])) for s in data['test'])
benchmark_metrics['test_patients']=len(data['test'])
'''
    elif relative.name=='01_tpp_models.ipynb':
        body='''benchmark_metrics['groups']['tpp_event_mark']={name:float(model.conditional_nll(dynamic_data['test'])) for name,model in [('TPP-B',tpp_b),('TPP-C',tpp_c)]}
benchmark_metrics['groups']['tpp_joint']={name:float(model.joint_nll(dynamic_data['test'])) for name,model in [('TPP-B',tpp_b),('TPP-C',tpp_c)]}
benchmark_metrics['test_patients']=len(dynamic_data['test'])
benchmark_metrics['test_events']=sum(len(s.times) for s in dynamic_data['test'])
benchmark_metrics['feature_snapshots']=sum(len(s.contexts) for s in dynamic_data['test'])
'''
    elif relative.name in ('02_gp_models_1_2.ipynb','03_gp_model_3.ipynb'):
        body="gp_models=[('GP-A',gp_a),('GP-B',gp_b)]\n" if relative.name.startswith('02') else "gp_models=[('GP-C',gp_c)]\n"
        body+='''from lifetime_clinical.data.longitudinal import split_forecast_block
forecast_losses={name:[] for name,_ in gp_models}
forecast_counts=[]
forecast_ids=[]
for block in blocks['test']:
    if len(np.unique(block.ages))<2: continue
    past,future=split_forecast_block(block,.6)
    forecast_counts.append(len(future.y));forecast_ids.append(block.patient_id)
    for name,model in gp_models:
        prediction=model.forecast(past,future.ages,future.variables)
        # Same per-query predictive target; GP-C uses 40-node quadrature,
        # avoiding comparison of its ELBO with exact GP-A/B likelihoods.
        forecast_losses[name].append(float(-prediction.marginal_log_prob(future.raw_y).sum()))
benchmark_metrics['groups']['gp_heldout_marginal_per_value']={name:sum(loss)/sum(forecast_counts) for name,loss in forecast_losses.items()}
benchmark_metrics['forecast_patient_ids']=forecast_ids
benchmark_metrics['forecast_values']=sum(forecast_counts)
if len(gp_models)==2:
    benchmark_metrics['groups']['gp_exact_block']={name:float(model.score(blocks['test'])) for name,model in gp_models}
    benchmark_metrics['block_values']=sum(len(b.y) for b in blocks['test'])
    benchmark_metrics['test_blocks']=len(blocks['test'])
'''
    elif relative.parent.name=='lifetime_model':
        body="benchmark_metrics['groups']['lifetime_objectives']=lifetime.score(patients['test'])\nbenchmark_metrics['coverage_example']=float(coverage.mean())\n"
    else:return None
    return nbformat.v4.new_code_cell(prefix+body+f"Path({str(destination)!r}).write_text(json.dumps(benchmark_metrics,indent=2))\nprint('Benchmark metrics saved')")


def create_copies(output):
    splits=json.loads((ROOT/'data/toy/splits.json').read_text())
    jobs=[]
    for label,full in [('simple',ROOT/'data/synthetic'),('rich',ROOT/'data/generated_synthetic')]:
        base=output/label
        subset=base/'data/toy'
        prepare_subset(full,subset,splits)
        # Exploratory notebooks need only schemas/timelines, not full split counts.
        # Map their load_splits call to the temporary subset split file.
        shutil.copytree(ROOT/'notebooks',base/'notebooks',dirs_exist_ok=True)
        (base/'metrics').mkdir(exist_ok=True)
        for path in sorted((base/'notebooks').rglob('*.ipynb')):
            relative=path.relative_to(base/'notebooks')
            notebook=nbformat.read(path,4)
            for cell in notebook.cells:
                if cell.cell_type!='code':continue
                cell.outputs=[];cell.execution_count=None
                source=cell.source
                source=re.sub(r'FAST_MODE\s*=\s*(False|True)','FAST_MODE=True',source)
                source=re.sub(r'''ROOT\s*/\s*['"]data['"]\s*/\s*\(['"]toy['"]\s+if\s+FAST_MODE\s+else\s+['"]synthetic['"]\)''',f'Path({subset.as_posix()!r})',source)
                # Exploratory data/introduction notebooks have no training mode;
                # use the corresponding full cohort for their illustrative selection.
                source=re.sub(r'''ROOT\s*/\s*['"]data['"]\s*/\s*['"]synthetic['"]''',f'Path({full.as_posix()!r})',source)
                source=re.sub(r'''ROOT\s*/\s*['"]data['"]\s*/\s*['"]toy['"]''',f'Path({subset.as_posix()!r})',source)
                source=re.sub(r'''ROOT\s*/\s*['"]artifacts['"]''',f'Path({(base/"artifacts").as_posix()!r})',source)
                if relative.parent.name=='data':
                    source=source.replace('splits=load_splits(DATA)',f'splits=load_splits(Path({subset.as_posix()!r}))')
                cell.source=source
            metric=metric_cell(relative,base/'metrics'/f'{path.stem}.json')
            if metric is not None:
                notebook.cells.extend([nbformat.v4.new_markdown_cell(
                    'Temporary benchmark: score the same targets within each applicable model group. These cells do not modify the source notebook.'),metric])
            notebook.metadata['synthetic_benchmark']={'scenario':label,'training_patients':len(splits['train']),'epochs':5,'source':str(ROOT/'notebooks'/relative)}
            nbformat.write(notebook,path)
            jobs.append((label,path))
    return jobs


def execute(job):
    label,path=job
    start=time.monotonic()
    print(f'START {label}/{path.name}',flush=True)
    notebook=nbformat.read(path,4)
    error=None
    try:
        NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        nbformat.write(notebook,path)
    result={'scenario':label,'notebook':str(path),'seconds':round(time.monotonic()-start,1),'success':error is None,'error':error}
    print(f'{"DONE" if error is None else "FAILED"} {label}/{path.name} ({result["seconds"]}s)',flush=True)
    return result



def matched_visit_scores(output):
    """Extra common-window cohort comparisons, expressed as day-based NLLs.

    Full-entry group: baseline cohort MLPs versus TPP-A.
    Complete-physiology group: refit baseline MLPs on precisely the same
    trimmed train/validation sequences as TPP-B/C, then score the same tests.
    Per-patient prefix fits are intentionally absent from these groups.
    """
    import numpy as np
    import torch
    from lifetime_clinical.tutorials.experiments import load_toy,tpp_data,fit_tpp_a
    from lifetime_clinical.models.tpp.classical import CovariatePointProcess,process_nll
    from lifetime_clinical.data.synthetic import YEAR_DAYS
    torch.set_num_threads(1)
    for label in ('simple','rich'):
        base=output/label
        dataset,splits,pre=load_toy(base/'data/toy')
        result={'groups':{},'units':'Mean nats/patient; event densities measured per day, marks included.',
                'protocol':'Cohort models fit the same train/validation windows within each group; hidden excitation is reset at the selected origin.'}
        for dynamic in (False,True):
            data=tpp_data(dataset,splits,pre,dynamic=dynamic)
            counts=np.bincount(np.concatenate([seq.marks for seq in data['train']]),minlength=3)+.5
            pi=counts/counts.sum()
            scores={}
            for kind,name in [('poisson','Poisson MLP'),('hawkes','Hawkes MLP'),('marked_hawkes','Marked Hawkes MLP')]:
                model=CovariatePointProcess(len(data['train'][0].static),kind).fit(
                    data['train'],data['validation'],epochs=5,device='cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    values=[]
                    for seq in data['test']:
                        value=float(process_nll(seq,*model.parameters_for(seq)))
                        if model.n_marks==1:value-=np.log(pi[seq.marks]).sum()
                        # Change time-density units: f_day=f_year/YEAR_DAYS.
                        value+=len(seq.times)*np.log(YEAR_DAYS)
                        values.append(value)
                scores[name]=float(np.mean(values))
            if dynamic:
                other=json.loads((base/'metrics/01_tpp_models.json').read_text())
                scores.update(other['groups']['tpp_event_mark'])
                group='visits_complete_physiology'
            else:
                model=fit_tpp_a(data,epochs=5,device='cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():scores['TPP-A']=float(model.conditional_nll(data['test']))
                group='visits_from_entry'
            result['groups'][group]=scores
            result[group+'_patients']=len(data['test'])
            result[group+'_events']=sum(len(seq.times) for seq in data['test'])
        (base/'metrics/04_matched_visit_comparison.json').write_text(json.dumps(result,indent=2))


def summarize(output,status):
    metrics={label:{} for label in ('simple','rich')}
    for label in metrics:
        for path in (output/label/'metrics').glob('*.json'):
            metrics[label][path.stem]=json.loads(path.read_text())
    split_counts={k:len(v) for k,v in json.loads((output/'rich/data/toy/splits.json').read_text()).items()}
    total=sum(split_counts.values())
    split_label='/'.join(str(split_counts[k]) for k in ('train','validation','test'))
    report={'controls':{'seed':42,'epochs':5,'patients':total,'splits':split_counts,
                        'limitations':'One seed and a small test cohort; different observations/windows across scenarios. No GP-C ELBO versus exact-NLL comparison.'},
            'runs':status,'metrics':metrics}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Fast synthetic-data benchmark','',
           f'Matched patient IDs, {split_label} split, model seed 42, five epochs. All source notebooks are unchanged.',
           f'Exploratory notebooks use full cohorts; fitted models use {total}-patient subsets.', '',
           'Scores compare models only within matched-target groups. Cross-scenario gaps are exploratory; a small test cohort and one seed cannot establish superiority.','',
           '| Scenario | Notebook | Successful | Seconds |','|---|---|---|---|']
    for run in status:lines.append(f'| {run["scenario"]} | {Path(run["notebook"]).name} | {run["success"]} | {run["seconds"]} |')
    for label,notebooks in metrics.items():
        lines.extend(['',f'## {label}',''])
        for name,values in notebooks.items():
            lines.extend([f'### {name}',''])
            for group,scores in values['groups'].items():
                lines.extend([f'{group}:',''])
                for model,score in scores.items():lines.append(f'- {model}: {score:.6f}')
                lines.append('')
    # Differences within groups cancel shared density-unit constants.
    gains=[]
    for label in ('simple','rich'):
        groups=metrics[label]['04_matched_visit_comparison']
        static=groups['groups']['visits_from_entry']
        dynamic=groups['groups']['visits_complete_physiology']
        gp=metrics[label]['02_gp_models_1_2']['groups']['gp_heldout_marginal_per_value']
        c=metrics[label]['03_gp_model_3']['groups']['gp_heldout_marginal_per_value']['GP-C']
        assert metrics[label]['02_gp_models_1_2']['forecast_patient_ids']==metrics[label]['03_gp_model_3']['forecast_patient_ids']
        gains.append({'TPP-A versus Poisson MLP':(static['Poisson MLP']-static['TPP-A'])*groups['visits_from_entry_patients']/groups['visits_from_entry_events'],
                      'TPP-C versus TPP-B':(dynamic['TPP-B']-dynamic['TPP-C'])*groups['visits_complete_physiology_patients']/groups['visits_complete_physiology_events'],
                      'GP-B versus GP-A':gp['GP-A']-gp['GP-B'],
                      'GP-C versus GP-A':gp['GP-A']-c})
    increased=[gains[1][key]>gains[0][key] for key in gains[0]]
    conclusion=('All four measured advantages increase on richer data in this run.' if all(increased) else 'The richer data does not consistently increase the complex models\' advantage in this run.')
    summary=['# Does richer synthetic data separate the models?', '',
             conclusion, '',
             'Positive values below mean the first model has lower NLL. Visit rows include marks and censoring, normalised per observed visit; GP rows use the same held-out predictive marginal targets, normalised per physiological value. Different rows are different targets and must not be ranked against each other.', '',
             '| Comparison | Original simple data | Richer data | Units |',
             '|---|---:|---:|---|']
    for name in gains[0]:
        units='NLL gain / visit' if name.startswith('TPP') else 'NLL gain / value'
        summary.append(f'| {name} | {gains[0][name]:+.4f} | {gains[1][name]:+.4f} | {units} |')
    summary.extend(['', f'{split_counts["train"]} training, {split_counts["validation"]} validation and {split_counts["test"]} test patients, one model seed and five neural-training epochs were used. Local Poisson fits are analytic; local Hawkes fits use constrained maximum likelihood. The scenarios have different observed event/measurement counts, so even normalised gap changes are exploratory.', '',
                    'The richer generator is not designed to reproduce Hawkes excitation exactly, and the small MLP baselines keep covariates fixed at baseline. Limited training and one seed cannot establish that the flexible models have learned the new mechanisms. A full-data, multi-seed comparison is needed before claiming superiority.', '',
                    'GP-C is compared using 40-node quadrature for predictive marginal densities, never its negative ELBO against exact GP-A/B NLL. Lifetime objectives remain separate. Cohort visit comparisons use matched training/evaluation windows and convert baseline event densities from years to days.', '',
                    'All 20 copied notebooks executed without errors. The toy cohort was prepared separately; benchmark execution used disposable copies. See README.md and report.json for all scores and execution details.'])
    (output/'comparison_summary.md').write_text('\n'.join(summary)+'\n')
    lines.extend(['','## Interpretation','',*summary[2:]])
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'tmp'/('fast_synthetic_benchmark_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=create_copies(output)
    print('Output:',output,flush=True)
    status=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            status.append(future.result())
            (output/'run_status.json').write_text(json.dumps(status,indent=2))
    if all(r['success'] for r in status): matched_visit_scores(output)
    summarize(output,status)
    print(f'Completed {sum(r["success"] for r in status)}/{len(status)} notebooks. Report: {output/"README.md"}',flush=True)
    return 0 if all(r['success'] for r in status) else 1

if __name__=='__main__':raise SystemExit(main())
