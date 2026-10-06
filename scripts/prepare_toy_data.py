"""Prepare a deterministic rich toy cohort, preserving existing split memberships."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lifetime_clinical.data import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/generated_synthetic')
    parser.add_argument('--output',type=Path,default=ROOT/'data/toy')
    parser.add_argument('--splits',type=Path,default=ROOT/'data/synthetic/splits.json')
    parser.add_argument('--train',type=int,default=100)
    parser.add_argument('--validation',type=int,default=20)
    parser.add_argument('--test',type=int,default=10)
    args=parser.parse_args()
    pools=json.loads(args.splits.read_text())
    old=json.loads((args.output/'splits.json').read_text()) if (args.output/'splits.json').exists() else {}
    splits={}
    for group in ('train','validation','test'):
        desired=getattr(args,group)
        if not 1<=desired<=len(pools[group]):parser.error(f'Invalid size for {group}')
        priority=[pid for pid in old.get(group,[]) if pid in pools[group]]
        splits[group]=(priority+[pid for pid in pools[group] if pid not in priority])[:desired]
    ids=set(sum(splits.values(),[]))
    if len(ids)!=sum(map(len,splits.values())):parser.error('Split pools overlap')
    dataset=load_dataset(args.source)
    if not ids<=set(dataset.patients.patient_id):parser.error('Requested IDs absent from source')
    for name in ('patients','events','pre_study_events','measurements','dynamic_covariates'):
        frame=getattr(dataset,name)
        setattr(dataset,name,frame[frame.patient_id.isin(ids)].copy().reset_index(drop=True))
    dataset.save(args.output)
    (args.output/'splits.json').write_text(json.dumps(splits,indent=2))
    path=args.output/'metadata.json'
    metadata=json.loads(path.read_text())
    source_path=args.source.resolve()
    source_label=source_path.relative_to(ROOT).as_posix() if source_path.is_relative_to(ROOT) else str(source_path)
    metadata['subset']={'source':source_label,'patient_count':len(ids),
                        'split_counts':{k:len(v) for k,v in splits.items()},
                        'selection':'Existing IDs retained, then existing full split order; no outcome-based selection.'}
    path.write_text(json.dumps(metadata,indent=2))
    print('Saved',len(ids),'patients:',{k:len(v) for k,v in splits.items()},'scenario:',dataset.config.scenario)

if __name__=='__main__':main()
