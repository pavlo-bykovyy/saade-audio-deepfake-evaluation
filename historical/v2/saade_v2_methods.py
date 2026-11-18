"""SAADE v2 extensions: source-confounding diagnostics, safe segment expansion,
seeds, scenario-specific statistics and a one-shot hidden-test gate.

This module does NOT modify the user's original AUA media.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import math
import os
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import saade_core as S


def source_balance_audit(catalog, cfg, tag='all'):
    """Assess provenance/label confounding WITHOUT claiming true matching.

    A source must contain both authentic and synthetic examples; this is a
    necessary but not sufficient condition for matched capture conditions.
    """
    x=catalog[catalog.label.isin([0,1])].copy()
    if x.empty:raise ValueError('No labeled recordings')
    x['label']=x.label.astype(int)
    src=x.groupby(['source','label']).agg(files=('path','size'),
                    independent_groups=('group_id','nunique'),
                    minutes=('duration_sec',lambda a:float(pd.to_numeric(a,errors='coerce').sum()/60))).reset_index()
    wide=src.pivot(index='source',columns='label',values='files').fillna(0)
    for lab in (0,1):
        if lab not in wide.columns: wide[lab]=0
    wide=wide.rename(columns={0:'real_count',1:'fake_count'}).reset_index()
    wide['both_labels']=wide.real_count.gt(0)&wide.fake_count.gt(0)
    out=Path(cfg.out)/'v2_quality';out.mkdir(parents=True,exist_ok=True)
    src.to_csv(out/f'{tag}_source_label_details.csv',index=False)
    wide.to_csv(out/f'{tag}_source_class_balance.csv',index=False)
    x['label_name']=x.label.map({0:'REAL',1:'FAKE'})
    matches=[]
    if 'transcription' in x.columns:
        x['transcript_key']=x.transcription.fillna('').astype(str).str.lower().str.replace(r'\W+','',regex=True)
        tx=x[x.transcript_key.str.len()>15]
        g=tx.groupby('transcript_key').agg(labels=('label','nunique'),paths=('relative_path','size')).reset_index()
        mixed=set(g[g.labels.eq(2)].transcript_key)
        if mixed:
            for _,r in tx[tx.transcript_key.isin(mixed)].iterrows():
                matches.append(dict(transcript_group=r.transcript_key[:55],label=r.label,
                                    source=r.source,relative_path=r.relative_path,group_id=r.group_id))
    matched=pd.DataFrame(matches,columns=['transcript_group','label','source','relative_path','group_id'])
    matched.to_csv(out/f'{tag}_same_text_candidates.csv',index=False)
    warnings_list=[]
    if not wide.both_labels.any():
        warnings_list.append('NO_SOURCE_WITH_BOTH_CLASSES: cross-source confounding unresolved')
    if matched.empty:
        warnings_list.append('NO_MATCHED_TEXT_PAIRS: matched utterance content not confirmed')
    if x.group_confidence.astype(str).eq('weak_file_fallback').any():
        warnings_list.append('UNVERIFIABLE_GROUPS: cannot claim speaker/generator independence')
    report=dict(total_files=len(x),matched_sources=wide[wide.both_labels].source.tolist(),
                matched_text_groups=int(matched.transcript_group.nunique()),warnings=warnings_list,
                source_purity=float((wide[['real_count','fake_count']].min(axis=1)==0).mean()))
    S.save_json(out/f'{tag}_confounding_report.json',report)
    return wide,matched,report


def check_and_prepare_parent_split(catalog,cfg,max_items=400,policy='exploratory'):
    """Split *parents*, not audio windows. Strict mode fails when source/class confounded.
    Mutually exclusive by metadata group_id; no weak IDs are silently accepted.
    """
    if policy not in ('strict_source','exploratory'):raise ValueError('Policy must be strict_source or exploratory')
    df=catalog[catalog.usable & catalog.label.isin([0,1])].copy()
    df['label']=df.label.astype(int)
    df=df[~df.group_confidence.eq('weak_file_fallback')].copy()
    wide,_,report=source_balance_audit(df,cfg,tag='candidate')
    if policy=='strict_source':
        good=set(wide[wide.both_labels].source)
        if not good:
            raise ValueError('STRICT_SOURCE blocked: no source has both REAL and FAKE. Supply matched audio + verified metadata.')
        df=df[df.source.isin(good)].copy()
        if df.label.nunique()<2: raise ValueError('Strict paired sources do not provide both labels')
    else:
        if not wide.both_labels.any():
            warnings.warn('EXPLORATORY ONLY: source predicts class. Cannot claim source-invariant deepfake detection.')
    max_items=max_items or cfg.max_items
    local=replace(cfg,max_items=max_items)
    subset=S.get_experimental_subset(df,local)
    parent_splits=S.make_splits(subset,cfg)
    if 'parent_file_id' not in parent_splits:parent_splits['parent_file_id']=parent_splits.relative_path
    out=Path(cfg.out)/'v2_quality';out.mkdir(exist_ok=True,parents=True)
    parent_splits.to_csv(out/'03_PARENT_SPLITS.csv',index=False)
    return parent_splits,report


def expand_nonoverlapping_segments(parent_splits,cfg,target=1200,max_per_parent=30,min_segments_per_file=1):
    """Expand fixed non-overlapping windows AFTER parent group/split assignment.

    The parent/source group_id and original split remain unchanged. This DOES NOT
    invent additional independent speakers. Target is a cap, never a promise.
    """
    if target<1:raise ValueError('target must be positive')
    required={'path','relative_path','group_id','label','duration_sec','split'}
    if missing:=required-set(parent_splits.columns):raise ValueError('Missing: '+str(sorted(missing)))
    rows=[];step=float(cfg.seconds)
    for r in parent_splits.itertuples():
        duration=float(r.duration_sec)
        n=int(duration//step)
        if n<=0:
            if duration>=float(cfg.min_seconds):
                starts=[0.]
            else:continue
        else:
            n=min(n,max_per_parent)
            # distinct, non-overlapping starts, but cover the full recording when capped
            available=int(duration//step)
            indices=np.linspace(0,available-1,n,dtype=int)
            indices=np.unique(indices)
            starts=(indices*step).tolist()
        for seg_index,t0 in enumerate(starts):
            d=r._asdict().copy()
            parent=d.get('parent_file_id') or str(r.relative_path)
            d['parent_file_id']=parent
            d['media_relative_path']=str(r.relative_path)
            d['segment_start_sec']=round(float(t0),6)
            d['segment_index']=int(seg_index)
            d['relative_path']=str(r.relative_path)+f'::segment_{seg_index:03d}_t{t0:.2f}'
            d['duration_sec']=min(step,duration-t0)
            rows.append(d)
    allseg=pd.DataFrame(rows)
    if allseg.empty:raise ValueError('No segments can be extracted')
    # deterministic, label-stratified cap; retain all unique groups in already-assigned splits
    target=int(target)
    if len(allseg)>target:
        rng=np.random.default_rng(cfg.seed)
        chosen=[]
        for label in (0,1):
            x=allseg[allseg.label.eq(label)].copy()
            quota=min(len(x),target//2)
            # round-robin by original parent (prevents long recording from dominating)
            x=x.iloc[rng.permutation(len(x))].copy()
            x['rank']=x.groupby('parent_file_id').cumcount()
            x=x.sort_values(['rank','parent_file_id','segment_index'])
            chosen.append(x.head(quota).drop(columns='rank'))
        allseg=pd.concat(chosen,ignore_index=True)
    # grouped split and parent tests
    gcounts=allseg.groupby('group_id').split.nunique()
    if gcounts.gt(1).any():raise AssertionError('GROUP LEAKAGE after segment expansion')
    pc=allseg.groupby('parent_file_id').split.nunique()
    if pc.gt(1).any():raise AssertionError('PARENT LEAKAGE after segment expansion')
    # same master file & label may exist in different splits when group_id was different: catch
    fp=allseg.groupby('path').split.nunique()
    if fp.gt(1).any():raise AssertionError('SAME SOURCE FILE split across partitions')
    out=Path(cfg.out)/'v2_quality';out.mkdir(parents=True,exist_ok=True)
    allseg.to_csv(out/'04_SEGMENTS_GROUP_SAFE.csv',index=False)
    # Canonical locked splits for viewer, prediction code, and frozen protocol.
    allseg.to_csv(Path(cfg.out)/'03_locked_splits.csv',index=False)
    counts=allseg.groupby(['split','label']).agg(segments=('path','size'),
                   independent_groups=('group_id','nunique'),
                   parent_files=('parent_file_id','nunique'),
                   duration_min=('duration_sec',lambda a:float(a.sum()/60))).reset_index()
    counts.to_csv(out/'04_SEGMENT_COUNTS.csv',index=False)
    report={'target':target,'actual':len(allseg),'independent_groups':int(allseg.group_id.nunique()),
            'parent_files':int(allseg.parent_file_id.nunique()),'status':'PASS' if len(allseg)>=800 else 'CHECK_INSUFFICIENT',
            'warning':'Number of segments is NOT number of independent speech sources'}
    S.save_json(out/'04_SEGMENT_REPORT.json',report)
    return allseg,counts,report


def generator_independence(splits,cfg,tag='validation'):
    d=splits.copy()
    d['gen_id']=d.generator.fillna('').astype(str).str.strip().replace({'':'unknown'})
    fake=d[d.label.eq(1)].copy()
    cross=fake.groupby(['gen_id','split']).agg(segments=('path','size'),voices=('group_id','nunique')).reset_index()
    out=Path(cfg.out)/'v2_quality';out.mkdir(parents=True,exist_ok=True)
    cross.to_csv(out/'05_GENERATOR_SPLIT.csv',index=False)
    leak=cross.groupby('gen_id').split.nunique()
    known=leak[~leak.index.str.contains('unknown',case=False)]
    report={'known_generator_count':len(known),'generators_in_multiple_splits':known[known.gt(1)].index.tolist(),
            'true_generator_holdout':bool(len(known)>=3 and known.max()==1),
            'warning':'If generator appears in train and test, claims of unseen-generator generalization are not justified'}
    S.save_json(out/'05_GENERATOR_INDEPENDENCE.json',report)
    return cross,report


def seed_replicates(base_cfg,split_df,seeds=(42,123,777),models=None,force=False):
    """Run E1/E2/E3 3–5 seeds on the SAME frozen parent split and scenario transform seed.

    Use separate subdirectories to avoid overwriting checkpoints. VERY EXPENSIVE on GPU.
    """
    if len(seeds)<3 or len(seeds)>5:raise ValueError('Publication protocol requires 3–5 seeds')
    names=tuple(models or base_cfg.models)
    parent=Path(base_cfg.out);shared_cache=parent
    all_pred=[];all_asri=[];protocols=[]
    for seed in seeds:
        cfg=replace(base_cfg,seed=int(seed),out=str(parent/'replicates'/f'seed_{seed}'),
                    scenario_cache_dir=str(shared_cache),test_finalization=False)
        Path(cfg.out).mkdir(parents=True,exist_ok=True)
        # Frozen dataset hash depends only on splits, never on random seed
        S.lock_hidden_test(split_df,cfg)
        base=[];models_store={};modelinfo={}
        for name in names:
            model,meta=S.train_one(split_df,cfg,name,'static',force=force)
            models_store[(name,'static')]=model;modelinfo[(name,'static')]=meta
            p=S.run_scenario_predictions(split_df,model,name,'static',cfg,threshold=meta['threshold'],split='validation')
            base.append(p)
        base=pd.concat(base,ignore_index=True)
        weights_by_name={name:S.validation_difficulty(base[base.model.eq(name)]) for name in names}
        weights={cat:float(np.mean([weights_by_name[n][cat] for n in names])) for cat in S.CATEGORIES}
        z=sum(weights.values());weights={k:v/z for k,v in weights.items()}
        S.save_json(Path(cfg.out)/'05_adaptive_weights.json',weights)
        full=[base]
        for name in names:
            for regime in ('random','adaptive'):
                model,meta=S.train_one(split_df,cfg,name,regime,
                      adaptive_weights=weights if regime=='adaptive' else None,force=force)
                models_store[(name,regime)]=model;modelinfo[(name,regime)]=meta
                pred=S.run_scenario_predictions(split_df,model,name,regime,cfg,threshold=meta['threshold'],split='validation')
                full.append(pred)
        preds=pd.concat(full,ignore_index=True)
        preds['training_seed']=seed
        preds.to_csv(Path(cfg.out)/'04_val_predictions_ALL.csv',index=False)
        _,asri=S.summarize_predictions(preds,cfg,report_tag='validation')
        asri['training_seed']=seed
        all_pred.append(preds);all_asri.append(asri)
        protoc=S.save_locked_protocol(cfg,split_df,modelinfo)
        protocols.append({'seed':seed,'output':str(cfg.out),'dataset_hash':protoc['dataset_hash']})
        del models_store
    table=pd.concat(all_asri,ignore_index=True)
    table.to_csv(parent/'V2_REPLICATE_ASRI.csv',index=False)
    detail=pd.concat(all_pred,ignore_index=True)
    detail.to_csv(parent/'V2_REPLICATE_VALIDATION_PREDICTIONS.csv',index=False)
    stats=table.groupby(['model','regime']).asri.agg(['count','mean','std','min','max']).reset_index()
    stats.to_csv(parent/'V2_REPLICATE_SUMMARY.csv',index=False)
    S.save_json(parent/'V2_REPLICATE_PROTOCOLS.json',protocols)
    return table,stats,detail


def interaction_statistics(predictions,cfg,tag='validation'):
    """Fit GEE with explicit model × regime and category × severity interactions.
    Returns diagnostic text; not an automatic claim of statistical significance.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    d=predictions.copy()
    d=d[d.category.isin(S.CATEGORIES)].copy()
    if d.empty:raise ValueError('No acoustic scenarios')
    d['correct']=(d.pred.astype(int)==d.label.astype(int)).astype(int)
    d['level']=d.level.astype(str)
    d['cluster_id']=d.group_id.astype(str)
    out=Path(cfg.out)/'v2_quality';out.mkdir(parents=True,exist_ok=True)
    formula='correct ~ C(model)*C(regime) + C(category)*C(level)'
    result={'formula':formula,'clusters':int(d.cluster_id.nunique()),'observations':len(d)}
    if d.cluster_id.nunique()<30:
        result['status']='CHECK_TOO_FEW_CLUSTERS'
        S.save_json(out/f'{tag}_interaction_status.json',result)
        return result
    try:
        fit=smf.gee(formula,groups='cluster_id',data=d,family=sm.families.Binomial()).fit(maxiter=100)
        summary=fit.summary().as_text()
        (out/f'{tag}_GEE_interactions.txt').write_text(summary,encoding='utf-8')
        result['status']='FIT';result['converged']=bool(fit.converged)
        result['model_regime_terms']=[n for n in fit.params.index if 'C(model)' in n and 'C(regime)' in n]
        result['warning']='Check sparse cells, separation, and cluster dependence; inspect full summary and CIs.'
    except Exception as exc:
        result['status']='FAILED';result['error']=str(exc)[:400]
    S.save_json(out/f'{tag}_interaction_status.json',result)
    return result


def reserve_hidden_evaluation(cfg,split_df,expected_seed_protocols=None):
    """Strong fail-closed one-shot marker. If interrupted, researcher must document recovery.
    Important: it cannot prevent a human manually modifying the output directory.
    """
    S.verify_locked_protocol(cfg,split_df)
    lock=S.frozen_hash(split_df)
    folder=Path(cfg.out);folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'HIDDEN_TEST_ONE_SHOT_STARTED.json'
    rec={'dataset_sha256':lock,'started_utc':pd.Timestamp.utcnow().isoformat(),
         'protocol':'SAADE v2','note':'Do not rerun hidden test or change hyperparameters after reading results.'}
    try:
        with marker.open('x',encoding='utf8') as f:
            json.dump(rec,f,ensure_ascii=False,indent=2)
    except FileExistsError:
        raise RuntimeError('Hidden-test evaluation has already been started. Do NOT rerun. See HIDDEN_TEST_ONE_SHOT_STARTED.json')
    return marker


def final_technical_summary(cfg,splits,preds=None,source_report=None,segment_report=None):
    out=Path(cfg.out)/'v2_quality';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def add(key,target,actual,status,details=''):
        rows.append(dict(check=key,target=target,actual=actual,status=status,details=details))
    c=splits.groupby(['split','label']).size().to_dict()
    add('Class-complete three splits','both labels in train/validation/hidden',str(c),
        'PASS' if all(c.get((x,y),0)>0 for x in ['train','validation','hidden_test'] for y in (0,1)) else 'FAIL')
    add('Group leakage','0',sum((splits.groupby('group_id').split.nunique()>1).values),
        'PASS' if not (splits.groupby('group_id').split.nunique()>1).any() else 'FAIL')
    add('Original-file leakage','0',sum((splits.groupby('path').split.nunique()>1).values),
        'PASS' if not (splits.groupby('path').split.nunique()>1).any() else 'FAIL')
    add('800–1500 segments','800–1500',len(splits),'PASS' if 800<=len(splits)<=1500 else 'CHECK',
        'Segment count is not independent sample count')
    if source_report:
        add('Matched source classes','≥1 source both classes',len(source_report.get('matched_sources',[])),
            'PASS' if source_report.get('matched_sources') else 'FAIL',
            'A mixed source is necessary, not sufficient, for matched acoustic conditions')
    if preds is not None and not preds.empty:
        levels=set(preds.level.unique());required=set(S.available_levels(cfg.scenario_profile))|{0}
        add('Scenario level coverage',sorted(required),sorted(levels),
            'PASS' if required.issubset(levels) else 'FAIL')
    d=pd.DataFrame(rows);d.to_csv(out/'V2_ACCEPTANCE.csv',index=False)
    return d


def hidden_all_repeats(base_cfg,split_df,seeds=(42,123,777),models=None):
    """ONE final heldout evaluation of ALL frozen seed checkpoints.

    Caller MUST invoke reserve_hidden_evaluation BEFORE this function. Predictions
    are analyzed per seed — multiple seeds are not independent audio samples.
    """
    out=Path(base_cfg.out)
    if not (out/'HIDDEN_TEST_ONE_SHOT_STARTED.json').exists():
        raise RuntimeError('Hidden test reservation missing; call reserve_hidden_evaluation first.')
    if len(seeds)<3 or len(seeds)>5:raise ValueError('Need 3–5 frozen seeds')
    names=tuple(models or base_cfg.models)
    seed_asri=[];all_parts=[];coverage=[]
    for seed in seeds:
        cfg=replace(base_cfg,seed=int(seed),out=str(out/'replicates'/f'seed_{seed}'),
                    scenario_cache_dir=str(out),test_finalization=True)
        S.verify_locked_protocol(cfg,split_df)
        train_dir=Path(cfg.out)
        weights=json.loads((train_dir/'05_adaptive_weights.json').read_text(encoding='utf-8'))
        parts=[]
        for name in names:
            for regime in ('static','random','adaptive'):
                # Only validated frozen weights are loaded; train_one reuses checkpoints.
                model,meta=S.train_one(split_df,cfg,name,regime,
                    adaptive_weights=weights if regime=='adaptive' else None,force=False)
                result=S.run_scenario_predictions(split_df,model,name,regime,cfg,
                   threshold=meta['threshold'],split='hidden_test')
                result['training_seed']=seed
                parts.append(result)
        pred=pd.concat(parts,ignore_index=True)
        pred.to_csv(train_dir/'07_hidden_predictions_ALL.csv',index=False)
        _,asri=S.summarize_predictions(pred,cfg,report_tag='hidden_test')
        asri['training_seed']=seed
        seed_asri.append(asri)
        all_parts.append(pred)
        coverage.append({'seed':int(seed),'scenarios':len(pred),'base_clips':
                         int(split_df.split.eq('hidden_test').sum()),
                         'distinct_groups':int(split_df.loc[split_df.split.eq('hidden_test'),'group_id'].nunique())})
    allpred=pd.concat(all_parts,ignore_index=True)
    allpred.to_csv(out/'07_hidden_predictions_ALL_SEEDS.csv',index=False)
    asri_by_seed=pd.concat(seed_asri,ignore_index=True)
    asri_by_seed.to_csv(out/'07_hidden_asri_by_seed.csv',index=False)
    asri_summary=asri_by_seed.groupby(['model','regime']).asri.agg(['count','mean','std','min','max']).reset_index()
    asri_summary.to_csv(out/'07_hidden_asri_summary.csv',index=False)
    S.save_json(out/'07_hidden_coverage.json',coverage)
    return allpred,asri_by_seed,asri_summary
