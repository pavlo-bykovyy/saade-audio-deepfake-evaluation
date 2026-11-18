"""SAADE v5: reproducible scenario-adaptive training for thesis experiments.

Key invariants:
* AUA and Melnyk_dus are read-only and are never output directories.
* parent/speaker groups are split BEFORE segmentation;
* an adaptation fold is carved exclusively from TRAIN speaker/voice groups;
* adaptation fold selects SAADE augmentation distribution and thresholds;
* validation is an untouched comparison fold; hidden_test never read here;
* E1/E2/E3 share identical training draws, epochs, seeds and augmentation budget;
* real/fake class balancing is within each partition, no file deletion;
* model files and all outputs live in cfg.out unique for this protocol.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from collections import defaultdict
import hashlib, json, math, os, subprocess, warnings
import numpy as np
import pandas as pd
import saade_final_core as S
import saade_final_methods as M

PARTITIONS=('train','adaptation','validation','hidden_test')


def _out(cfg):
    root=Path(cfg.root).expanduser().resolve()
    out=Path(cfg.out).expanduser().resolve()
    if out==root or root in out.parents or any('melnyk' in p.casefold() for p in out.parts):
        raise ValueError('Forbidden output path (AUA or Melnyk folder).')
    if not out.is_relative_to(Path('/home/jovyan/work/audio_real_fake').resolve()):
        # Permit a local fixture for developer tests, but do not silently target a different work tree.
        if str(root).startswith('/home/jovyan/work'):
            raise ValueError('Production outputs must stay inside /home/jovyan/work/audio_real_fake/')
    out.mkdir(parents=True,exist_ok=True)
    return out


def create_adaptation_fold(parents,cfg,seed=42,target_fraction=.22,require_two_groups=True):
    """Hold out known identities from *training* as adaptation; never touch validation/test.

    This is group-disjoint by group_id (source/voice), not proof of independent generators.
    Unknown-identity train_only material remains strictly training-only.
    """
    out=_out(cfg)
    df=parents.copy()
    if 'split' not in df:raise ValueError('Missing preassigned split')
    if 'group_id' not in df or 'identity_status' not in df:
        raise ValueError('Missing provenance grouping')
    if (df.split=='adaptation').any(): raise ValueError('Adaptation fold already exists: use fresh parents')
    rng=np.random.default_rng(seed)
    group_moves=[]
    for label in (0,1):
        train=df[(df.split=='train')&(df.label.astype(int)==label)&(df.split_role=='eligible')]
        info=train.groupby('group_id').agg(n=('path','size')).reset_index()
        if require_two_groups and len(info)<2:
            raise ValueError(f'Class {label}: need >=2 VERIFIED train groups to reserve adaptation. Found {len(info)}')
        if len(info)<2:
            warnings.warn(f'Class {label}: no independent adaptation possible; cannot run E3 safely')
            continue
        # Reserve at least 1 group and keep >=1 group in model-training data.
        idx=rng.permutation(len(info)); ordered=info.iloc[idx]
        quota=max(1,int(round(len(train)*float(target_fraction))))
        selected=[];count=0
        for r in ordered.itertuples():
            if len(selected)>=len(info)-1:break
            selected.append(r.group_id);count+=r.n
            if count>=quota:break
        group_moves+=selected
    if not group_moves:raise ValueError('No adaptation groups selected')
    df.loc[(df.split=='train')&df.group_id.isin(group_moves),'split']='adaptation'
    for partition in ('train','adaptation','validation'):
        part=df[df.split==partition]
        if part.label.nunique()!=2:
            raise ValueError(f'{partition} missing a class after adaptation carve-out')
    if df.groupby('group_id').split.nunique().max()>1:
        raise AssertionError('Group leakage after adaptation carving')
    for hcol in ['source_sha256','working_sha256','path']:
        if hcol not in df:continue
        h=df[df[hcol].fillna('').astype(str).str.strip().ne('')].groupby(hcol).split.nunique()
        if (h>1).any():raise AssertionError(f'{hcol} across folds')
    df.to_csv(out/'V5_01_PARENT_FOLDS.csv',index=False)
    S.save_json(out/'V5_01_ADAPTATION_PROVENANCE.json',dict(seed=seed,fraction=target_fraction,
      identities_moved=group_moves,selection='metadata verified train groups only',
      held_out_validation_untouched=True,hidden_test_untouched=True))
    return df


def create_matched_adaptation_fold(parents,cfg,seed=42,target_fraction=.22):
    """Group- AND pair-disjoint adaptation using connected components.

    Every text-pair connects the human speaker group to a synthetic voice group.
    All rows reachable through either shared group or pair must stay together.
    If there are insufficient independent components, fail rather than split pairs.
    """
    out=_out(cfg)
    df=parents.copy().reset_index(drop=True)
    if not {'split','pair_id','group_id','label'}.issubset(df.columns):
        raise ValueError('Matched parents require split,pair_id,group_id,label')
    if df.pair_id.isna().any() or df.pair_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every matched parent must have a pair_id')
    if df.groupby('pair_id').split.nunique().gt(1).any():
        raise AssertionError('Matched pair crosses existing folds')
    rng=np.random.default_rng(int(seed))
    train=df[df.split.eq('train')]
    if train.empty:raise ValueError('No paired training rows')
    # Bipartite union-find: pair nodes and voice/speaker group nodes.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[b]=a
    for r in train.itertuples():
        union('PAIR:'+str(r.pair_id),'GROUP:'+str(r.group_id))
    pair_component={pid:find('PAIR:'+str(pid)) for pid in train.pair_id.unique()}
    train=train.copy();train['component_id']=train.pair_id.map(pair_component)
    components=train.groupby('component_id').agg(rows=('path','size'),pairs=('pair_id','nunique'),
                              labels=('label','nunique')).reset_index()
    if len(components)<2:
        raise ValueError('Only one independent connected speaker/voice component in train; '
                         'cannot build pair-safe adaptation. Provide more independent TTS voices.')
    if components.labels.lt(2).any():
        raise ValueError('At least one connected component has one class; inspect pair registry')
    components=components.iloc[rng.permutation(len(components))]
    quota=max(1,int(round(len(train)*float(target_fraction))))
    picked=[];count=0
    for rr in components.itertuples():
        if len(picked)>=len(components)-1:break
        picked.append(rr.component_id);count+=int(rr.rows)
        if count>=quota:break
    move_pairs=set(train[train.component_id.isin(picked)].pair_id)
    df.loc[df.split.eq('train')&df.pair_id.isin(move_pairs),'split']='adaptation'
    for col in ['pair_id','group_id','source_sha256','working_sha256','path']:
        if col not in df.columns:continue
        present=df[col].fillna('').astype(str).str.strip().ne('')
        if df.loc[present].groupby(col).split.nunique().gt(1).any():
            raise AssertionError(f'Leakage in matched split: {col}')
    if df.groupby(['split','pair_id']).label.nunique().ne(2).any():
        raise AssertionError('Split has an incomplete REAL/FAKE pair')
    for part in ('train','adaptation','validation','hidden_test'):
        if df[df.split.eq(part)].label.nunique()!=2:
            raise ValueError(f'{part}: not both classes')
    df.to_csv(out/'V5_01_PARENT_FOLDS.csv',index=False)
    S.save_json(out/'V5_01_ADAPTATION_PROVENANCE.json',dict(method='connected_components_speaker_voice_pair',
        train_components_before=len(components),components_to_adaptation=len(picked),
        adaptation_pairs=len(move_pairs),seed=int(seed),validation_untouched=True,hidden_untouched=True))
    return df


def balance_per_partition(segments,cfg,real_pct=.50,seed=42):
    """Random *downsampling* of segment rows without altering their split or parent link.

    Training statistics are computed on segments, not independent speakers. Outputs
    include exactly reproducible retained/excluded manifests. For text-matched pairs
    use balance_pairs() instead: do not sever parent-real/TTS coupling.
    """
    out=_out(cfg)
    if not 0<float(real_pct)<1:raise ValueError('real_pct must be in (0,1)')
    if segments.groupby('group_id').split.nunique().gt(1).any():
        raise AssertionError('Preexisting group leakage')
    rng=np.random.default_rng(seed)
    kept=[];removed=[];report=[]
    for part, sub in segments.groupby('split',sort=False):
        a=sub[sub.label.astype(int)==0];b=sub[sub.label.astype(int)==1]
        if len(a)==0 or len(b)==0:
            raise ValueError(f'{part}: only one class; cannot balance')
        # For strict 50/50 n is min count, otherwise fill largest possible exact ratio approx.
        size=min(int(len(a)/real_pct),int(len(b)/(1-real_pct)))
        if abs(real_pct-.5)<1e-12:
            n_real=n_fake=min(len(a),len(b));size=2*n_real
        else:
            n_real=min(len(a),int(round(size*real_pct)))
            n_fake=min(len(b),int(round(size*(1-real_pct))))
        aa=a.iloc[rng.choice(len(a),n_real,replace=False)]
        bb=b.iloc[rng.choice(len(b),n_fake,replace=False)]
        pick=pd.concat([aa,bb]);drop=sub.drop(index=pick.index)
        kept.append(pick);removed.append(drop)
        report.append(dict(split=part,real_before=len(a),fake_before=len(b),
            real_after=n_real,fake_after=n_fake,excluded=len(drop),
            real_fraction_after=n_real/(n_real+n_fake),
            groups_after=pick.group_id.nunique(),parents_after=pick.parent_file_id.nunique()))
    final=pd.concat(kept,ignore_index=True).sort_values(['split','relative_path']).reset_index(drop=True)
    dropped=pd.concat(removed,ignore_index=True) if removed else segments.iloc[:0].copy()
    assert final.groupby('group_id').split.nunique().max()==1
    assert final.groupby('parent_file_id').split.nunique().max()==1
    final.to_csv(out/'V5_02_BALANCED_SEGMENTS.csv',index=False)
    dropped.to_csv(out/'V5_02_EXCLUDED_SEGMENTS.csv',index=False)
    info=pd.DataFrame(report)
    info.to_csv(out/'V5_02_BALANCE_REPORT.csv',index=False)
    return final,info


def _unique_adaptation_subset(df,max_per_class=24,seed=42):
    # Parent-aware, not dominated by long recordings. Scenario evaluation unchanged.
    chunks=[];rng=np.random.default_rng(seed)
    for label in (0,1):
        x=df[df.label.astype(int)==label].copy()
        x=x.iloc[rng.permutation(len(x))].copy()
        x['parent_rank']=x.groupby('parent_file_id').cumcount()
        x=x.sort_values(['parent_rank','relative_path']).head(max_per_class).drop(columns='parent_rank')
        chunks.append(x)
    return pd.concat(chunks,ignore_index=True)


def _difficulty_policy(predictions,prior=10.,temperature=.12,uniform_floor=.20):
    """Group-wise confusion-balanced scenario difficulty -> category and level probabilities.

    Groups in adaptation (not signal windows) each contribute equally to difficulty.
    Shrink tiny cells toward 0.5. Uniform floor prevents collapsing the search.
    """
    x=predictions[predictions.category.isin(S.CATEGORIES)].copy()
    x['incorrect']=(x.pred.astype(int)!=x.label.astype(int)).astype(float)
    levels=list(S.available_levels('v2_5level'))
    risk={}; counts={}
    for cat in S.CATEGORIES:
        for lv in levels:
            cell=x[(x.category==cat)&(x.level==lv)]
            er=[]
            for lab in (0,1):
                xlab=cell[cell.label==lab]
                if len(xlab):
                    # One mean per independently identified group, not per duplicate window.
                    er.append(float(xlab.groupby('group_id').incorrect.mean().mean()))
            v=float(np.mean(er)) if len(er)==2 else .5
            support=int(len(cell))
            risk[(cat,lv)]=(support*v+prior*.5)/(support+prior)
            counts[(cat,lv)]=support
    overall=np.array([np.mean([risk[(c,l)] for l in levels]) for c in S.CATEGORIES],float)
    def soft(vals):
        z=np.asarray(vals,dtype=float);z=(z-z.max())/max(.03,temperature)
        p=np.exp(z);p=p/p.sum()
        return (1-uniform_floor)*p+uniform_floor/len(p)
    cat_prob=soft(overall)
    cat_weights=dict(zip(S.CATEGORIES,[float(j) for j in cat_prob]))
    lev_weights={c:dict(zip(levels,[float(v) for v in soft([risk[(c,l)] for l in levels])])) for c in S.CATEGORIES}
    table=pd.DataFrame([dict(category=c,level=l,group_balanced_error=risk[(c,l)],
        n_scenario_rows=counts[(c,l)],category_probability=cat_weights[c],level_probability=lev_weights[c][l])
        for c in S.CATEGORIES for l in levels])
    return cat_weights,lev_weights,table


class CurriculumAudioDataset(S.AudioDataset):
    def __init__(self,records,cfg,regime,cat_weights=None,level_weights=None):
        super().__init__(records,cfg,train=True,regime=regime,
          adaptive_weights=cat_weights if regime=='adaptive' else None)
        self.cat_weights=cat_weights
        self.level_weights=level_weights
    def __getitem__(self,i):
        r=self.records.iloc[i]
        seed=S.deterministic_seed(self.cfg.seed,self.epoch,i,r.relative_path)
        wav=S.read_fixed_segment(r.path,self.cfg.sample_rate,self.cfg.seconds,
            start_sec=r.get('segment_start_sec',None),seed=seed,train=True)
        if self.regime!='static':
            rng=np.random.default_rng(seed+123)
            if rng.random()<self.cfg.augmentation_probability:
                categories=S.CATEGORIES
                p=np.array([self.cat_weights[c] for c in categories]) if self.regime=='adaptive' else None
                cat=str(rng.choice(categories,p=p))
                levels=list(S.available_levels(self.cfg.scenario_profile))
                probs=(np.array([self.level_weights[cat][l] for l in levels])
                       if self.regime=='adaptive' else None)
                lv=int(rng.choice(levels,p=probs))
                val=S.scenario_severity(cat,lv,self.cfg.scenario_profile)
                wav=S.transform(wav,cat,lv,self.cfg.sample_rate,seed,severity_override=val)
        return np.asarray(wav,dtype='float32'),int(r.label)


def _model_predictions_on_adaptation(model,name,regime,df,cfg,max_per_class,seed):
    subset=_unique_adaptation_subset(df[df.split=='adaptation'],max_per_class,seed)
    # Original split tag stays adaptation, hidden_test and validation are never accessed.
    yy,ss=S.get_validation_scores(model,name,subset,cfg)
    thr=S.best_threshold(yy,ss)
    p=S.run_scenario_predictions(subset,model,name,regime,cfg,threshold=thr,split='adaptation')
    return float(thr),p


def train_curriculum_model(splits,cfg,name,regime,seed,epochs,update_every=2,
                           initial_policy=None,adapt_max_per_class=24):
    """Train fixed epochs, identical updates/augmentation count across regimes.

    Adaptive distribution changes ONLY using `adaptation` predictions. Does not
    touch validation nor hidden_test for threshold selection or policy updates.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    out=_out(cfg);mdir=out/'models';mdir.mkdir(exist_ok=True)
    if name not in ('spectrogram_cnn','raw_wave_cnn','mfcc_svm','aasist','rawnet2','wavlm'):
        raise ValueError(name)
    tr=splits[splits.split=='train'].reset_index(drop=True)
    adaptation=splits[splits.split=='adaptation'].reset_index(drop=True)
    if tr.label.nunique()!=2 or adaptation.label.nunique()!=2:
        raise ValueError('Train and adaptation must each contain two classes')
    trial_cfg=replace(cfg,seed=int(seed))
    S.seed_everything(seed+S.deterministic_seed(name)%1000)
    model=S.create_model(name,trial_cfg)
    equal={c:.25 for c in S.CATEGORIES};lev={c:{l:.2 for l in range(1,6)} for c in S.CATEGORIES}
    cat_weights,lv_weights=(initial_policy if initial_policy else (equal,lev))
    if regime!='adaptive':cat_weights,lv_weights=equal,lev
    ds=CurriculumAudioDataset(tr,trial_cfg,regime,cat_weights,lv_weights)
    logs=[]; policy_history=[]; draws=0
    if name=='mfcc_svm':
        all_x=[];all_y=[]
    else:
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=model.to(device)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=trial_cfg.lr)
        pos=torch.tensor([int((tr.label==0).sum())/max(1,int((tr.label==1).sum()))],device=device,dtype=torch.float32)
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos)
    for ep in range(int(epochs)):
        ds.epoch=ep
        ds.cat_weights=cat_weights;ds.level_weights=lv_weights
        if name=='mfcc_svm':
            for i in range(len(ds)):
                wav,label=ds[i];all_x.append(S.mfcc_features(wav,trial_cfg.sample_rate));all_y.append(label)
            model.fit(np.stack(all_x),np.asarray(all_y))
            loss=np.nan
        else:
            # Reset same deterministic shuffle epoch per regime, seed and model.
            g=torch.Generator();g.manual_seed(seed+ep*101)
            loader=DataLoader(ds,batch_size=trial_cfg.batch_size,shuffle=True,
                              num_workers=0,collate_fn=S._batch_tensors,generator=g)
            model.train();losses=[]
            for a,b in loader:
                a=a.to(device);b=b.to(device)
                opt.zero_grad(set_to_none=True)
                logits=model(a).view(-1)
                L=loss_fn(logits,b)
                if not torch.isfinite(L):raise FloatingPointError('NaN train loss')
                L.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                losses.append(float(L.detach().cpu()))
            loss=float(np.mean(losses))
        draws+=len(ds)
        logs.append(dict(model=name,regime=regime,seed=seed,epoch=ep+1,
                         n_training_draws=len(ds),loss=loss))
        if regime=='adaptive' and ep+1<epochs and ((ep+1)%max(1,int(update_every))==0):
            threshold,pred=_model_predictions_on_adaptation(model,name,regime,splits,trial_cfg,adapt_max_per_class,seed)
            new_cat,new_levels,table=_difficulty_policy(pred)
            # Momentum stabilizes small adaptation sets; fixed independent fold.
            cat_weights={c:.5*cat_weights[c]+.5*new_cat[c] for c in S.CATEGORIES}
            lv_weights={c:{lv:.5*lv_weights[c][lv]+.5*new_levels[c][lv] for lv in range(1,6)} for c in S.CATEGORIES}
            for r in table.itertuples():
                policy_history.append(dict(model=name,seed=seed,epoch=ep+1,
                    category=r.category,level=r.level,risk=r.group_balanced_error,
                    n_scenario_rows=r.n_scenario_rows,category_weight=cat_weights[r.category],
                    level_weight=lv_weights[r.category][r.level]))
        print(f'  {name} {regime} seed={seed} epoch {ep+1}/{epochs}: train loss {loss:.4f}' if np.isfinite(loss)
              else f'  {name} {regime} seed={seed} epoch {ep+1}/{epochs}: MFCC-SVM fitted')
    threshold,adapt_preds=_model_predictions_on_adaptation(model,name,regime,splits,trial_cfg,adapt_max_per_class,seed)
    metrics=S.metric_row(adapt_preds.loc[adapt_preds.category=='baseline','label'],
                         adapt_preds.loc[adapt_preds.category=='baseline','score'],threshold)
    sub=mdir/f'{name}_{regime}_seed{seed}'
    if name=='mfcc_svm':
        import joblib
        joblib.dump(model,str(sub)+'.joblib')
    else:torch.save(model.state_dict(),str(sub)+'.pt')
    meta=dict(model=name,regime=regime,seed=int(seed),epochs=int(epochs),
        threshold=float(threshold),adaptation_metrics=metrics,training_draws=int(draws),
        augmentation_probability=(0. if regime=='static' else float(cfg.augmentation_probability)),
        final_category_weights=cat_weights,final_level_weights=lv_weights,
        caveat='Threshold chosen ONLY on adaptation. Validation untouched until all models are trained.',
        model_path=str(sub)+('.joblib' if name=='mfcc_svm' else '.pt'))
    S.save_json(str(sub)+'_meta.json',meta)
    pd.DataFrame(logs).to_csv(str(sub)+'_train_history.csv',index=False)
    if policy_history:pd.DataFrame(policy_history).to_csv(str(sub)+'_policy_history.csv',index=False)
    return model,meta,adapt_preds


def evaluate_validation_only(splits,cfg,trained,seed):
    """Read validation only. Do not modify policy, checkpoint, threshold, hyperparams."""
    result=[]
    for (model_name,regime), (model,meta) in trained.items():
        print('VAL',model_name,regime,seed)
        p=S.run_scenario_predictions(splits,model,model_name,regime,cfg,
                            threshold=meta['threshold'],split='validation')
        p['seed']=int(seed)
        result.append(p)
    return pd.concat(result,ignore_index=True)


def evaluate_all_seeds(splits,cfg,models=('spectrogram_cnn','raw_wave_cnn','mfcc_svm'),
                      seeds=(42,123,777),epochs=12,update_every=3,adapt_max_per_class=24,
                      require_preflight=True):
    out=_out(cfg)
    if splits.groupby('group_id').split.nunique().gt(1).any():raise AssertionError('Group leakage')
    if require_preflight and not set(PARTITIONS).issubset(splits.split.unique()):
        raise ValueError('Expected train/adaptation/validation/hidden_test')
    if cfg.scenario_profile!='v2_5level':raise ValueError('Required 21-scenario profile')
    all_preds=[];adapt_reports=[];all_metas=[]
    for seed in seeds:
        trial_cfg=replace(cfg,seed=int(seed),out=str(out))
        trained={}
        for name in models:
            e1,meta,p1=train_curriculum_model(splits,trial_cfg,name,'static',int(seed),epochs,
                  update_every,adapt_max_per_class=adapt_max_per_class)
            trained[(name,'static')]=(e1,meta);all_metas.append(meta)
            policy=_difficulty_policy(p1)[:2]  # ONLY adaptation risk, no validation use
            e2,meta,p2=train_curriculum_model(splits,trial_cfg,name,'random',int(seed),epochs,
                  update_every,adapt_max_per_class=adapt_max_per_class)
            trained[(name,'random')]=(e2,meta);all_metas.append(meta)
            e3,meta,p3=train_curriculum_model(splits,trial_cfg,name,'adaptive',int(seed),epochs,
                  update_every,initial_policy=policy,adapt_max_per_class=adapt_max_per_class)
            trained[(name,'adaptive')]=(e3,meta);all_metas.append(meta)
            adapt_reports.extend([p1.assign(seed=seed),p2.assign(seed=seed),p3.assign(seed=seed)])
        val=evaluate_validation_only(splits,trial_cfg,trained,int(seed))
        all_preds.append(val)
        val.to_csv(out/f'V5_04_VALIDATION_PREDICTIONS_SEED_{seed}.csv',index=False)
    preds=pd.concat(all_preds,ignore_index=True)
    adapts=pd.concat(adapt_reports,ignore_index=True)
    preds.to_csv(out/'V5_04_VALIDATION_PREDICTIONS_ALL.csv',index=False)
    adapts.to_csv(out/'V5_03_ADAPTATION_PREDICTIONS_ALL.csv',index=False)
    pd.DataFrame(all_metas).drop(columns=['final_level_weights'],errors='ignore').to_csv(
        out/'V5_03_MODEL_RUNS.csv',index=False)
    return preds,adapts


def metric_tables(preds,cfg):
    """Per-seed scenario metrics and per-seed ASRI; never pool repeats as independent audio."""
    o=_out(cfg);sm=[];ari=[]
    for seed,g in preds.groupby('seed'):
        scores,asri=S.summarize_predictions(g,cfg,report_tag=f'v5_seed_{seed}')
        scores['seed']=int(seed);asri['seed']=int(seed)
        sm.append(scores);ari.append(asri)
    metrics=pd.concat(sm,ignore_index=True);table=pd.concat(ari,ignore_index=True)
    metrics.to_csv(o/'V5_05_SCENARIO_METRICS.csv',index=False)
    table.to_csv(o/'V5_05_ASRI_BY_SEED.csv',index=False)
    # Paired comparison: each seed/model E3 vs E2; mean and variation across seeds.
    pv=table.pivot(index=['model','seed'],columns='regime',values='asri').reset_index()
    pv['delta_adaptive_minus_random']=pv['adaptive']-pv['random']
    pv['delta_adaptive_minus_static']=pv['adaptive']-pv['static']
    pv.to_csv(o/'V5_05_PAIRED_SEED_DELTAS.csv',index=False)
    stats=pv.groupby('model').agg(n_seeds=('seed','nunique'),
       mean_delta_E3_E2=('delta_adaptive_minus_random','mean'),
       std_delta_E3_E2=('delta_adaptive_minus_random','std'),
       n_seeds_E3_wins=('delta_adaptive_minus_random',lambda x:int(sum(x>0)))).reset_index()
    stats.to_csv(o/'V5_05_DELTA_SUMMARY.csv',index=False)
    return metrics,table,pv,stats


def generate_tts_job_queue(parents,cfg,max_real=400,seed=42):
    """Export matched text synthesis plan. Does not silently call any vendor/network.

    The queue provides original transcript, parent, original speaker and locked fold.
    Human transcript is NOT verified unless its provenance says so. Edit/review
    before optional Piper generation. TTS files do not become real independent voices.
    """
    out=_out(cfg)
    x=parents[(parents.label.astype(int)==0)&parents.transcription.fillna('').astype(str).str.strip().ne('')].copy()
    x['text_normalized']=x.transcription.astype(str).str.replace(r'\s+',' ',regex=True).str.strip()
    x=x[x.text_normalized.str.len().between(20,350)]
    x=x.drop_duplicates(subset=['source_record_id','text_normalized','split'])
    # Hold each exact normalized transcript in ONE fold to prevent text leakage.
    x['text_key']=x.text_normalized.str.casefold().str.replace(r'\W+','',regex=True)
    x['split_priority']=x.split.map({'train':0,'adaptation':1,'validation':2,'hidden_test':3}).fillna(9)
    x=x.sort_values(['split_priority','relative_path']).drop_duplicates('text_key',keep='first')
    # Drop unsafe ambiguous source attribution.
    x=x[x.source.isin(['CommonVoice','FLEURS','AUA_own'])]
    rng=np.random.default_rng(seed)
    x=x.iloc[rng.permutation(len(x))].head(max_real)
    q=x[['path','relative_path','source','source_record_id','speaker_id','group_id','split','text_normalized']].copy()
    q['source_license']=x.get('license',pd.Series('',index=x.index)).astype(str).values
    q['source_approval_status']=x.get('approval_status',pd.Series('',index=x.index)).astype(str).values
    q['pair_id']=q.apply(lambda r:'PAIR_'+hashlib.sha256((r['relative_path']+'|'+r['text_normalized']).encode()).hexdigest()[:15],axis=1)
    q['tts_status']='PENDING_REVIEW'
    q['tts_model_id']=''
    q['tts_voice_id']=''
    q['generated_path']=''
    queue_path=out/'V5_00_TTS_GENERATION_QUEUE.csv'
    if queue_path.exists():
        saved=pd.read_csv(queue_path,dtype=str,keep_default_na=False)
        if set(saved.pair_id)!=set(q.pair_id):
            raise RuntimeError('TTS queue parents/transcripts have changed. Preserve existing approvals; '
                               'start a new experiment directory and review the changes.')
        return saved
    q.to_csv(queue_path,index=False)
    q.groupby('split',dropna=False).size().rename('queued_texts').reset_index().to_csv(
        out/'V5_00_TTS_QUEUE_SPLIT_COUNTS.csv',index=False)
    S.save_json(out/'V5_00_TTS_QUEUE_INFO.json',dict(rows=len(q),
      warning='User must review text and licenses, assign voice models and keep voice identities disjoint between locked splits. No audio is generated by this function.'))
    return q


def synthesize_piper_reviewed(queue_csv,cfg,model_by_voice,limit=None):
    """Optional local TTS using preinstalled Piper CLI and explicitly supplied voice models.

    Input CSV must have tts_status=APPROVED, tts_voice_id, tts_model_id,
    and rows with valid human transcripts and consent/license. Each model_by_voice
    maps tts_voice_id -> local .onnx model path. No internet calls or fake claims.
    """
    out=_out(cfg)
    from shutil import which
    if not which('piper'):raise FileNotFoundError('piper CLI not found in PATH; TTS step is OPTIONAL')
    q=pd.read_csv(queue_csv,dtype=str,keep_default_na=False)
    required=['tts_status','tts_voice_id','tts_model_id','pair_id','text_normalized','split']
    missing=[c for c in required if c not in q.columns]
    if missing:raise ValueError('Missing columns: '+','.join(missing))
    approved=q[q.tts_status.eq('APPROVED')].copy()
    if limit:approved=approved.head(limit)
    if approved.empty:raise ValueError('No APPROVED TTS jobs')
    # A global voice cannot appear in two data partitions under a voice-disjoint protocol.
    repeats=approved.groupby('tts_voice_id').split.nunique()
    if repeats.gt(1).any():raise ValueError('Same TTS voice assigned to multiple folds: '+str(repeats[repeats>1].index.tolist()))
    # Distinct voice IDs MUST NOT point at the same ONNX model (aliases are not new voices).
    resolved={}
    model_hashes={}
    for voice in approved.tts_voice_id.unique():
        path=Path(model_by_voice.get(voice,'')).expanduser()
        if not path.is_file() or path.suffix.lower()!='.onnx':
            raise FileNotFoundError(f'Piper .onnx model missing for {voice}: {path}')
        resolved[voice]=path
        model_hashes[voice]=_sha256_file(path)
    inverted=defaultdict(set)
    for voice, digest in model_hashes.items():inverted[digest].add(voice)
    aliases={digest:sorted(names) for digest,names in inverted.items() if len(names)>1}
    if aliases:
        raise ValueError('Same Piper ONNX file is aliased as multiple independent TTS voices: '+str(aliases))
    outdir=out/'paired_tts_generated';outdir.mkdir(exist_ok=True)
    receipts=[]
    for r in approved.itertuples():
        path=resolved[r.tts_voice_id]
        dest=outdir/f'{r.pair_id}_{S.safe_key(r.tts_voice_id)}.wav'
        proc=subprocess.run(['piper','--model',str(path),'--output_file',str(dest)],
                  input=r.text_normalized,capture_output=True,text=True,timeout=240)
        if proc.returncode or not dest.is_file():
            raise RuntimeError(f'Piper failed for {r.pair_id}: {proc.stderr[-350:]}')
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        receipts.append(dict(pair_id=r.pair_id,original_path=r.path,generated_path=str(dest),
            original_group_id=r.group_id,split=r.split,source=r.source,model_id=r.tts_model_id,
            voice_id=r.tts_voice_id,text=r.text_normalized,sha256=digest,
            tts_model_file_sha256=model_hashes[r.tts_voice_id]))
    pd.DataFrame(receipts).to_csv(out/'V5_00_PIPER_GENERATION_RECEIPTS.csv',index=False)
    return pd.DataFrame(receipts)


def source_and_split_diagnostics(splits,cfg):
    out=_out(cfg)
    d=splits.copy()
    a=d.groupby(['split','source','label']).agg(segments=('path','size'),
               independent_groups=('group_id','nunique'),parents=('parent_file_id','nunique')).reset_index()
    a.to_csv(out/'V5_02_SPLIT_SOURCE_CLASS.csv',index=False)
    risks=[]
    for part in ['adaptation','validation','hidden_test']:
        sub=d[d.split==part]
        n=int(sub[sub.label==1].group_id.nunique())
        risks.append(dict(partition=part,independent_fake_groups=n,
          adequate_for_generalization=bool(n>=3)))
    pd.DataFrame(risks).to_csv(out/'V5_02_FAKE_GROUP_RISKS.csv',index=False)
    print('IMPORTANT: A class=source confound remains until matched TTS audio is actually generated and included.')
    return a,pd.DataFrame(risks)


def integrate_tts_pairs(original_parents,receipts_csv,cfg,min_pairs_per_split=2):
    """Replace current experimental parents with SAME-TEXT real/fake pairs.

    This is opt-in; creates a new dataset manifest only in result directory.
    Preserves the original parent split and the verified human speaker ID.
    Voices must be split-disjoint; each pair has exactly one human and one TTS file.
    """
    out=_out(cfg)
    receipts=pd.read_csv(receipts_csv,dtype=str,keep_default_na=False)
    needed={'pair_id','original_path','generated_path','split','voice_id','model_id','sha256','text'}
    if needed-set(receipts.columns):raise ValueError(f'Missing receipts fields: {sorted(needed-set(receipts.columns))}')
    receipts=receipts.drop_duplicates('pair_id',keep=False).copy()
    if receipts.empty:raise ValueError('No uniquely identifiable paired TTS examples')
    if receipts.assign(_voice_key=receipts.voice_id.str.casefold().str.strip()).groupby('_voice_key').split.nunique().gt(1).any():
        raise AssertionError('TTS voice crosses train/validation/test; assign distinct voices per partition')
    if 'tts_model_file_sha256' in receipts.columns:
        dig=receipts[receipts.tts_model_file_sha256.fillna('').astype(str).str.strip().ne('')].copy()
        if not dig.empty:
            aliases=dig.groupby('tts_model_file_sha256').voice_id.nunique()
            if aliases.gt(1).any():
                raise AssertionError('One TTS model audio hash is assigned as multiple independent voices')
    orig=original_parents.copy()
    by_path=orig.drop_duplicates('path').set_index('path',drop=False)
    rows=[]
    for r in receipts.itertuples():
        if r.original_path not in by_path.index:
            raise ValueError(f'Original recording not in locked parent split: {r.original_path}')
        rr=by_path.loc[r.original_path]
        if isinstance(rr,pd.DataFrame):raise ValueError(f'Ambiguous original record: {r.original_path}')
        if int(rr.label)!=0 or str(rr.split)!=str(r.split):
            raise ValueError(f'Incorrect matched-pair label or split: {r.pair_id}')
        dest=Path(r.generated_path).resolve()
        if not dest.is_file() or out!=dest.parent and out not in dest.parents:
            raise ValueError('Generated WAV missing or outside results directory')
        sha=hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha.lower()!=r.sha256.lower():raise ValueError('Generated TTS SHA mismatch')
        if not str(r.voice_id).strip() or not str(r.model_id).strip():
            raise ValueError('TTS model and voice IDs required')
        # A real matched pair must contain the SAME utterance, not just a shared ID.
        import re
        def _norm_text(t):
            return re.sub(r'[^\w]+','',str(t).casefold(),flags=re.UNICODE)
        original_transcript=str(rr.get('transcription','')).strip()
        if not original_transcript or _norm_text(original_transcript)!=_norm_text(r.text):
            raise ValueError('TTS receipt text does not match original AUA transcript: '+str(r.pair_id))
        h=rr.copy();h['pair_id']=r.pair_id;h['paired_source']=f'PAIR::{rr.source}'
        h['source_original']=rr.source;h['source']=h['paired_source'];h['transcription']=r.text
        h['input_text_id']=r.pair_id
        f=h.copy();f['label']=1;f['path']=str(dest)
        f['relative_path']=f'generated_pairs/{dest.name}'
        f['group_id']=f'VOICE::{r.model_id}::{r.voice_id}'
        f['identity_status']='synthetic_voice_confirmed';f['split_role']='eligible'
        f['group_confidence']='metadata_voice'
        f['speaker_id']='';f['voice_id']=r.voice_id;f['generator']=r.model_id
        f['source_record_id']='GENERATED_TTS_'+r.pair_id
        f['source_sha256']=sha;f['working_sha256']=sha
        f['duration_sec']=S.audio_info_seconds(dest)
        if not np.isfinite(f['duration_sec']) or f['duration_sec']<cfg.min_seconds:
            raise ValueError('Paired synthesized audio invalid/short')
        f['pair_id']=r.pair_id
        rows.extend([h,f])
    paired=pd.DataFrame(rows).reset_index(drop=True)
    paired['text_key']=paired.transcription.fillna('').astype(str).str.casefold().str.replace(r'\W+','',regex=True)
    if (paired[paired.text_key.ne('')].groupby('text_key').split.nunique()>1).any():
        raise AssertionError('Matched text is reused across data folds')
    if ((paired.groupby('pair_id').label.nunique()!=2) | (paired.groupby('pair_id').size()!=2)).any():
        raise AssertionError('Each pair must contain exactly one REAL and one FAKE')
    if paired.groupby('group_id').split.nunique().gt(1).any():
        raise AssertionError('Paired voices or speakers cross splits')
    for split in ['train','validation','hidden_test']:
        if paired[paired.split==split].pair_id.nunique()<min_pairs_per_split:
            raise ValueError(f'Not enough matched pairs in {split}: requires {min_pairs_per_split}')
    paired.to_csv(out/'V5_01_MATCHED_PAIRS_PARENTS.csv',index=False)
    S.save_json(out/'V5_01_MATCHED_PAIR_REPORT.json',dict(pairs=len(paired)//2,
        train=int(paired[paired.split=='train'].pair_id.nunique()),
        validation=int(paired[paired.split=='validation'].pair_id.nunique()),
        hidden_test=int(paired[paired.split=='hidden_test'].pair_id.nunique()),
        caution='Same-text matching does NOT match capture channel, codec, room or mic.'))
    return paired


def balance_pairs(paired_segments,cfg,seed=42):
    """Preserve real/fake paired structure and parent group independence.

    If one class has more extracted windows, take the same random number per pair.
    This is not an independence guarantee for windows from the same parent.
    """
    out=_out(cfg);rng=np.random.default_rng(seed);kept=[];lost=[]
    for (split,pairid),g in paired_segments.groupby(['split','pair_id'],sort=True):
        a=g[g.label.astype(int)==0];b=g[g.label.astype(int)==1]
        n=min(len(a),len(b))
        if n<=0:lost.extend(g.index.tolist());continue
        kept.extend(a.iloc[rng.choice(len(a),n,replace=False)].index.tolist())
        kept.extend(b.iloc[rng.choice(len(b),n,replace=False)].index.tolist())
        lost.extend(g.index.difference(kept).tolist())
    res=paired_segments.loc[sorted(set(kept))].reset_index(drop=True)
    assert res.groupby(['split','pair_id','label']).size().unstack(fill_value=0).pipe(lambda d:(d[0]==d[1]).all())
    assert res.groupby('group_id').split.nunique().max()==1
    res.to_csv(out/'V5_02_PAIRED_SEGMENTS.csv',index=False)
    return res


def stratified_voice_bootstrap(preds,cfg,B=300,seed=42,min_groups_per_class=3):
    """Paired E3−E2 ASRI cluster bootstrap, stratified by TRUE class.

    The sampling unit is an entire independent speaker/voice group. The same sampled
    groups evaluate both regimes and all 20 acoustic scenarios. A bootstrap is NOT
    reported when there are fewer than `min_groups_per_class` groups in either class.
    Evaluates seeds separately; this does not treat random seeds as audio examples.
    """
    out=_out(cfg);rng=np.random.default_rng(seed);results=[]
    levels=list(S.available_levels(cfg.scenario_profile));cats=S.CATEGORIES
    keys=[(c,l) for c in cats for l in levels];K=len(keys)
    for (model,seed_value),g in preds.groupby(['model','seed']):
        g=g[g.regime.isin(['random','adaptive'])&g.category.isin(cats)].copy()
        info={lab:sorted(g[g.label==lab].group_id.unique()) for lab in (0,1)}
        common=dict(model=model,seed=int(seed_value),real_groups=len(info[0]),fake_groups=len(info[1]))
        if min(len(info[0]),len(info[1]))<min_groups_per_class:
            results.append(dict(**common,status='INSUFFICIENT_INDEPENDENT_VOICES',
                   delta_asri=np.nan,ci95_lower=np.nan,ci95_upper=np.nan,n_valid=0))
            continue
        arrays={}
        for label in (0,1):
            keyrows=[]
            for group in info[label]:
                one=g[(g.label==label)&(g.group_id==group)]
                vals=[]
                for cat,lv in keys:
                    cell=one[(one.category==cat)&(one.level==lv)]
                    dat=[]
                    for reg in ('random','adaptive'):
                        part=cell[cell.regime==reg]
                        dat.extend([float((part.pred==label).sum()),float(len(part))])
                    vals.append(dat)
                keyrows.append(vals)
            arrays[label]=np.asarray(keyrows,float) # [group,20,4] correct,count × regimes
        def calc(ix0,ix1):
            z0=arrays[0][ix0].sum(axis=0);z1=arrays[1][ix1].sum(axis=0)
            if (z0[:,1]==0).any() or (z0[:,3]==0).any() or (z1[:,1]==0).any() or (z1[:,3]==0).any():
                return np.nan
            ba_r=.5*(z0[:,0]/z0[:,1]+z1[:,0]/z1[:,1])
            ba_a=.5*(z0[:,2]/z0[:,3]+z1[:,2]/z1[:,3])
            return float(np.mean(np.maximum(0,2*ba_a-1)-np.maximum(0,2*ba_r-1)))
        obs=calc(np.arange(len(info[0])),np.arange(len(info[1])))
        sims=[]
        for _ in range(int(B)):
            ii=rng.integers(0,len(info[0]),len(info[0]))
            jj=rng.integers(0,len(info[1]),len(info[1]))
            v=calc(ii,jj)
            if np.isfinite(v):sims.append(v)
        result=dict(**common,status='OK' if len(sims)>=.9*B else 'INCOMPLETE',
            delta_asri=obs,ci95_lower=float(np.quantile(sims,.025)) if sims else np.nan,
            ci95_upper=float(np.quantile(sims,.975)) if sims else np.nan,n_valid=len(sims))
        results.append(result)
    report=pd.DataFrame(results)
    report.to_csv(out/'V5_05_BOOTSTRAP_STRATIFIED_VOICES.csv',index=False)
    return report


def research_readiness(splits,cfg,mode='pilot_existing',min_voices=3,min_seeds=3,trained_seeds=()):
    """Conservative report. A positive score never overrides missing provenance."""
    out=_out(cfg)
    rows=[]
    for part in PARTITIONS:
        x=splits[splits.split.eq(part)]
        for label in (0,1):
            z=x[x.label.astype(int).eq(label)]
            rows.append(dict(split=part,label=label,segments=len(z),parents=z.parent_file_id.nunique(),
                             independent_groups=z.group_id.nunique(),voices=z.voice_id.fillna('').astype(str).loc[lambda a:a.str.strip().ne('')].nunique() if 'voice_id' in z else 0))
    info=pd.DataFrame(rows)
    info.to_csv(out/'V5_READINESS_CLASS_GROUPS.csv',index=False)
    matched=(mode=='matched_tts')
    groups_ok=all(int(info.loc[(info.split==part)&(info.label==label),'independent_groups'].iloc[0])>=min_voices
                  for part in ('adaptation','validation','hidden_test') for label in (0,1))
    text_ok=True
    if matched:
        if 'text_key' not in splits or not splits.text_key.fillna('').astype(str).str.strip().ne('').all():
            text_ok=False
        else:
            text_ok=not splits.groupby('text_key').split.nunique().gt(1).any()
    voice_ok=True
    if matched and 'voice_id' in splits:
        v=splits[splits.label.astype(int).eq(1)].copy()
        v['_k']=v.voice_id.fillna('').astype(str).str.casefold().str.strip()
        voice_ok=v._k.ne('').all() and not v.groupby('_k').split.nunique().gt(1).any()
    leak_ok=not splits.groupby('group_id').split.nunique().gt(1).any()
    if matched:leak_ok=leak_ok and not splits.groupby('pair_id').split.nunique().gt(1).any()
    checks=dict(mode=mode,matched_tts=matched,independent_groups_ready=bool(groups_ok),
        independent_tts_voices_across_folds=bool(voice_ok),no_cross_split_text=bool(text_ok),
        no_cross_split_group_or_pair=bool(leak_ok),n_seeds=len(trained_seeds),min_seeds=int(min_seeds),
        hidden_test_opened=False,expert_assessment_included=False)
    checks['candidate_for_strict_review']=bool(matched and groups_ok and voice_ok and text_ok and leak_ok
        and len(trained_seeds)>=min_seeds)
    checks['warning']='Candidate status is NOT proof of channel matching, statistical significance, or causal benefit.'
    S.save_json(out/'V5_RESEARCH_READINESS.json',checks)
    return info,checks


def _sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for piece in iter(lambda:stream.read(2**20),b''):
            digest.update(piece)
    return digest.hexdigest()


def freeze_for_hidden_once(splits,cfg,models,seeds,epochs,mode='matched_tts',min_voice_groups=3):
    """Lock V5 model checkpoints and pre-selected thresholds BEFORE hidden audio is read.

    Requires a sufficient paired-text design. Pilot existing-AUA cannot claim finality.
    Existing protocol in the same output directory must match byte-for-byte.
    """
    out=_out(cfg)
    info,readiness=research_readiness(splits,cfg,mode=mode,min_voices=min_voice_groups,
                                      trained_seeds=seeds)
    if mode!='matched_tts' or not readiness['candidate_for_strict_review']:
        raise RuntimeError('Not ready for dissertation hidden test: see V5_RESEARCH_READINESS.json')
    if len(seeds)<3 or len(seeds)>5:
        raise ValueError('At least 3 and at most 5 declared seeds required')
    checkpoint_info=[]
    for seed in seeds:
        for model in models:
            for regime in ('static','random','adaptive'):
                stem=f'{model}_{regime}_seed{seed}'
                meta_path=out/'models'/f'{stem}_meta.json'
                if not meta_path.is_file():raise FileNotFoundError(meta_path)
                meta=json.loads(meta_path.read_text(encoding='utf-8'))
                if meta.get('epochs')!=epochs or meta.get('seed')!=seed:
                    raise RuntimeError('Mismatch between declared and trained epochs/seeds: '+stem)
                cp=Path(meta['model_path']).resolve()
                if not cp.is_file() or not cp.is_relative_to(out):
                    raise FileNotFoundError(f'Model checkpoint outside experiment folder: {cp}')
                checkpoint_info.append(dict(model=model,regime=regime,seed=seed,
                    model_path=str(cp),sha256=_sha256_file(cp),threshold=float(meta['threshold']),
                    epochs=int(epochs),model_meta_sha256=_sha256_file(meta_path)))
    val=out/'V5_04_VALIDATION_PREDICTIONS_ALL.csv'
    if not val.is_file():raise FileNotFoundError('Validation predictions missing: '+str(val))
    conf=dict(protocol_version='SAADE_V5_FINAL',split_sha256=S.frozen_hash(splits),
        checkpoints=checkpoint_info,mode=mode,models=list(models),seeds=list(seeds),
        epochs=int(epochs),scenario_profile=cfg.scenario_profile,
        validation_predictions_sha256=_sha256_file(val),
        channel_matching_not_guaranteed=True,
        notes='Thresholds and E3 policies were selected using adaptation, NOT hidden test.')
    path=out/'V5_HIDDEN_FROZEN_PROTOCOL.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old!=conf:
            raise RuntimeError('Already frozen protocol differs; use a NEW experiment directory')
    else:
        S.save_json(path,conf)
    return path


def run_hidden_once(splits,cfg,authorization=False):
    """Perform ONE hidden evaluation per locked checkpoint; no training or tuning.

    Authorization must be explicit; one-shot marker is created atomically BEFORE
    reading any hidden audio. The marker is not deleted after an interrupted run.
    """
    if not authorization:
        raise PermissionError('Set authorization=True ONLY after protocol and data acceptance')
    out=_out(cfg)
    pp=out/'V5_HIDDEN_FROZEN_PROTOCOL.json'
    if not pp.is_file():raise RuntimeError('Call freeze_for_hidden_once first')
    frozen=json.loads(pp.read_text(encoding='utf-8'))
    if frozen['split_sha256']!=S.frozen_hash(splits):
        raise RuntimeError('Locked split has changed')
    for entry in frozen['checkpoints']:
        p=Path(entry['model_path'])
        if not p.is_file() or _sha256_file(p)!=entry['sha256']:
            raise RuntimeError('Model checkpoint changed after freeze: '+str(p))
        meta=out/'models'/f"{entry['model']}_{entry['regime']}_seed{entry['seed']}_meta.json"
        if _sha256_file(meta)!=entry['model_meta_sha256']:
            raise RuntimeError('Model meta changed after freeze')
    if _sha256_file(out/'V5_04_VALIDATION_PREDICTIONS_ALL.csv') != frozen['validation_predictions_sha256']:
        raise RuntimeError('Validation results changed after freeze')
    marker=out/'V5_HIDDEN_TEST_STARTED.json'
    with marker.open('x',encoding='utf-8') as fh:
        json.dump({'protocol_sha256':_sha256_file(pp),'one_shot':True,
            'started_utc':pd.Timestamp.utcnow().isoformat()},fh,ensure_ascii=False,indent=2)
    from dataclasses import replace
    results=[]
    for item in frozen['checkpoints']:
        name=item['model']; path=Path(item['model_path'])
        if name=='mfcc_svm':
            import joblib
            model=joblib.load(path)
        else:
            import torch
            c=replace(cfg,seed=int(item['seed']))
            model=S.create_model(name,c)
            try:weights=torch.load(path,map_location='cpu',weights_only=True)
            except TypeError:weights=torch.load(path,map_location='cpu')
            model.load_state_dict(weights)
            model=model.to('cuda' if torch.cuda.is_available() else 'cpu')
        p=S.run_scenario_predictions(splits,model,name,item['regime'],cfg,
            threshold=float(item['threshold']),split='hidden_test')
        p['seed']=int(item['seed'])
        results.append(p)
    preds=pd.concat(results,ignore_index=True)
    preds.to_csv(out/'V5_07_HIDDEN_PREDICTIONS_ALL.csv',index=False)
    # Keep per-seed summaries independent.
    tab=[]
    for seed,g in preds.groupby('seed'):
        _,a=S.summarize_predictions(g,cfg,report_tag=f'hidden_seed_{seed}')
        a['seed']=int(seed);tab.append(a)
    summary=pd.concat(tab,ignore_index=True)
    summary.to_csv(out/'V5_07_HIDDEN_ASRI_BY_SEED.csv',index=False)
    return summary
