"""SAADE v3 — AUA authoritative metadata adapter and conservative parent splits.

This module reads *only* canonical data under AUA/01_audio and AUA source metadata.
It never edits the dataset. Audio provenance and human-speaker identity are distinct:
S1108/S1109 with source_group_unknown are NOT human/synthetic voice IDs.
Unknown-identity corpora are train-only in the default documented pilot protocol.
No script claims that a class label is independently expert verified.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import hashlib, json, re, warnings
import numpy as np
import pandas as pd
import saade_final_core as C

BAD_VALUES={'','nan','none','null','unknown','n/a','na'}

def clean(v):
    if v is None: return ''
    s=str(v).strip()
    return '' if s.lower() in BAD_VALUES else s

def get(row,*cols):
    for col in cols:
        if col in row and clean(row[col]): return clean(row[col])
    return ''

def read_csv(p):
    p=Path(p)
    if not p.is_file(): return pd.DataFrame()
    return pd.read_csv(p,dtype=str,keep_default_na=False,low_memory=False)

def key_path(s): return str(s).replace('\\','/').strip().lstrip('./')

def table_by(df,col):
    d=defaultdict(list)
    if col not in df: return d
    for _,r in df.iterrows():
        k=key_path(clean(r[col]))
        if k: d[k].append(r.to_dict())
    return d

def one_entry(mapping,key):
    rows=mapping.get(key,[])
    return rows[0] if len(rows)==1 else {}

def csv_source_candidates(root, patterns):
    """Find potential original metadata without assuming every file describes speakers."""
    out=[]
    for sub in patterns:
        folder=root/'00_sources_and_licenses'/'audio'/sub
        if folder.is_dir():
            out.extend([str(p) for p in folder.rglob('*') if p.suffix.lower() in ('.csv','.tsv','.json','.jsonl','.parquet')])
    return sorted(out)

def source_key(src_value,path):
    s=(str(src_value)+' '+str(path)).lower()
    if 'common_voice' in s or 'common voice' in s: return 'CommonVoice'
    if 'fleurs' in s: return 'FLEURS'
    if 'drssu' in s: return 'DRSSU'
    if 'new-tts' in s or 'new_tts' in s: return 'AUA_new_TTS'
    if '/own_processed/' in str(path).lower().replace('\\','/'): return 'AUA_own'
    return 'unknown_source'

def valid_speaker(speaker,basis,source):
    """Do not treat synthetic/technical group labels as verified speaker identity."""
    basis=basis.lower()
    if not speaker: return False
    if any(t in basis for t in ('unknown','source_group','fallback','synthetic_collection','batch')):return False
    if source in ('DRSSU','FLEURS') and not basis:return False
    if source=='CommonVoice': return 'client_id' in basis
    if source=='AUA_own': return True
    return bool('speaker' in basis or 'identity' in basis or 'verified' in basis)

def build_verified_catalog(cfg, require_source_metadata=True):
    root=Path(cfg.root).expanduser().resolve();aud=root/'01_audio';out=Path(cfg.out)
    if out.resolve()==root or root in out.resolve().parents:
        raise ValueError('Output folder must not be placed inside AUA')
    if any('melnyk' in part.casefold() for part in out.resolve().parts):
        raise ValueError('Do not write to Melnyk workspace')
    out.mkdir(parents=True,exist_ok=True)
    if not aud.is_dir():raise FileNotFoundError(f'Cannot find {aud}')
    general=read_csv(aud/'metadata_audio.csv')
    own=read_csv(aud/'own_processed'/'metadata_audio.csv')
    tts=read_csv(aud/'synthetic'/'new-tts'/'metadata_new-tts.csv')
    drssu=read_csv(root/'00_sources_and_licenses'/'audio'/'drssu'/'metadata.csv')
    # Direct metadata matches first; basename match only when globally unambiguous.
    by_work=table_by(general,'working_path');by_name=table_by(general,'file_name')
    own_by_name=table_by(own,'file_name');own_by_work=table_by(own,'working_path')
    tts_by_filename=table_by(tts,'filename')
    # The AUA field source_record_id is `drssu:<selection_rank>`; the
    # source CSV's `file_name` instead contains `generated/4308.wav`, etc.
    # Join on selection_rank and verify source SHA, never on the folder name.
    drssu_rank=table_by(drssu,'selection_rank')
    # Report potential source speaker metadata for manual linkage without inventing IDs.
    candidates=csv_source_candidates(root,['fleurs','common_voice','drssu'])
    C.save_json(out/'00_available_metadata_sources.json',dict(root=str(root),
        general_rows=len(general),own_rows=len(own),tts_rows=len(tts),drssu_rows=len(drssu),candidate_files=candidates[:300]))
    allowed=[aud/'own_processed',aud/'external_selected',aud/'synthetic']
    files=sorted({p for d in allowed if d.is_dir() for p in d.rglob('*')
                  if p.is_file() and p.suffix.lower() in C.AUDIO_EXT})
    rows=[]
    for p in files:
        rp=p.relative_to(root).as_posix()
        name=p.name
        g=one_entry(by_work,key_path(rp)) or one_entry(by_name,name)
        o=one_entry(own_by_work,key_path(rp)) or one_entry(own_by_name,name)
        t=one_entry(tts_by_filename,name)
        source=source_key(get(g,'source_dataset'),rp)
        source_record=get(o,'source_record_id') or get(g,'source_record_id')
        src_sha=get(o,'source_sha256') or get(g,'source_sha256')
        working_sha=get(o,'working_sha256') or get(g,'working_sha256')
        sp=get(o,'speaker_id') or get(g,'speaker_id')
        sp_basis=get(g,'speaker_id_basis') or get(o,'speaker_id_basis')
        mode=get(g,'mode').lower()
        note=[]
        dr={}
        if source=='DRSSU':
            # AUA metadata uses drssu:selection_rank, NOT a verified voice ID.
            rank=source_record.split(':')[-1] if source_record else ''
            dr=one_entry(drssu_rank,rank)
            # Missing rank is never silently inferred from a file name.
            if not dr:note.append('DRSSU_SOURCE_ROW_NOT_LINKED')
        label=None;label_basis='NOT_VERIFIED'
        if source=='DRSSU' and dr:
            flag=get(dr,'generated').lower()
            # Class comes from the source register, NOT from its filename.
            if flag in ('true','1','yes'):
                label=1;label_basis='DRSSU_source_metadata_generated_true'
            elif flag in ('false','0','no'):
                label=0;label_basis='DRSSU_source_metadata_generated_false'
            else:note.append('DRSSU_GENERATED_FLAG_INVALID')
            # A missing or conflicting source SHA must not be treated as
            # independently corroborated provenance. Refuse the class.
            original_sha=get(dr,'sha256')
            if not original_sha or not src_sha:
                note.append('DRSSU_SOURCE_SHA_MISSING')
                label=None;label_basis='NOT_VERIFIED'
            elif original_sha.lower()!=src_sha.lower():
                note.append('DRSSU_SOURCE_SHA_CONFLICT')
                label=None;label_basis='NOT_VERIFIED'
        elif source=='AUA_new_TTS' and t:
            # A generator-specific register establishes that this is TTS even
            # when voice_id is unavailable. Unknown voice remains TRAIN ONLY.
            if get(t,'model_name'):
                label=1
                label_basis=('TTS_source_generation_register' if get(t,'voice_id')
                             else 'TTS_source_generation_register_voice_unknown')
                if not get(t,'voice_id'):
                    note.append('TTS_VOICE_ID_UNKNOWN_TRAIN_ONLY')
            else:
                note.append('TTS_MODEL_NAME_MISSING')
        elif source in ('CommonVoice','FLEURS') and g:
            label=0;label_basis=f'{source}_dataset_origin_real_assumed'
        elif source=='AUA_own' and o:
            if get(o,'source_path') or get(o,'source_record_id'):
                label=0;label_basis='AUA_own_original_record_mapping'
            else:note.append('OWN_ORIGINAL_MAPPING_MISSING')
        else:
            note.append('SOURCE_METADATA_MISSING')
        if source=='unknown_source':note.append('UNKNOWN_SOURCE')
        # Explicit mode only when it really denotes class, not READC etc.
        class_mode={'real':0,'human':0,'bona_fide':0,'genuine':0,
                    'fake':1,'synthetic':1,'spoof':1,'tts':1,'vc':1,'generated':1}.get(mode)
        if class_mode is not None and label is not None and class_mode!=label:
            note.append('EXPLICIT_MODE_CLASS_CONFLICT')
            label=None
        known_speaker=valid_speaker(sp,sp_basis,source)
        voice=get(t,'voice_id');generator=get(t,'model_name')
        input_text_id=get(t,'input_text_id')
        voice_confirmed=bool(source=='AUA_new_TTS' and voice and generator)
        if source=='DRSSU':generator='UNKNOWN_DRSSU_GENERATOR'
        elif label==1 and not generator:generator='unknown_generator'
        if known_speaker:
            group=f'SPK::{source}::{sp}'
            confidence='metadata_speaker_verified_basis' if source=='CommonVoice' else 'metadata_speaker'
            identity='speaker_confirmed'
            split_role='eligible'
        elif voice_confirmed:
            group=f'VOICE::{source}::{generator}::{voice}'
            confidence='metadata_voice'
            identity='synthetic_voice_confirmed'
            split_role='eligible'
        else:
            # Conservative source-wise grouping, never pretend each DRSSU file is a new person.
            # Unknown voices cannot enter validation/test under exploratory protocol.
            group=f'UNKNOWN_IDENTITY::{source}'
            confidence='source_identity_unknown'
            identity='unknown_identity'
            split_role='train_only'
            note.append('IDENTITY_UNKNOWN_TRAIN_ONLY')
        if source=='DRSSU' and not dr and require_source_metadata:label=None
        duration=C.audio_info_seconds(p)
        usable=(label in (0,1) and np.isfinite(duration) and duration>=float(cfg.min_seconds)
                and not any('CONFLICT' in s for s in note))
        if not usable: note.append('UNUSABLE_OR_UNVERIFIED')
        rows.append(dict(path=str(p),relative_path=rp,source=source,source_dataset=get(g,'source_dataset') or source,
             label=label,label_source=label_basis,label_conflict=any('CONFLICT' in s for s in note),
             speaker_id=sp if known_speaker else '',technical_speaker_id=sp,
             speaker_id_basis=sp_basis,source_record_id=source_record,source_sha256=src_sha,
             working_sha256=working_sha,group_id=group,group_confidence=confidence,
             identity_status=identity,split_role=split_role,voice_id=voice,generator=generator,
             input_text_id=input_text_id,transcription=get(o,'transcription') or get(g,'transcription'),
             script_id=get(o,'script_id'),speaker_origin_group=get(o,'speaker_origin_group'),
             script_profile=get(o,'script_profile'),dialect_authenticity=get(o,'dialect_authenticity'),
             approval_status=get(o,'approval_status') or get(g,'approval_status') or get(t,'approval_status'),
             license=get(g,'license') or get(t,'license_terms'),duration_sec=duration,
             usable=bool(usable),warning='; '.join(note),primary_meta_found=bool(g),
             primary_source_meta_found=bool(dr) if source=='DRSSU' else bool(t) if source=='AUA_new_TTS' else bool(g)))
    x=pd.DataFrame(rows)
    if x.empty:raise ValueError('No audio found in canonical audio folders')
    # Dedup by exact working SHA, and by original source SHA within DRSSU.
    # AUA contains 10 repeat DRSSU source records with distinct working WAV
    # hashes; they must not be counted as 10 independent source recordings.
    nonempty=x.working_sha256.fillna('').ne('')
    dupl=nonempty & x.working_sha256.duplicated(keep='first')
    x.loc[dupl,'usable']=False
    x.loc[dupl,'warning']=x.loc[dupl,'warning'].astype(str)+'; DUPLICATE_WORKING_SHA'
    drmask=(x.source.eq('DRSSU') & x.source_sha256.fillna('').astype(str).str.strip().ne(''))
    dr_dup=drmask & x.loc[:, 'source_sha256'].where(drmask).duplicated(keep='first')
    x.loc[dr_dup, 'usable']=False
    x.loc[dr_dup, 'warning']=x.loc[dr_dup,'warning'].astype(str)+'; DRSSU_DUPLICATE_SOURCE_SHA'
    # A SHA conflict is not proof of fake/real identity; keep it out of experiments.
    x.to_csv(out/'01_catalog_all.csv',index=False)
    x[~x.usable | x.warning.ne('')].to_csv(out/'01_catalog_issues.csv',index=False)
    summary=x.groupby(['source','label','label_source','identity_status','split_role'],dropna=False).agg(
        files=('path','size'),hours=('duration_sec',lambda s:pd.to_numeric(s,errors='coerce').sum()/3600),
        groups=('group_id','nunique'),usable=('usable','sum')).reset_index()
    summary.to_csv(out/'01_sources_summary.csv',index=False)
    crosstab=pd.crosstab(x.source,x.label,dropna=False)
    crosstab.to_csv(out/'01_source_class_table.csv')
    C.save_json(out/'01_data_diagnostics.json',dict(total=len(x),usable=int(x.usable.sum()),
         class_labels_according_to_provenance=int(x.label.notna().sum()),
         metadata_unlinked=int((~x.primary_meta_found).sum()),
         unknown_identity_files=int(x.identity_status.eq('unknown_identity').sum()),
         label_source_counts=x.label_source.value_counts(dropna=False).to_dict(),
         warning='Dataset origin real/fake is not expert audio validation; source-vs-class confounding may remain.'))
    dr=x[x.source.eq('DRSSU')]
    provenance=dict(
        DRSSU_total=int(len(dr)),
        DRSSU_linked=int(dr.primary_source_meta_found.sum()) if len(dr) else 0,
        DRSSU_labelled_fake=int(dr.label.eq(1).sum()) if len(dr) else 0,
        DRSSU_source_sha_conflicts=int(dr.warning.fillna('').str.contains('DRSSU_SOURCE_SHA_CONFLICT').sum()) if len(dr) else 0,
        DRSSU_duplicate_source_sha=int(dr.warning.fillna('').str.contains('DRSSU_DUPLICATE_SOURCE_SHA').sum()) if len(dr) else 0,
        DRSSU_split_role_train_only=int(dr.split_role.eq('train_only').sum()) if len(dr) else 0,
        warning='Known source sha duplicates and technical speaker identities are not independent data.'
    )
    C.save_json(out/'01_DRSSU_PROVENANCE.json',provenance)
    print('DRSSU provenance:',provenance)
    print('AUTHORITATIVE CATALOG:',len(x),'usable',x.usable.sum())
    print(summary.to_string(index=False,max_rows=20))
    return x


def prepare_parent_splits(catalog,cfg,max_parent_files=850,unknown_train_cap=150,policy='exploratory'):
    """Create group-safe split on VERIFIED speaker/voice groups; append unknown-identity TRAIN ONLY.

    A DRSSU `source_group_unknown` is NOT a valid independent synthetic voice ID.
    Paired source/class prerequisite for strict publication mode.
    """
    out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
    v=catalog[catalog.usable & catalog.label.isin([0,1])].copy()
    v['label']=v.label.astype(int)
    eligible=v[v.split_role.eq('eligible')].copy()
    trainonly=v[v.split_role.eq('train_only')].copy()
    if eligible.empty or eligible.label.nunique()!=2:
        raise ValueError('Not enough metadata-confirmed REAL/FAKE groups for validation/test. Check TTS voice register.')
    src=pd.crosstab(v.source,v.label)
    for val in (0,1):
        if val not in src:src[val]=0
    matched=src[(src[0]>0)&(src[1]>0)].index.tolist()
    if policy=='strict_source' and not matched:
        raise ValueError('STRICT SOURCE MATCH FAILED: no confirmed source contains both REAL and FAKE. '
                         'Provide comparable real/fake source data; do not present exploratory as final.')
    # Strict mode uses matched sources only, including identity-known subset.
    if policy=='strict_source':
        eligible=eligible[eligible.source.isin(matched)].copy()
        trainonly=trainonly.iloc[:0].copy()
    cap=int(max_parent_files)
    if cap>0:
        # Set cap for split-eligible and train-only separately so unverified voices never dominate validation.
        elig_cap=max(50,cap-int(unknown_train_cap))
        local=replace(cfg,max_items=elig_cap)
        eligible=C.get_experimental_subset(eligible,local)
    # min groups for val/test from each label
    cnt=eligible.groupby('label').group_id.nunique()
    if len(cnt)!=2 or cnt.min()<3:
        raise ValueError('Need at least 3 independently identified speaker/voice groups PER CLASS')
    split=C.make_splits(eligible,cfg)
    if not trainonly.empty and unknown_train_cap>0:
        # Select per-source examples without granting speaker independence.
        def order_key(z):
            return z.sort_values(['source','source_record_id','relative_path'])
        trainonly=order_key(trainonly)
        chunks=[]
        for _,a in trainonly.groupby('source',sort=True):
            chunks.append(a.head(max(1,int(unknown_train_cap)//max(1,trainonly.source.nunique()))))
        trainonly=pd.concat(chunks,ignore_index=True)
        trainonly['split']='train'
        split=pd.concat([split,trainonly],ignore_index=True)
    # Guard source and working hashes, speaker/voice group IDs
    for col in ('group_id','source_sha256','working_sha256','path'):
        if col not in split:continue
        z=split.copy()
        z=z[z[col].fillna('').astype(str).str.strip().ne('')]
        leaks=z.groupby(col).split.nunique()
        if leaks.gt(1).any():
            raise AssertionError(f'Cross-split leakage for {col}: {int(leaks.gt(1).sum())} groups')
    split['parent_file_id']=split.relative_path
    split.to_csv(out/'03_parent_splits.csv',index=False)
    # Cross-split TTS prompt reuse: diagnostic only; identical input text is NOT a verified voice leak.
    if 'input_text_id' in split.columns:
        tx=split[split.input_text_id.fillna('').astype(str).str.strip().ne('')]
        repeat=tx.groupby('input_text_id').agg(
            splits=('split',lambda a:' | '.join(sorted(set(a)))),
            n_splits=('split','nunique'),files=('path','size'),voices=('voice_id','nunique')).reset_index()
        repeat.to_csv(out/'03_TTS_TEXT_OVERLAP.csv',index=False)
        C.save_json(out/'03_TTS_TEXT_OVERLAP_SUMMARY.json',dict(
            cross_split_text_ids=int((repeat.n_splits>1).sum()),
            note='Cross-split TTS input texts can influence text-generalization. They are NOT automatically speaker leaks.'))
    report=(split.groupby(['split','source','label','identity_status'],dropna=False)
                .agg(parent_files=('path','size'),groups=('group_id','nunique'))
                .reset_index())
    report.to_csv(out/'03_parent_split_summary.csv',index=False)
    C.save_json(out/'03_split_provenance.json',dict(policy=policy,matched_sources=matched,
         unknown_voice_files_train_only=int(split.split_role.eq('train_only').sum()),
         train_only_unknown_identity=True,
         warning='Group-safe split conditional on metadata. Does NOT resolve source/class confounding or hidden identity repetition.'))
    return split,report


def data_readiness_report(catalog, parent_splits, cfg, policy):
    out=Path(cfg.out)
    both=(pd.crosstab(catalog[catalog.usable].source,catalog[catalog.usable].label)
          .reindex(columns=[0,1],fill_value=0))
    pairs=both[(both[0]>0)&(both[1]>0)]
    check=[
        ('provenance_source_links',int((catalog.primary_meta_found).sum()),len(catalog),'CHECK' if (~catalog.primary_meta_found).any() else 'PASS'),
        ('label_unverified',int(catalog.label.isna().sum()),0,'PASS' if catalog.label.notna().all() else 'CHECK'),
        ('unknown_identity_in_validation_test',int(((parent_splits.split!='train') & parent_splits.split_role.eq('train_only')).sum()),0, 'PASS' if not ((parent_splits.split!='train') & parent_splits.split_role.eq('train_only')).any() else 'FAIL'),
        ('matched_source_both_classes',len(pairs),1,'PASS' if len(pairs)>0 else 'FAIL'),
        ('source_group_unknown_ID_reused_as_speaker',int((parent_splits.identity_status.eq('unknown_identity') & parent_splits.group_confidence.str.contains('metadata_speaker')).sum()),0,'PASS'),
        ('source_class_confounding_resolved',len(pairs),1,'PASS' if len(pairs)>0 else 'FAIL')]
    data=pd.DataFrame(check,columns=['item','actual','required','status'])
    data.to_csv(out/'DATA_READINESS.csv',index=False)
    C.save_json(out/'DATA_READINESS.json',dict(strict_ready=bool(len(pairs)>0 and policy=='strict_source'),
        purpose='SAADE diagnostics and pilot ML; not expert evaluation',
        publication_warning='Matched source is necessary, not sufficient; capture-channel / speaker / text / generator confounds still require checks.'))
    return data
