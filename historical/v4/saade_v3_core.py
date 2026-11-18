"""SAADE: Scenario-Adaptive Audio Deepfake Evaluation for AUA.
User-facing executable research scaffold. Python 3.10+, uses local server media.
All writes are confined to output directory. Never edits AUA.
"""
from __future__ import annotations
import dataclasses, hashlib, importlib.util, json, math, os, re, shutil, subprocess, sys, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_curve, roc_auc_score,
    confusion_matrix, accuracy_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

AUDIO_EXT={'.wav','.flac','.mp3','.ogg','.m4a','.aac'}
CATEGORIES=['noise','compression','reverberation','lowpass']
LEVELS={
    'noise':{1:20.,2:10.,3:0.},
    'compression':{1:64,2:32,3:16},
    'reverberation':{1:0.2,2:0.5,3:0.9},
    'lowpass':{1:7000.,2:5000.,3:3400.},
}
EXTRA_LEVELS={
    'noise':{4:15.,5:5.},
    'compression':{4:48,5:24},
    'reverberation':{4:0.35,5:0.7},
    'lowpass':{4:6000.,5:4200.},
}
# NEW PROTOCOL: five severity levels ordered from light to severe.
# V1_LEGACY levels 1..3 are preserved unchanged for reproducibility.
FIVE_LEVELS={
    'noise':{1:20.,2:15.,3:10.,4:5.,5:0.},
    'compression':{1:64,2:48,3:32,4:24,5:16},
    'reverberation':{1:0.2,2:0.35,3:0.5,4:0.7,5:0.9},
    'lowpass':{1:7000.,2:6000.,3:5000.,4:4200.,5:3400.},
}
VALID_SCENARIO_PROFILES=('v1_3level','v2_5level')

def available_levels(profile):
    if profile=='v1_3level': return (1,2,3)
    if profile=='v2_5level': return (1,2,3,4,5)
    raise ValueError(f'Unknown scenario_profile={profile}; choose {VALID_SCENARIO_PROFILES}')

def scenario_severity(category, level, profile='v1_3level'):
    if category=='baseline': return None
    if profile=='v2_5level': return FIVE_LEVELS[category][int(level)]
    if int(level) in LEVELS[category]: return LEVELS[category][int(level)]
    if int(level) in EXTRA_LEVELS[category]: return EXTRA_LEVELS[category][int(level)]
    raise ValueError((category,level,profile))

@dataclass
class Config:
    root: str='/home/jovyan/work/audio_real_fake/AUA'
    out: str='/home/jovyan/work/audio_real_fake/results/SAADE_AUA_2026_v3_1_metadata_fixed'
    seed: int=42
    sample_rate: int=16000
    seconds: float=4.0
    min_seconds: float=2.0
    max_items: int=400  # 0=all available, equal counts per label when limiting
    max_per_group: int=30
    epochs: int=4
    batch_size: int=12
    lr: float=0.0005
    num_workers: int=0
    augmentation_probability: float=0.5
    use_cache: bool=True
    max_ambiguous_label_fraction: float=0.05
    allow_weak_groups: bool=False  # explicit override for exploratory only
    models: tuple=('spectrogram_cnn','raw_wave_cnn','mfcc_svm')
    repo_asvspoof5: str='/home/jovyan/work/asvspoof5'
    repo_aasist: str='/home/jovyan/work/aasist'
    wavlm_checkpoint: str='microsoft/wavlm-base-plus'
    bootstraps: int=200
    scenario_seed: int=1001   # fixed across training seeds; prevents comparison confounding
    scenario_cache_dir: str=''  # optional shared immutable cache across seed repeats
    scenario_profile: str='v1_3level' # v1_3level=13; v2_5level=21 ordered scenarios
    test_finalization: bool=False # frozen test locked and one-time evaluation gate

    def __post_init__(self):
        # Explicit safety invariant: this experiment never writes into AUA or Melnyk's workspace.
        output=Path(self.out).expanduser().resolve()
        dataset=Path(self.root).expanduser().resolve()
        if output == dataset or dataset in output.parents:
            raise ValueError('OUT may not be inside the read-only AUA data folder')
        if any('melnyk' in part.casefold() for part in output.parts):
            raise ValueError('Writing to Melnyk_dus or any Melnyk folder is prohibited')


def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def normalize_path_key(s):
    return str(s).replace('\\','/').strip().lstrip('./')


def getval(row, *names):
    for name in names:
        v=row.get(name,'')
        if v is not None and not (isinstance(v,float) and np.isnan(v)):
            s=str(v).strip()
            if s and s.lower() not in ('nan','none','null'):
                return s
    return ''


def source_from_path(s):
    ss=str(s).lower().replace('\\','/')
    if '/own_processed/' in ss: return 'AUA_own'
    if '/common_voice/' in ss: return 'CommonVoice'
    if '/fleurs/' in ss: return 'FLEURS'
    if '/new-tts/' in ss: return 'AUA_new_TTS'
    if '/drssu/' in ss: return 'DRSSU'
    if '/synthetic/' in ss: return 'synthetic_other'
    return 'external_other'


def safe_key(s):
    return re.sub(r'[^\w.-]+','_',str(s), flags=re.UNICODE)[:160]


def _read_csv(path):
    if not Path(path).exists(): return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def build_catalog(cfg:Config):
    """Build conservative catalog from the 01_audio directory and the REAL AUA metadata.
    Never use 05_splits for sample counts. Suspect provenance/labels are flagged, not silently verified.
    """
    root=Path(cfg.root).expanduser().resolve()
    if not (root/'01_audio').is_dir():
        raise FileNotFoundError(f'Cannot find {root}/01_audio. Set cfg.root correctly.')
    aud=root/'01_audio'
    meta=_read_csv(aud/'metadata_audio.csv')
    own=_read_csv(aud/'own_processed'/'metadata_audio.csv')
    newtts=_read_csv(aud/'synthetic'/'new-tts'/'metadata_new-tts.csv')
    own_regions=_read_csv(aud/'own_master'/'metadata_audio.csv')
    master_by_speaker={}
    if not own_regions.empty:
        for _,rr in own_regions.iterrows():
            sid=getval(rr,'speaker_id')
            if sid: master_by_speaker.setdefault(sid,rr.to_dict())
    canonical_dirs=[aud/'own_processed',aud/'external_selected',aud/'synthetic']
    paths=[]
    for folder in canonical_dirs:
        if folder.exists():
            paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    paths=sorted(set(paths))
    byname=defaultdict(list)
    for p in paths: byname[p.name].append(p)
    meta_by_path, meta_by_name={}, defaultdict(list)
    for _,rr in meta.iterrows():
        row=rr.to_dict()
        key=getval(row,'working_path')
        if key:
            meta_by_path[normalize_path_key(key)]=row
        name=getval(row,'file_name')
        if name: meta_by_name[Path(name).name].append(row)
    own_by_name={Path(getval(r.to_dict(),'working_path','file_name')).name:r.to_dict() for _,r in own.iterrows()}
    tts_by_name={Path(getval(r.to_dict(),'filename')).name:r.to_dict() for _,r in newtts.iterrows()}
    records=[]
    for p in paths:
        rp=p.relative_to(root).as_posix(); r=None
        for candidate in (str(p),rp):
            if normalize_path_key(candidate) in meta_by_path:
                r=meta_by_path[normalize_path_key(candidate)];break
        if r is None and len(meta_by_name.get(p.name,[]))==1:
            r=meta_by_name[p.name][0]
        r=r or {}; ownr=own_by_name.get(p.name,{}) if '/own_processed/' in '/'+rp else {}
        tr=tts_by_name.get(p.name,{}) if '/new-tts/' in '/'+rp else {}
        src=source_from_path(rp)
        # A named corpus can legitimately contain BOTH real and fake examples.
        # Class comes from canonical folder + explicit mode, NOT the corpus name.
        declared_source=getval(r,'source_dataset').lower()
        if 'drssu' in declared_source or '/drssu/' in '/'+rp.lower():src='DRSSU'
        elif 'fleurs' in declared_source:src='FLEURS'
        elif 'common voice' in declared_source or 'common_voice' in declared_source:src='CommonVoice'
        source_label=1 if '/synthetic/' in '/'+rp.replace('\\','/').lower() else 0
        mode=getval(r,'mode').lower()
        explicit_label=None
        if mode in ('real','bona fide','bona_fide','bonafide','genuine','human'):
            explicit_label=0
        elif mode in ('fake','spoof','synthetic','tts','vc','voice_conversion','generated'):
            explicit_label=1
        conflict=explicit_label is not None and explicit_label!=source_label
        label=source_label if not conflict else None
        sp=getval(ownr,'speaker_id') or getval(r,'speaker_id')
        basis=getval(r,'speaker_id_basis')
        srid=getval(ownr,'source_record_id') or getval(r,'source_record_id')
        shash=getval(ownr,'source_sha256') or getval(r,'source_sha256')
        voice=getval(tr,'voice_id')
        generator=getval(tr,'model_name')
        if src=='DRSSU' and label==1: generator='DRSSU_generator_unknown'  # may be refined from source metadata
        if not generator and label==1: generator=getval(r,'source_dataset') or src
        if sp:
            group=f'SPEAKER::{src}::{sp}'
            group_confidence='metadata_speaker'
        elif voice:
            group=f'VOICE::{src}::{generator}::{voice}'
            group_confidence='metadata_voice'
        elif shash:
            group=f'SOURCEHASH::{shash}'
            group_confidence='source_sha256'
        elif srid:
            group=f'SOURCE_RECORD::{src}::{srid}'
            group_confidence='source_record_id'
        else:
            group=f'UNVERIFIED_FILE::{rp}'
            group_confidence='weak_file_fallback'
        region=master_by_speaker.get(sp,{}) if src=='AUA_own' else {}
        transcript=getval(ownr,'transcription') or getval(r,'transcription')
        records.append(dict(path=str(p),relative_path=rp,source=src,source_dataset=getval(r,'source_dataset') or src,
             label=label,label_source='folder_and_metadata' if explicit_label is not None and not conflict else 'folder_inferred',
             label_conflict=conflict,speaker_id=sp,speaker_id_basis=basis,source_record_id=srid,
             source_sha256=shash,working_sha256=getval(ownr,'working_sha256') or getval(r,'working_sha256'),
             group_id=group,group_confidence=group_confidence,voice_id=voice,generator=generator,
             transcription=transcript,script_id=getval(ownr,'script_id'),
             speaker_origin_group=getval(region,'speaker_origin_group'),
             script_profile=getval(region,'script_profile'),
             dialect_authenticity=getval(region,'dialect_authenticity'),
             approval_status=getval(ownr,'approval_status') or getval(r,'approval_status') or getval(tr,'approval_status'),
             license=getval(r,'license') or getval(tr,'license_terms')))
    df=pd.DataFrame(records)
    if df.empty: raise ValueError('No canonical media found in 01_audio.')
    # Duration: prefer reliable WAV header; for other codecs use soundfile info or ffprobe
    df['duration_sec']=[audio_info_seconds(Path(p)) for p in df.path]
    df['usable']=(df.label.notna())&(df.duration_sec>=cfg.min_seconds)&(~df.label_conflict)
    df['warning']=''
    df.loc[df.label_conflict,'warning']='CONFLICT: metadata mode versus folder label'
    df.loc[df.duration_sec.isna(),'warning']='unreadable audio'
    df.loc[df.group_confidence.eq('weak_file_fallback'),'warning']='WEAK GROUP: potential speaker/content leakage'
    dups=df[df.working_sha256.ne('') & df.working_sha256.duplicated(keep=False)]
    if len(dups):
        df.loc[dups.index,'warning']=df.loc[dups.index,'warning'].astype(str)+'; duplicate working SHA'
    # dedup repeated file hashes deterministically
    df.loc[df.working_sha256.ne('') & df.working_sha256.duplicated(keep='first'),'usable']=False
    out=Path(cfg.out);out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'01_catalog_all.csv',index=False)
    issues=df[~df.usable | df.warning.ne('')]
    issues.to_csv(out/'01_catalog_issues.csv',index=False)
    summary=(df.groupby(['source','label'],dropna=False).agg(files=('path','size'),hours=('duration_sec',lambda x:x.sum()/3600),
         speakers=('speaker_id',lambda x: x[x.ne('')].nunique()),weak_groups=('group_confidence',lambda x: (x=='weak_file_fallback').sum())).reset_index())
    summary.to_csv(out/'01_sources_summary.csv',index=False)
    print('Catalog:',len(df),'files; usable:',int(df.usable.sum()),'hours usable:',round(df.loc[df.usable,'duration_sec'].sum()/3600,3))
    print(summary.to_string(index=False))
    if df.label_conflict.any(): print('WARNING: conflicting labels:',int(df.label_conflict.sum()))
    if (df.group_confidence=='weak_file_fallback').any(): print('WARNING: unverified grouping:',int((df.group_confidence=='weak_file_fallback').sum()))
    if not (out/'01_catalog_all.csv').exists():raise AssertionError('Catalog not saved')
    return df


def audio_info_seconds(path):
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            r=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nokey=1:noprint_wrappers=1',str(path)],capture_output=True,text=True,timeout=25)
            return float(r.stdout.strip()) if r.returncode==0 else np.nan
        except Exception:
            return np.nan


def get_experimental_subset(catalog:pd.DataFrame,cfg:Config):
    out=Path(cfg.out)
    d=catalog[catalog.usable & catalog.label.notna()].copy()
    d['label']=d.label.astype(int)
    if d.label.nunique()<2: raise ValueError('Both real=0 and fake=1 required after catalog filtering.')
    if cfg.max_items:
        rng=np.random.default_rng(cfg.seed)
        chosen=[]
        for lab in (0,1):
            x=d[d.label.eq(lab)].copy()
            n=int(cfg.max_items//2)
            if len(x)<n:
                warnings.warn(f'Only {len(x)} files for class {lab}; cannot reach {n}')
                n=len(x)
            # round-robin over source then group to cap dominance
            x=x.iloc[rng.permutation(len(x))]
            x['grank']=x.groupby('group_id').cumcount()
            x=x[x.grank<cfg.max_per_group]
            x['source_rank']=x.groupby('source').cumcount()
            x=x.sort_values(['source_rank','grank'])
            chosen.append(x.head(n).drop(columns=['grank','source_rank']))
        d=pd.concat(chosen,ignore_index=True)
    else:
        d=d.reset_index(drop=True)
    d.to_csv(out/'02_experimental_subset.csv',index=False)
    print('Subset',len(d),'label counts:',d.label.value_counts().to_dict(), 'group counts:',d.groupby('label').group_id.nunique().to_dict())
    return d


def make_splits(d:pd.DataFrame,cfg:Config):
    """Multiple group-aware candidates, never split one group between train/val/test.
    Attempts class balance. No reliance on possibly contaminated existing 05_splits.
    """
    if not cfg.allow_weak_groups and d.group_confidence.eq('weak_file_fallback').any():
        n=int(d.group_confidence.eq('weak_file_fallback').sum())
        raise ValueError(f'{n} samples have unverifiable speaker/source groups. Enrich metadata or set cfg.allow_weak_groups=True for EXPLORATORY run.')
    if min(d.groupby('label').group_id.nunique())<3:
        raise ValueError('Fewer than three independent groups in one label; cannot create safe train/val/test.')
    best=None;best_score=float('inf')
    X=np.zeros((len(d),1));groups=d.group_id.astype(str).to_numpy(); y=d.label.to_numpy()
    for i in range(150):
        try:
            first=GroupShuffleSplit(n_splits=1,test_size=.40,random_state=cfg.seed+i)
            tr,rest=next(first.split(X,y,groups))
            second=GroupShuffleSplit(n_splits=1,test_size=.50,random_state=cfg.seed+10000+i)
            va0,te0=next(second.split(X[rest],y[rest],groups[rest]))
            va,te=rest[va0],rest[te0]
        except Exception:continue
        if any(len(set(y[idx]))!=2 for idx in (tr,va,te)):continue
        counts=np.array([[np.sum(y[idx]==label) for label in (0,1)] for idx in (tr,va,te)],dtype=float)
        target=np.array([[.60,.60],[.20,.20],[.20,.20]])*np.bincount(y,minlength=2)
        penalty=float(np.square((counts-target)/np.maximum(1,target)).sum())
        if penalty<best_score:best=(tr,va,te);best_score=penalty
    if best is None:raise ValueError('Cannot produce class-complete group-disjoint splits. Check group size and metadata.')
    out=d.copy();out['split']=''
    for name,ix in zip(('train','validation','hidden_test'),best):out.iloc[ix,out.columns.get_loc('split')]=name
    groups_by={k:set(out[out.split.eq(k)].group_id) for k in ('train','validation','hidden_test')}
    for a in groups_by:
        for b in groups_by:
            if a!=b:assert not (groups_by[a]&groups_by[b]),f'Group leakage {a} {b}'
    # Exact same working hash must not straddle splits
    hashes=out[out.working_sha256.ne('')].groupby('working_sha256').split.nunique()
    if (hashes>1).any():raise AssertionError('Duplicate working SHA across splits; repair source groups.')
    # metadata-based source original links if known
    src=out[out.source_sha256.ne('')].groupby('source_sha256').split.nunique()
    if (src>1).any():raise AssertionError('Source SHA across splits; group by source_id or original hash.')
    out.to_csv(Path(cfg.out)/'03_locked_splits.csv',index=False)
    metrics=out.groupby(['split','label']).agg(n=('path','size'),groups=('group_id','nunique'),sources=('source','nunique')).reset_index()
    metrics.to_csv(Path(cfg.out)/'03_split_summary.csv',index=False)
    print(metrics.to_string(index=False))
    print('Group and exact hash checks: PASS; score',round(best_score,4))
    if len(set(out.source))>1:
        print('WARNING: class is confounded with corpus source. Report source-stratified results; see README.')
    return out


def frozen_hash(df):
    cols=['relative_path','split','label','group_id','source','speaker_id','working_sha256']
    cols=[*cols,*[c for c in ('segment_start_sec','parent_file_id') if c in df.columns]]
    s=df[cols].sort_values('relative_path').to_csv(index=False)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def lock_hidden_test(df,cfg):
    out=Path(cfg.out); p=out/'03_SPLIT_LOCK.json';sha=frozen_hash(df)
    if p.exists():
        old=json.loads(p.read_text())
        if old['split_manifest_sha256']!=sha:raise RuntimeError('SPLITS CHANGED since first lock. New experiment ID required; do not reuse hidden test.')
    else:save_json(p,dict(split_manifest_sha256=sha,seed=cfg.seed,
      hidden_samples=int(df.split.eq('hidden_test').sum()),meaning='Freeze BEFORE adaptive validation; do not select or tune on test.'))
    print('Locked split SHA:',sha)
    return sha


@lru_cache(maxsize=512)
def read_wave(path,sr=16000,seconds=4.0,seed=0,train=False):
    import soundfile as sf
    p=Path(path)
    try:
        x,old_sr=sf.read(str(p),dtype='float32',always_2d=False)
    except Exception:
        # decoder fallback only; no mutation of source files
        cmd=['ffmpeg','-v','error','-i',str(p),'-ar',str(sr),'-ac','1','-f','f32le','pipe:1']
        r=subprocess.run(cmd,capture_output=True,timeout=45)
        if r.returncode: raise RuntimeError(f'ffmpeg cannot decode {p}: {r.stderr[:150]}')
        x=np.frombuffer(r.stdout,dtype='<f4').copy();old_sr=sr
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==2:x=x.mean(axis=1)
    if old_sr!=sr:
        from math import gcd
        g=gcd(old_sr,sr);x=signal.resample_poly(x,sr//g,old_sr//g).astype('float32')
    if not np.isfinite(x).all():raise ValueError('NaN/Inf in decoded waveform '+str(p))
    need=int(sr*seconds)
    if len(x)>=need:
        off=int(np.random.default_rng(seed).integers(0,len(x)-need+1)) if train else (len(x)-need)//2
        x=x[off:off+need]
    else:x=np.pad(x,(0,need-len(x)))
    return np.ascontiguousarray(x,dtype=np.float32)


# V2_SEGMENT_PATCH: deterministic, non-overlapping fixed offset windows.
def read_fixed_segment(path, sr=16000, seconds=4.0, start_sec=None, seed=0, train=False):
    """Read exact window if start_sec is supplied; else legacy center/random crop.
    Segment windows must be assigned to a split via their parent BEFORE expansion.
    """
    if start_sec is None or (isinstance(start_sec, float) and np.isnan(start_sec)):
        return read_wave(str(path),sr,seconds,seed,train)
    import soundfile as sf
    start_sec=float(start_sec)
    if not np.isfinite(start_sec) or start_sec<0:raise ValueError('Invalid segment offset')
    length=int(round(seconds*sr))
    try:
        with sf.SoundFile(str(path)) as handle:
            old_sr=int(handle.samplerate)
            handle.seek(int(round(start_sec*old_sr)))
            x=handle.read(int(np.ceil(seconds*old_sr))+3,dtype='float32',always_2d=False)
    except Exception:
        cmd=['ffmpeg','-v','error','-ss',str(start_sec),'-i',str(path),'-t',str(seconds+0.05),
             '-ar',str(sr),'-ac','1','-f','f32le','pipe:1']
        run=subprocess.run(cmd,capture_output=True,timeout=50)
        if run.returncode:raise RuntimeError('Unable to decode segment: '+str(path))
        x=np.frombuffer(run.stdout,dtype='<f4').copy();old_sr=sr
    x=np.asarray(x,dtype=np.float32)
    if x.ndim>1:x=x.mean(axis=1)
    if old_sr!=sr:
        from math import gcd
        g=gcd(old_sr,sr);x=signal.resample_poly(x,sr//g,old_sr//g).astype('float32')
    if len(x)>=length:x=x[:length]
    else:x=np.pad(x,(0,length-len(x)))
    if not np.isfinite(x).all():raise ValueError('Nonfinite signal: '+str(path))
    return np.ascontiguousarray(x,dtype=np.float32)


def _ffmpeg_compress(x,sr,kbps):
    if not shutil.which('ffmpeg'):raise FileNotFoundError('ffmpeg required for true MP3 compression scenarios. Install ffmpeg; no silent fake compression.')
    enc=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','f32le','-ar',str(sr),'-ac','1','-i','pipe:0',
        '-codec:a','libmp3lame','-b:a',f'{int(kbps)}k','-f','mp3','pipe:1'],input=np.asarray(x,dtype='<f4').tobytes(),capture_output=True,timeout=40)
    if enc.returncode:raise RuntimeError('MP3 encoding error: '+enc.stderr.decode(errors='replace')[:300])
    dec=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','mp3','-i','pipe:0','-f','f32le','-ar',str(sr),'-ac','1','pipe:1'],input=enc.stdout,capture_output=True,timeout=40)
    if dec.returncode:raise RuntimeError('MP3 decoding error: '+dec.stderr.decode(errors='replace')[:300])
    out=np.frombuffer(dec.stdout,dtype='<f4').copy()
    if len(out)>=len(x):out=out[:len(x)]
    else:out=np.pad(out,(0,len(x)-len(out)))
    return out.astype('float32')


def transform(x,category,level,sr=16000,seed=42,severity_override=None):
    """Deterministic perturbation; class label remains unchanged."""
    if category=='baseline':return np.asarray(x,dtype=np.float32).copy()
    table=LEVELS if level in (1,2,3) else EXTRA_LEVELS
    if category not in table or level not in table[category]:raise ValueError((category,level))
    severity=table[category][level] if severity_override is None else severity_override
    x=np.asarray(x,dtype='float32');rng=np.random.default_rng(seed)
    if category=='noise':
        noise=rng.normal(0,1,len(x)).astype('float32')
        power=max(1e-7,float(np.mean(x*x)));npow=max(1e-7,float(np.mean(noise*noise)))
        result=x+noise*np.sqrt(power/(npow*10**(severity/10)))
    elif category=='compression':result=_ffmpeg_compress(x,sr,severity)
    elif category=='reverberation':
        # EXPLICIT: synthetic, NOT a measured physical RIR; RT60 is a nominal decay parameter.
        rt60=float(severity)
        n=min(int(sr*1.2),len(x));tt=np.arange(n)/sr
        rir=(rng.normal(size=n)*np.exp(-6.9078*tt/rt60)).astype('float32')
        rir[0]=max(2.0,float(np.max(np.abs(rir))))
        rir/=max(1e-8,float(np.sum(np.abs(rir))))
        result=signal.fftconvolve(x,rir,mode='full')[:len(x)].astype('float32')
        # preserve perceived level, avoid globally normalizing fake/noise too much
        rms0=np.sqrt(np.mean(x*x)+1e-10);rms1=np.sqrt(np.mean(result*result)+1e-10)
        result=result*(rms0/max(rms1,1e-8))
    elif category=='lowpass':
        cutoff=min(float(severity),sr*.49)
        sos=signal.butter(6,cutoff,fs=sr,btype='lowpass',output='sos')
        result=signal.sosfilt(sos,x)
    else:raise KeyError(category)
    return np.nan_to_num(result).astype('float32')


def deterministic_seed(*parts):
    s='|'.join(str(a) for a in parts).encode('utf-8');return int(hashlib.blake2b(s,digest_size=4).hexdigest(),16)


def scenario_manifest(df,split='validation',extra=False,profile='v1_3level'):
    d=df[df.split.eq(split)]
    rows=[]
    for r in d.itertuples():
        common=dict(path=r.path,relative_path=r.relative_path,group_id=r.group_id,speaker_id=r.speaker_id,
                    segment_start_sec=float(getattr(r,'segment_start_sec',float('nan'))),
                    parent_file_id=str(getattr(r,'parent_file_id','')),
                    label=int(r.label),source=r.source,generator=r.generator,split=split)
        rows.append(dict(**common,category='baseline',level=0,value=np.nan,scenario_id=f'{r.relative_path}::baseline::0'))
        for c in CATEGORIES:
            levels=(list(FIVE_LEVELS[c].items()) if profile=='v2_5level' else
                    list(LEVELS[c].items())+(list(EXTRA_LEVELS[c].items()) if extra else []))
            for lev,value in levels:
                rows.append(dict(**common,category=c,level=lev,value=value,scenario_id=f'{r.relative_path}::{c}::{lev}'))
    return pd.DataFrame(rows)


def mel_filter(sr=16000,n_fft=256,n_mels=40):
    def hz_to_mel(f):return 2595*np.log10(1+f/700)
    def mel_to_hz(m):return 700*(10**(m/2595)-1)
    hz=mel_to_hz(np.linspace(hz_to_mel(0),hz_to_mel(sr/2),n_mels+2))
    bins=hz/(sr/2)*(n_fft//2)
    f=np.arange(n_fft//2+1);out=np.zeros((n_mels,len(f)),dtype='float32')
    for i in range(n_mels):
        l,c,r=bins[i:i+3]
        out[i]=np.maximum(0,np.minimum((f-l)/max(1e-8,c-l),(r-f)/max(1e-8,r-c)))
    return out


def mfcc_features(wave,sr=16000):
    wave=np.asarray(wave,dtype='float32')
    _,_,z=signal.stft(wave,fs=sr,nperseg=512,noverlap=352,boundary='zeros')
    power=np.abs(z)**2; fb=mel_filter(sr,512,40)
    m=np.log(np.maximum(1e-8,fb@power)); cc=dct(m,type=2,norm='ortho',axis=0)[:20]
    delta=np.diff(cc,axis=1,prepend=cc[:,0:1])
    return np.concatenate([cc.mean(axis=1),cc.std(axis=1),delta.mean(axis=1),delta.std(axis=1)]).astype('float32')


def metric_row(y,scores,threshold=None):
    y=np.asarray(y,dtype=int);scores=np.asarray(scores,dtype=float)
    if threshold is None:threshold=0.0
    pred=(scores>=threshold).astype(int)
    out=dict(n=len(y),real=int(sum(y==0)),fake=int(sum(y==1)),threshold=float(threshold),
            balanced_accuracy=float(balanced_accuracy_score(y,pred)),accuracy=float(accuracy_score(y,pred)),
            f1_fake=float(f1_score(y,pred,zero_division=0)))
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    out.update(false_alarm_rate=float(fp/max(1,tn+fp)),miss_rate=float(fn/max(1,tp+fn)))
    if len(set(y))==2:
        fpr,tpr,ths=roc_curve(y,scores);fnr=1-tpr;idx=np.nanargmin(abs(fnr-fpr))
        out['eer']=float((fpr[idx]+fnr[idx])/2);out['roc_auc']=float(roc_auc_score(y,scores))
    else:out.update(eer=np.nan,roc_auc=np.nan)
    return out


def best_threshold(y,scores):
    y=np.asarray(y).astype(int);scores=np.asarray(scores,float)
    if len(np.unique(y))<2:raise ValueError('Cannot calibrate threshold with one class')
    fpr,tpr,ths=roc_curve(y,scores)
    idx=int(np.nanargmax(tpr-fpr))
    return float(ths[idx])


def asri_table(preds,expected_levels=3):
    """Scenario-balanced ASRI: average over four perturbation families x requested severities.
    Scale max(0, 2*BA-1). Missing cells => NaN, not a fake passing score.
    """
    levels=tuple(range(1,int(expected_levels)+1))
    x=preds[preds.category.isin(CATEGORIES)&preds.level.isin(levels)]
    rows=[]
    for key,g in x.groupby(['model','regime']):
        cell=[]
        for c in CATEGORIES:
            for level in levels:
                gg=g[(g.category==c)&(g.level==level)]
                if gg.empty or gg.label.nunique()!=2:continue
                mm=metric_row(gg.label,gg.score,float(gg.threshold.iloc[0]))
                cell.append((c,level,max(0.,2*mm['balanced_accuracy']-1)))
        complete=len(cell)==4*len(levels)
        value=float(np.mean([j[2] for j in cell])) if complete else np.nan
        rows.append(dict(model=key[0],regime=key[1],asri=value,complete_cells=len(cell),required_cells=4*len(levels),scenario_profile=('v2_5level' if len(levels)==5 else 'v1_3level')))
    return pd.DataFrame(rows)


def get_scenario_wave(r,cfg):
    '''Deterministic disk cache for costly MP3/reverb scenario waveforms.
    Cache key includes source size and mtime so updated server files invalidate entries.
    Original AUA files remain untouched.'''
    path=Path(r.path)
    if cfg.use_cache:
        st=path.stat()
        payload='|'.join(map(str,[r.relative_path,st.st_size,st.st_mtime_ns,r.category,r.level,
                                  cfg.scenario_seed,cfg.seconds,cfg.sample_rate,cfg.scenario_profile,getattr(r,'segment_start_sec',None)]))
        digest=hashlib.sha256(payload.encode()).hexdigest()
        folder=Path(cfg.scenario_cache_dir or cfg.out)/'cache_scenario_waves'/digest[:2]
        cached=folder/(digest+'.npy')
        if cached.exists():
            wav=np.load(cached,allow_pickle=False)
            if wav.ndim==1 and len(wav)==int(cfg.seconds*cfg.sample_rate):return wav
    wav=read_fixed_segment(r.path,cfg.sample_rate,cfg.seconds,
            start_sec=getattr(r,'segment_start_sec',None),seed=cfg.scenario_seed)
    seed=deterministic_seed(cfg.scenario_seed,r.relative_path,r.category,r.level,cfg.scenario_profile)
    if r.category!='baseline':
        severity=scenario_severity(r.category,r.level,cfg.scenario_profile)
        wav=transform(wav,r.category,r.level,cfg.sample_rate,seed,severity_override=severity)
    if cfg.use_cache:
        folder.mkdir(parents=True,exist_ok=True)
        temp=folder/(digest+'.partial.npy')
        np.save(temp,wav)
        os.replace(temp,cached)
    return wav


def run_scenario_predictions(df,model,model_name,regime,cfg,threshold=0.0,split='validation',extra=False):
    scen=scenario_manifest(df,split,extra=extra,profile=cfg.scenario_profile)
    rows=[]
    for i,r in enumerate(scen.itertuples()):
        wav=get_scenario_wave(r,cfg)
        score=float(model_score(model,model_name,wav,cfg))
        rows.append(dict(scenario_id=r.scenario_id,path=r.path,relative_path=r.relative_path,
            group_id=r.group_id,source=r.source,generator=r.generator,speaker_id=r.speaker_id,
            label=r.label,category=r.category,level=r.level,value=r.value,split=split,
            segment_start_sec=getattr(r,'segment_start_sec',None),
            parent_file_id=getattr(r,'parent_file_id',''),
            model=model_name,regime=regime,score=score,threshold=float(threshold),
            pred=int(score>=threshold)))
        if (i+1)%500==0:print('  scenarios scored:',i+1,'/',len(scen))
    return pd.DataFrame(rows)


def summarize_predictions(preds,cfg,report_tag=None):
    summary=[]
    for (model,regime,category,level),g in preds.groupby(['model','regime','category','level']):
        if g.label.nunique()!=2:continue
        row=metric_row(g.label,g.score,float(g.threshold.iloc[0]))
        row.update(model=model,regime=regime,category=category,level=int(level))
        summary.append(row)
    s=pd.DataFrame(summary)
    if not s.empty:
        suffix=('_'+safe_key(report_tag)) if report_tag else ''
        s.to_csv(Path(cfg.out)/f'08_scenario_metrics{suffix}.csv',index=False)
        a=asri_table(preds,expected_levels=len(available_levels(cfg.scenario_profile)))
        a.to_csv(Path(cfg.out)/f'09_ASRI{suffix}.csv',index=False)
    else:a=pd.DataFrame()
    return s,a


def validation_difficulty(p):
    """Per-category mean validation classification error, balanced equally over classes & levels."""
    scenario_levels=sorted(set(p.loc[p.category.isin(CATEGORIES),'level'].astype(int))-{0})
    v=p[p.category.isin(CATEGORIES)&p.level.isin(scenario_levels)]
    weights={}
    for c in CATEGORIES:
        v1=v[v.category==c]
        if v1.empty: weights[c]=.25;continue
        losses=[]
        for lev in scenario_levels:
            g=v1[v1.level==lev]
            if g.empty:continue
            for lab in (0,1):
                g2=g[g.label==lab]
                if len(g2):losses.append(float((g2.pred!=g2.label).mean()))
        weights[c]=float(np.mean(losses)) if losses else .25
    # smoothing so every category retained; no test access
    a=np.array([weights[c]+.05 for c in CATEGORIES],float);a/=a.sum()
    return dict(zip(CATEGORIES,a.tolist()))


def paired_group_bootstrap_delta(preds,model,regime_a='random',regime_b='adaptive',B=200,seed=42):
    x=preds[(preds.model==model)&(preds.regime.isin([regime_a,regime_b]))]
    a=x[x.regime==regime_a];b=x[x.regime==regime_b]
    key=['scenario_id','group_id','category','level','label']
    paired=a[key+['score','threshold']].merge(b[key+['score','threshold']],on=key,suffixes=('_a','_b'))
    if paired.empty:return dict(status='NO_PAIRED_DATA')
    group_ids=paired.group_id.unique();rng=np.random.default_rng(seed)
    if len(group_ids)<3:return dict(status='TOO_FEW_GROUPS',n_groups=len(group_ids))
    scenario_levels=sorted(set(paired.loc[paired.category.isin(CATEGORIES),'level'].astype(int))-{0})
    def mean_delta(sub):
        sub=sub[sub.category.isin(CATEGORIES)&sub.level.isin(scenario_levels)]
        deltas=[]
        for c in CATEGORIES:
            for lev in scenario_levels:
                g=sub[(sub.category==c)&(sub.level==lev)]
                if g.empty or g.label.nunique()<2:continue
                v1=balanced_accuracy_score(g.label,(g.score_a>=g.threshold_a).astype(int))
                v2=balanced_accuracy_score(g.label,(g.score_b>=g.threshold_b).astype(int))
                deltas.append(max(0,2*v2-1)-max(0,2*v1-1))
        return float(np.mean(deltas)) if len(deltas)==4*len(scenario_levels) else np.nan
    observed=mean_delta(paired);boots=[]
    grouped={g:paired[paired.group_id==g] for g in group_ids}
    for _ in range(B):
        sampled=rng.choice(group_ids,size=len(group_ids),replace=True)
        sub=pd.concat([grouped[g] for g in sampled],ignore_index=True)
        v=mean_delta(sub)
        if np.isfinite(v):boots.append(v)
    if len(boots)<max(20,B//10):return dict(status='INSUFFICIENT_BOOTSTRAPS',n_valid=len(boots),observed=observed)
    return dict(status='OK',model=model,delta_asri=observed,ci95_lower=float(np.quantile(boots,.025)),
        ci95_upper=float(np.quantile(boots,.975)),n_valid=len(boots),n_groups=len(group_ids))


def render_charts(metric_table,asri_table_df,cfg):
    import matplotlib.pyplot as plt
    out=Path(cfg.out)
    if not metric_table.empty:
        x=metric_table[metric_table.category.isin(CATEGORIES)].copy()
        for (model,regime),g in x.groupby(['model','regime']):
            pv=g.pivot(index='category',columns='level',values='balanced_accuracy').reindex(CATEGORIES)
            fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(pv.values,vmin=0,vmax=1,cmap='viridis')
            ax.set_xticks(range(len(pv.columns)),pv.columns);ax.set_yticks(range(len(pv.index)),pv.index)
            ax.set_xlabel('Severity (1–3)');ax.set_title(f'{model}: {regime} — Balanced Accuracy')
            for i in range(len(pv.index)):
                for j in range(len(pv.columns)):
                    v=pv.iloc[i,j]
                    if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<.5 else 'black')
            fig.colorbar(im,ax=ax,label='Balanced Accuracy');fig.tight_layout()
            fig.savefig(out/f'chart_heatmap_{safe_key(model)}_{safe_key(regime)}.png',dpi=160);plt.close(fig)
    if not asri_table_df.empty:
        z=asri_table_df.pivot(index='model',columns='regime',values='asri')
        fig,ax=plt.subplots(figsize=(8,4));z.plot(kind='bar',ax=ax,ylim=(0,1),rot=0)
        ax.set_title('ASRI: scenario-balanced robustness');ax.set_ylabel('ASRI [0,1]')
        fig.tight_layout();fig.savefig(out/'chart_ASRI.png',dpi=170);plt.close(fig)


def zip_results(cfg,include_models=False):
    from zipfile import ZipFile,ZIP_DEFLATED
    out=Path(cfg.out); destination=out/'SAADE_RESULTS.zip'
    with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in out.rglob('*'):
            if not p.is_file() or p==destination:continue
            if p.suffix.lower() in ('.csv','.json','.md','.txt','.png','.svg','.log') or (include_models and p.suffix=='.pt'):
                z.write(p,p.relative_to(out))
    print('Report ZIP:',destination,'bytes:',destination.stat().st_size)
    return destination

# ==========================================================
# Trainable models and evaluation adapters
# ==========================================================

def seed_everything(seed):
    import torch
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def _nn_components():
    import torch
    import torch.nn as nn
    return torch,nn


def create_model(name,cfg):
    """Actual named models. Lightweight baselines are NEVER presented as AASIST/RawNet2.
    Optional official AASIST/RawNet2 need downloaded source checkout.
    """
    if name=='mfcc_svm':
        return make_pipeline(StandardScaler(),SVC(C=1.0,kernel='rbf',class_weight='balanced'))
    torch,nn=_nn_components()
    if name=='spectrogram_cnn':
        class SpectrogramCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('window',torch.hann_window(512))
                self.encoder=nn.Sequential(
                    nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                    nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                    nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))
                self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.2),nn.Linear(64,1))
            def forward(self,x):
                z=torch.stft(x,n_fft=512,hop_length=256,window=self.window,return_complex=True)
                spec=torch.log1p(z.abs().square()).unsqueeze(1)
                return self.head(self.encoder(spec)).view(-1)
        return SpectrogramCNN()
    if name=='raw_wave_cnn':
        class RawWaveCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=nn.Sequential(
                    nn.Conv1d(1,16,101,stride=10,padding=50),nn.BatchNorm1d(16),nn.LeakyReLU(.1),
                    nn.Conv1d(16,32,11,stride=4,padding=5),nn.BatchNorm1d(32),nn.LeakyReLU(.1),
                    nn.Conv1d(32,64,7,stride=4,padding=3),nn.BatchNorm1d(64),nn.LeakyReLU(.1),
                    nn.Conv1d(64,128,5,stride=4,padding=2),nn.BatchNorm1d(128),nn.LeakyReLU(.1),
                    nn.AdaptiveAvgPool1d(1))
                self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128,1))
            def forward(self,x):return self.head(self.layers(x.unsqueeze(1))).view(-1)
        return RawWaveCNN()
    if name=='aasist':
        p=Path(cfg.repo_aasist).expanduser().resolve()
        modfile=p/'models'/'AASIST.py'
        config=p/'config'/'AASIST.conf'
        if not modfile.exists() or not config.exists():
            raise FileNotFoundError('Official AASIST not installed. Clone https://github.com/clovaai/aasist into cfg.repo_aasist; see README.')
        spec=importlib.util.spec_from_file_location('_saade_official_aasist',str(modfile))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        d_args=json.loads(config.read_text())['model_config']
        backbone=module.Model(d_args)
        class OfficialAASIST(nn.Module):
            def __init__(self,backbone):super().__init__();self.backbone=backbone
            def forward(self,x):
                logits=self.backbone(x)
                if isinstance(logits,tuple):logits=logits[-1]
                if logits.ndim!=2 or logits.shape[1]!=2:raise ValueError('Unexpected official AASIST output shape')
                return logits[:,1]-logits[:,0]  # trained against fake=1 in SAADE
        return OfficialAASIST(backbone)
    if name=='rawnet2':
        p=Path(cfg.repo_asvspoof5).expanduser().resolve()/'Baseline-RawNet2'/'model.py'
        if not p.exists():raise FileNotFoundError('Official RawNet2 missing. Clone https://github.com/asvspoof-challenge/asvspoof5 into cfg.repo_asvspoof5.')
        spec=importlib.util.spec_from_file_location('_saade_official_rawnet2',str(p))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        backbone=module.RawNet2(device)
        class OfficialRawNet2(nn.Module):
            def __init__(self,backbone):super().__init__();self.backbone=backbone
            def forward(self,x):
                logits=self.backbone(x)
                if isinstance(logits,tuple):logits=logits[-1]
                if logits.ndim!=2 or logits.shape[1]!=2:raise ValueError('Unexpected official RawNet2 output shape')
                return logits[:,1]-logits[:,0]  # trained against fake=1 in SAADE
        return OfficialRawNet2(backbone)
    if name=='wavlm':
        try:
            from transformers import WavLMModel
        except ImportError as e:
            raise ImportError('Install transformers compatible with your PyTorch version; see README.') from e
        backbone=WavLMModel.from_pretrained(cfg.wavlm_checkpoint)
        for p in backbone.parameters():p.requires_grad=False
        class FrozenWavLMHead(nn.Module):
            def __init__(self,backbone):
                super().__init__();self.backbone=backbone
                self.head=nn.Sequential(nn.LayerNorm(backbone.config.hidden_size),nn.Linear(backbone.config.hidden_size,1))
            def forward(self,x):
                self.backbone.eval()
                with torch.no_grad():rep=self.backbone(input_values=x).last_hidden_state.mean(dim=1)
                return self.head(rep).view(-1)
            def train(self,mode=True):
                super().train(mode);self.backbone.eval();return self
        return FrozenWavLMHead(backbone)
    raise KeyError(f'Unknown model name: {name}')


class AudioDataset:
    def __init__(self,records,cfg,train=False,regime='static',adaptive_weights=None):
        self.records=records.reset_index(drop=True)
        self.cfg=cfg;self.train=train;self.regime=regime;self.adaptive_weights=adaptive_weights
        self.epoch=0
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        row=self.records.iloc[i]
        seed=deterministic_seed(self.cfg.seed,self.epoch,i,row.relative_path)
        x=read_fixed_segment(row.path,self.cfg.sample_rate,self.cfg.seconds,
                start_sec=row.get('segment_start_sec',None),seed=seed,train=self.train)
        if self.train and self.regime!='static':
            rng=np.random.default_rng(seed+123)
            if rng.random()<self.cfg.augmentation_probability:
                if self.regime=='adaptive':
                    assert self.adaptive_weights is not None
                    weights=np.array([self.adaptive_weights[c] for c in CATEGORIES],float)
                    cat=str(rng.choice(CATEGORIES,p=weights/weights.sum()))
                else:cat=str(rng.choice(CATEGORIES))
                lev=int(rng.choice(available_levels(self.cfg.scenario_profile)))
                severity=scenario_severity(cat,lev,self.cfg.scenario_profile)
                x=transform(x,cat,lev,self.cfg.sample_rate,seed,severity_override=severity)
        return np.asarray(x,dtype='float32'),int(row.label)


def _batch_tensors(batch):
    import torch
    xs,ys=zip(*batch)
    return torch.from_numpy(np.stack(xs)),torch.tensor(ys,dtype=torch.float32)


def get_validation_scores(model,model_name,df,cfg):
    scores=[];labels=[]
    for r in df.itertuples():
        w=read_fixed_segment(r.path,cfg.sample_rate,cfg.seconds,
             start_sec=getattr(r,'segment_start_sec',None),seed=cfg.scenario_seed,train=False)
        scores.append(float(model_score(model,model_name,w,cfg)))
        labels.append(int(r.label))
    return np.array(labels,int),np.array(scores,float)


def train_one(df,cfg,name,regime='static',adaptive_weights=None,force=False):
    """Train all models for SAME epoch count and seed per model x regime.
    Threshold determined strictly on CLEAN validation, hidden test not read.
    """
    out=Path(cfg.out)/'models';out.mkdir(parents=True,exist_ok=True)
    modelpath=out/f'{safe_key(name)}_{safe_key(regime)}.pt'
    metapath=out/f'{safe_key(name)}_{safe_key(regime)}_meta.json'
    signature=dict(split_hash=frozen_hash(df),epochs=cfg.epochs,seed=cfg.seed,
                   seconds=cfg.seconds,sample_rate=cfg.sample_rate,lr=cfg.lr,
                   batch_size=cfg.batch_size,augmentation_probability=cfg.augmentation_probability,
                   adaptive_weights=adaptive_weights,scenario_profile=cfg.scenario_profile,
                   scenario_seed=cfg.scenario_seed,model=name)
    if not force and modelpath.exists() and metapath.exists():
        meta=json.loads(metapath.read_text())
        if meta.get('training_signature')!=signature:
            raise RuntimeError('Existing checkpoint belongs to DIFFERENT data or hyperparameters: '
                +str(modelpath)+'. Use a NEW cfg.out directory (preferred), or force=True after deliberate review.')
        print('Reusing checkpoint:',name,regime)
        if name=='mfcc_svm':
            import joblib
            m=joblib.load(modelpath)
        else:
            import torch
            m=create_model(name,cfg)
            m.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True))
            m.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        return m,meta
    seed_everything(cfg.seed+deterministic_seed(name)%1000)
    tr=df[df.split.eq('train')].reset_index(drop=True)
    val=df[df.split.eq('validation')].reset_index(drop=True)
    if tr.label.nunique()!=2 or val.label.nunique()!=2:raise ValueError('Two classes required in training and validation.')
    model=create_model(name,cfg)
    if name=='mfcc_svm':
        dataset=AudioDataset(tr,cfg,train=True,regime=regime,adaptive_weights=adaptive_weights)
        xx=[];yy=[]
        for epoch in range(cfg.epochs):
            dataset.epoch=epoch
            for i in range(len(dataset)):
                wave,label=dataset[i];xx.append(mfcc_features(wave,cfg.sample_rate));yy.append(label)
        model.fit(np.stack(xx),yy)
        y,s=get_validation_scores(model,name,val,cfg)
        threshold=best_threshold(y,s);score=metric_row(y,s,threshold)
        import joblib
        joblib.dump(model,modelpath)
    else:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=model.to(device)
        ds=AudioDataset(tr,cfg,train=True,regime=regime,adaptive_weights=adaptive_weights)
        loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,
              collate_fn=_batch_tensors,drop_last=False)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr)
        n0=int((tr.label==0).sum());n1=int((tr.label==1).sum())
        pos=torch.tensor([n0/max(1,n1)],device=device,dtype=torch.float32)
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos)
        best_auc=-np.inf;best_state=None;best_metrics={};best_thr=0.0
        for epoch in range(cfg.epochs):
            ds.epoch=epoch
            model.train();losses=[]
            for batch_x,batch_y in loader:
                batch_x=batch_x.to(device);batch_y=batch_y.to(device)
                opt.zero_grad(set_to_none=True)
                output=model(batch_x).view(-1)
                loss=criterion(output,batch_y)
                if not torch.isfinite(loss):raise FloatingPointError('NaN training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
                losses.append(float(loss.detach().cpu()))
            y,s=get_validation_scores(model,name,val,cfg)
            threshold=best_threshold(y,s);scores=metric_row(y,s,threshold)
            print(f'  {name} {regime} epoch {epoch+1}/{cfg.epochs}: loss={np.mean(losses):.4f} val AUC={scores["roc_auc"]:.3f} EER={scores["eer"]:.3f}')
            if scores['roc_auc']>best_auc:
                best_auc=scores['roc_auc'];best_metrics=scores;best_thr=threshold
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);model.eval()
        torch.save(best_state,modelpath)
        score=best_metrics;threshold=best_thr
    meta=dict(name=name,regime=regime,seed=cfg.seed,threshold=float(threshold),
             clean_val_metrics=score,training_records=len(tr),val_records=len(val),
             augmentation_weights=adaptive_weights,epochs=cfg.epochs,training_signature=signature)
    save_json(metapath,meta)
    return model,meta


def model_score(model,name,wave,cfg):
    if name=='mfcc_svm':
        feature=mfcc_features(wave,cfg.sample_rate).reshape(1,-1)
        score=model.decision_function(feature)
        return float(np.asarray(score).ravel()[0])
    import torch
    device=next(model.parameters()).device
    x=torch.from_numpy(np.asarray(wave,dtype='float32')).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():s=model(x).view(-1)
    return float(s[0].detach().cpu())


def summarize_source_results(preds,cfg):
    lines=[]
    for (model,regime,split,source),g in preds.groupby(['model','regime','split','source']):
        m={'model':model,'regime':regime,'split':split,'source':source,'n':len(g),'class_count':g.label.nunique()}
        if g.label.nunique()==2:m.update(metric_row(g.label,g.score,g.threshold.iloc[0]))
        else:
            # Single-class detection rates are valid but not balanced accuracy or EER
            m['class_recall']=float((g.pred==g.label).mean())
            m['warning']='ONLY_ONE_CLASS_IN_SOURCE; EER/BA undefined'
        lines.append(m)
    res=pd.DataFrame(lines);res.to_csv(Path(cfg.out)/'10_source_strata.csv',index=False)
    return res


def ensure_test_gate(cfg):
    if not cfg.test_finalization:
        raise RuntimeError('Hidden test is locked. Set cfg.test_finalization=True ONLY AFTER model selection and scenario policy are frozen. No exploratory hidden-test tuning.')
    p=Path(cfg.out)/'03_SPLIT_LOCK.json'
    if not p.exists():raise RuntimeError('No frozen split manifest.')


def make_readme(cfg,df,validation_df=None,test_df=None):
    out=Path(cfg.out);t=[
        '# SAADE–AUA: experiment report',
        f'Root: `{cfg.root}`',f'Seed: {cfg.seed}; sample rate: {cfg.sample_rate}; crop length: {cfg.seconds}s',
        f'Raw canonical audio catalog: {len(df)} rows',
        f'Scenario profile: {cfg.scenario_profile}; scenarios/audio: {1+4*len(available_levels(cfg.scenario_profile))}',
        '## Methodology limitations',
        '- The uploaded audit ZIP contains reports and notebooks, not the audio payload. This experiment runs against the server-side AUA directory.',
        '- Source-specific recording artefacts may be confounded with real/fake label; source-stratified results mandatory.',
        '- Synthetic recordings have generator provenance; unknown/weak speaker groups must not be interpreted as leak-free.',
        '- Own dialectal script_profile must never substitute for true speaker_origin_group.',
        '- Synthetic reverb uses exponentially decaying simulated RIR, not physical room measurements.',
        '- Compression is real MP3 round-trip via ffmpeg.',
        '- ASRI is a *proposed experimental index*, not an established standard; always report EER and ROC-AUC.',
        '- The 5-level v2 protocol is a NEW experimental configuration. Its levels are ordered and are NOT equivalent to legacy v1 level IDs.',
        '- Accuracy threshold selected only from clean validation; hidden test locked throughout model selection.',
        '- Cluster bootstrap resamples source/speaker group, not audio variants as independent observations.',
        '- Metadata approval does not establish legal redistribution permission. No recordings are bundled in result ZIP.',
        '- Official AASIST/RawNet2 are optional distinct model architectures; lightweight CNN baselines are not mislabeled as them.',
        '', '## Output files',
        '01_catalog_all.csv, 01_catalog_issues.csv, 02_experimental_subset.csv, 03_locked_splits.csv,',
        '04_val_predictions_*.csv, 05_adaptive_weights.json, 07_hidden_predictions.csv,',
        '08_scenario_metrics.csv, 09_ASRI.csv, 10_source_strata.csv, 11_bootstrap.csv, charts and SAADE_RESULTS.zip.'
    ]
    (out/'README_RESULTS.md').write_text('\n'.join(t),encoding='utf-8')
    return out/'README_RESULTS.md'


def transcript_leakage_report(df,cfg):
    # Detector only. Cross-split repeated text can be legitimate, but needs expert decision.
    d=df[df.transcription.astype(str).str.len()>=25].copy()
    d['transcript_key']=d.transcription.fillna('').map(lambda t: hashlib.sha256(' '.join(str(t).lower().split()).encode()).hexdigest())
    leak_ids=d.groupby('transcript_key').split.nunique()
    repeated=d[d.transcript_key.isin(leak_ids[leak_ids>1].index)]
    repeated.to_csv(Path(cfg.out)/'03_transcription_cross_split_REVIEW.csv',index=False)
    return repeated


def generator_holdout_report(df,cfg):
    """Feasibility for an additional leave-one-generator-out study.
    Actual training must rebuild GROUPED train/validation without the held-out generator.
    """
    fake=df[(df.label==1)].copy()
    rows=[]
    for gen,g in fake.groupby('generator'):
        if not gen or len(g)<10:continue
        other=fake[fake.generator.ne(gen)]
        rows.append(dict(heldout_generator=gen,heldout_n=len(g),heldout_groups=g.group_id.nunique(),
             remaining_fake_n=len(other),remaining_fake_generators=other.generator.nunique(),
             feasibility='CHECK' if other.generator.nunique()<2 or g.group_id.nunique()<2 else 'POSSIBLE'))
    res=pd.DataFrame(rows);res.to_csv(Path(cfg.out)/'03_generator_holdout_feasibility.csv',index=False)
    return res


def mcdm_rank(summary_df,asri_df,cfg):
    """Exploratory MCDM robustness ranking. Explicit criteria, no hallucinated significance.
    Uses mean EER/AUC and ASRI plus worst-case BA; min/max polarity declared.
    """
    from scipy.stats import rankdata
    if summary_df.empty or asri_df.empty:return pd.DataFrame()
    rows=[]
    for (model,regime),g in summary_df.groupby(['model','regime']):
        z=g[g.category.isin(CATEGORIES)]
        a=asri_df[(asri_df.model==model)&(asri_df.regime==regime)]
        if z.empty or a.empty:continue
        rows.append(dict(model=model,regime=regime,ASRI=float(a.asri.iloc[0]),
            worst_BA=float(z.balanced_accuracy.min()),mean_AUC=float(z.roc_auc.mean()),
            mean_EER=float(z.eer.mean())))
    df=pd.DataFrame(rows).dropna()
    if len(df)<2:return df
    cols=['ASRI','worst_BA','mean_AUC','mean_EER']
    benefits=df[cols].copy();benefits['mean_EER']=1-benefits['mean_EER']
    lo,hi=benefits.min(),benefits.max()
    norm=(benefits-lo)/(hi-lo).replace(0,np.nan)
    norm=norm.fillna(.5).clip(0,1)
    W=np.full(len(cols),1/len(cols));A=norm.values
    df['WSM']=(A*W).sum(axis=1)
    dpos=np.linalg.norm((A-1)*W,axis=1);dneg=np.linalg.norm(A*W,axis=1)
    df['TOPSIS']=dneg/(dneg+dpos+1e-12)
    df['DnE']=1-np.linalg.norm((A-1)*W,axis=1)/np.linalg.norm(np.ones(len(cols))*W)
    df['DnM']=1-np.sum(abs(A-1)*W,axis=1)
    ranks=np.stack([rankdata(-A[:,i],method='average') for i in range(len(cols))],axis=1)
    df['SRD']=1-np.abs(ranks-1).sum(axis=1)/(len(cols)*max(1,len(df)-1))
    df['MCDM_consensus']=df[['WSM','TOPSIS','DnE','DnM','SRD']].mean(axis=1)
    df.sort_values('MCDM_consensus',ascending=False).to_csv(Path(cfg.out)/'12_MCDM_exploratory.csv',index=False)
    return df.sort_values('MCDM_consensus',ascending=False)


def gee_cluster_effects(preds,cfg):
    """Exploratory clustered GEE logistic model; interpretable main effects,
    not a replacement for paired bootstrap or holdout validity. Requires statsmodels.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print('statsmodels unavailable: skipping GEE');return None
    d=preds[preds.category.isin(CATEGORIES)&preds.level.isin(available_levels(cfg.scenario_profile))].copy()
    if d.empty or d.group_id.nunique()<10:return None
    d['correct']=(d.label==d.pred).astype(int)
    d['level']=d.level.astype(str)
    try:
        fit=smf.gee('correct ~ C(model) + C(regime) + C(category) + C(level)',
            groups='group_id',data=d, family=sm.families.Binomial()).fit()
        summary=fit.summary().as_text()
        (Path(cfg.out)/'13_GEE_clustered_main_effects.txt').write_text(summary,encoding='utf-8')
        return fit
    except Exception as exc:
        print('GEE did not converge / unavailable:',str(exc)[:180]);return None


def save_locked_protocol(cfg,df,models_info):
    """Freeze exactly which checkpoints and configurations will be evaluated on hidden test."""
    out=Path(cfg.out); p=out/'06_FINAL_PROTOCOL.json'
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        return h.hexdigest()
    models=[]
    for (name,regime),meta in models_info.items():
        cp=out/'models'/f'{safe_key(name)}_{safe_key(regime)}.pt'
        if not cp.exists():raise FileNotFoundError(cp)
        models.append(dict(model=name,regime=regime,threshold=meta['threshold'],
            checkpoint=str(cp),checkpoint_sha256=sha(cp)))
    obj=dict(dataset_hash=frozen_hash(df),config={k:v for k,v in dataclasses.asdict(cfg).items() if k!='test_finalization'},
        models=sorted(models,key=lambda m:(m['model'],m['regime'])))
    obj=json.loads(json.dumps(obj,default=str)) # canonicalize tuples/ndarrays before comparing to JSON
    if p.exists():
        old=json.loads(p.read_text())
        if old!=obj:raise RuntimeError('Frozen final protocol changed. Start new experiment directory, not re-use test.')
    else:save_json(p,obj)
    return obj


def verify_locked_protocol(cfg,df):
    p=Path(cfg.out)/'06_FINAL_PROTOCOL.json'
    if not p.exists():raise RuntimeError('Finalize training and lock final protocol FIRST.')
    obj=json.loads(p.read_text());
    if obj['dataset_hash']!=frozen_hash(df):raise RuntimeError('Dataset split changed since protocol lock')
    for rec in obj['models']:
        h=hashlib.sha256()
        with open(rec['checkpoint'],'rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        if h.hexdigest()!=rec['checkpoint_sha256']:raise RuntimeError('Checkpoint changed since protocol freeze: '+rec['checkpoint'])
    return obj
