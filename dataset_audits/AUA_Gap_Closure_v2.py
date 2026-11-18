# %% [markdown]
# AUA Gap Closure v2 — безпечна автоматизація виправлень за ТЗ
# Запускайте в JupyterLab. Вхідні файли AUA НЕ змінюються.
# Вихідні файли створюються тільки в папці results/AUA_gap_closure_v2.

# %%
from __future__ import annotations
import os, re, json, csv, hashlib, subprocess, wave, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# Змінити тільки якщо AUA лежить в іншому місці
DATA_ROOT = os.environ.get('AUA_ROOT', None)

if DATA_ROOT is None:
    bases = [Path.cwd(), *Path.cwd().parents, Path.home(), Path('/home/jovyan/work'), Path('/home/jovyan')]
    candidates = []
    for base in bases:
        for p in (base / 'AUA', base):
            if p.is_dir():
                score = sum((p / d).is_dir() for d in ('01_audio','02_video','03_text','05_splits'))
                if score >= 3:
                    candidates.append((score, p.resolve()))
    if not candidates:
        raise FileNotFoundError('Папку AUA не знайдено. Задайте: os.environ["AUA_ROOT"] = "/шлях/AUA"')
    DATA_ROOT = sorted(candidates, key=lambda a: (-a[0], len(str(a[1]))))[0][1]
else:
    DATA_ROOT = Path(DATA_ROOT).expanduser().resolve()

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(DATA_ROOT)

OUT = Path(os.environ.get('AUA_GAP_OUT', str(DATA_ROOT.parent / 'results' / 'AUA_gap_closure_v2'))).expanduser().resolve()
if OUT == DATA_ROOT or DATA_ROOT in OUT.parents:
    raise ValueError('OUT не може лежати всередині DATA_ROOT: це змішує дані і звіти')
OUT.mkdir(parents=True, exist_ok=True)
STAGE = OUT / 'staging'; STAGE.mkdir(exist_ok=True)
INCOMING = OUT / 'incoming'; (INCOMING / 'audio').mkdir(parents=True, exist_ok=True); (INCOMING / 'video').mkdir(exist_ok=True)
REPORT = OUT / 'reports'; REPORT.mkdir(exist_ok=True)
TARGET = {'own_audio_h':3.0,'real_audio_h':5.0,'synthetic_audio_h':5.0,'total_audio_h':10.0,'own_video_files':24,'unique_video_files':80,'text_cases':300}
AUDIO_EXT = {'.wav','.mp3','.m4a','.aac','.ogg','.flac','.opus'}
VIDEO_EXT = {'.mp4','.webm','.mov','.mkv','.avi','.m4v'}
NOW = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
print('AUA:', DATA_ROOT, '\nРезультати:', OUT)

# %% [markdown]
# 1. Джерела аудіо: скільки реальних годин є за кожним джерелом
# Не додаємо own_master повторно; не враховуємо копії у 05_splits.

# %%
def rel(p):
    return Path(p).resolve().relative_to(DATA_ROOT).as_posix()

def ffprobe_duration(p):
    p = Path(p)
    if p.suffix.lower()=='.wav':
        try:
            with wave.open(str(p),'rb') as w:
                return w.getnframes()/w.getframerate() if w.getframerate() else None
        except (wave.Error, OSError):
            pass
    try:
        r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)],capture_output=True,text=True,timeout=30)
        if r.returncode==0:
            v = float(r.stdout.strip()); return v if np.isfinite(v) and v>=0 else None
    except (ValueError,OSError,subprocess.TimeoutExpired):
        pass
    return None

def read_table(p):
    p = Path(p)
    try:
        if p.suffix.lower()=='.csv': return pd.read_csv(p,low_memory=False)
        if p.suffix.lower()=='.tsv': return pd.read_csv(p,sep='\t',low_memory=False)
        if p.suffix.lower()=='.parquet': return pd.read_parquet(p)
        if p.suffix.lower()=='.jsonl': return pd.read_json(p,lines=True)
        if p.suffix.lower()=='.json':
            obj=json.loads(p.read_text(encoding='utf-8'))
            if isinstance(obj,list):return pd.DataFrame(obj)
            if isinstance(obj,dict):
                for k in ['cases','records','data','items']:
                    if isinstance(obj.get(k),list):return pd.DataFrame(obj[k])
                return pd.DataFrame([obj])
    except Exception as e:
        print('ПРИМІТКА: не прочитано',p.name,str(e)[:100])
    return pd.DataFrame()

def choose_col(df, names):
    lookup={str(c).lower().strip():c for c in df.columns}
    return next((lookup[n] for n in names if n in lookup),None)

def safe_str(v):
    if v is None:return ''
    try:
        if pd.isna(v):return ''
    except (TypeError,ValueError):pass
    return str(v).strip()

def category_audio(p):
    q=p.relative_to(DATA_ROOT / '01_audio').parts
    if not q:return None
    if q[0]=='own_processed':return 'own_processed'
    if q[0]=='external_selected' and len(q)>=3:return 'external_'+q[1].lower()
    if q[0]=='synthetic' and len(q)>=3:return 'synthetic_'+q[1].lower()
    return None

cache_file=REPORT/'audio_probe_cache.csv'
cache={}
if cache_file.exists():
    try:
        for r in pd.read_csv(cache_file).to_dict('records'):
            cache[(str(r['path']),int(r['size']),int(r['mtime_ns']))]=float(r['duration_sec'])
    except Exception:pass
rows=[]; cache_rows=[]
audio_root=DATA_ROOT/'01_audio'
for p in sorted(audio_root.rglob('*')):
    if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:continue
    cat=category_audio(p)
    if not cat:continue
    st=p.stat(); k=(str(p),st.st_size,st.st_mtime_ns)
    duration=cache.get(k)
    if duration is None or not np.isfinite(duration):duration=ffprobe_duration(p)
    cache_rows.append({'path':str(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'duration_sec':duration})
    rows.append({'path':rel(p),'source':cat,'duration_sec':duration,'duration_h':(duration or 0)/3600,'size_bytes':st.st_size,'readable':duration is not None})

audio=pd.DataFrame(rows)
pd.DataFrame(cache_rows).to_csv(cache_file,index=False)
audio.to_csv(REPORT/'01_audio_by_file.csv',index=False)
audio_summary=audio.groupby('source',as_index=False).agg(files=('path','count'),hours=('duration_h','sum'),unreadable=('readable',lambda x:int((~x).sum())))
audio_summary.to_csv(REPORT/'02_audio_by_source.csv',index=False)
print(audio_summary.round(4).to_string(index=False))

def audio_h(source):
    return float(audio.loc[audio.source==source,'duration_h'].sum())

own_h=audio_h('own_processed')
cv_h=audio_h('external_common_voice')
fleurs_h=audio_h('external_fleurs')
dr_h=audio_h('synthetic_drssu')
tts_h=audio_h('synthetic_new-tts')
external_h=float(audio.loc[audio.source.str.startswith('external_'),'duration_h'].sum())
synthetic_h=float(audio.loc[audio.source.str.startswith('synthetic_'),'duration_h'].sum())
real_h=own_h+external_h
print(f'\nВласне реальне {own_h:.3f} год; реальне всього {real_h:.3f} год; синтетичне {synthetic_h:.3f} год')
print(f'Потрібно ще власного реального: {max(0,(TARGET["own_audio_h"]-own_h)*60):.1f} хв')

# %% [markdown]
# 2. Реєстр власних мовців і приймання НОВИХ записів
# Скопіюйте нові аудіофайли в папку incoming/audio; заповніть шаблон реєстру.
# Записи без підтвердженої згоди лише перевіряємо, але НЕ обробляємо в прийнятий набір.

# %%
def speaker_name(p):
    m=re.search(r'(?<![A-Za-z0-9])(?:AUA[_-])?(S\d{3,4})(?!\d)',Path(p).name,re.I)
    return m.group(1).upper() if m else ''

speaker_meta_path=DATA_ROOT/'01_audio/own_master/metadata_audio.csv'
speaker_meta=read_table(speaker_meta_path)
known_speakers=set()
if not speaker_meta.empty:
    id_col=choose_col(speaker_meta,['speaker_id'])
    if id_col:known_speakers={safe_str(x).upper() for x in speaker_meta[id_col] if safe_str(x)}
known_speakers |= {speaker_name(Path(p)) for p in audio.loc[audio.source=='own_processed','path']}
known_speakers.discard('')

speaker_template=INCOMING/'recording_register.csv'
if not speaker_template.exists():
    missing_n=max(0,36-len(known_speakers))
    available=[f'S{i:03d}' for i in range(1,500) if f'S{i:03d}' not in known_speakers]
    data=[]
    for i,s in enumerate(available[:missing_n]):
        data.append({'speaker_id':s,'speaker_origin_group':'','birth_region':'','childhood_region':'',
                     'script_profile':'neutral','script_id':'','speech_mode':'read_spontaneous',
                     'dialect_authenticity':'not_verified','consent_id':'','consent_status':'PENDING',
                     'voice_cloning_consent':'NO','notes':''})
    pd.DataFrame(data,columns=['speaker_id','speaker_origin_group','birth_region','childhood_region','script_profile','script_id','speech_mode','dialect_authenticity','consent_id','consent_status','voice_cloning_consent','notes']).to_csv(speaker_template,index=False)

reg=pd.read_csv(speaker_template).fillna('')
reg['speaker_id']=reg.speaker_id.astype(str).str.upper().str.strip()
accepted_register={}
for _,r in reg.iterrows():
    val=str(r.get('consent_status','')).strip().upper()
    if val in {'YES','SIGNED','APPROVED','TAK','ТАК','ПІДПИСАНО'} and safe_str(r.get('consent_id')):
        accepted_register[r.speaker_id]=r.to_dict()

incoming_audio=[p for p in (INCOMING/'audio').rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXT]
stage_audio_dir=STAGE/'processed_new_audio'; stage_audio_dir.mkdir(exist_ok=True)
new_rows=[]
for p in sorted(incoming_audio):
    speaker=speaker_name(p); duration=ffprobe_duration(p)
    allowed=speaker in accepted_register
    record={'incoming_file':str(p),'speaker_id':speaker,'duration_min':(duration/60 if duration else None),
            'consent_verified_in_register':allowed,'status':'CHECK','processed_path':''}
    if not speaker:record['status']='NEED_SPEAKER_ID_IN_FILENAME'
    elif not allowed:record['status']='NEED_SIGNED_CONSENT_AND_REGISTER'
    elif not duration:record['status']='BAD_MEDIA'
    else:
        stem=re.sub(r'[^A-Za-z0-9_-]+','_',p.stem)
        dest=stage_audio_dir/(stem+'_16k.wav')
        if dest.exists():
            record['status']='ALREADY_STAGED'
        elif shutil.which('ffmpeg'):
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-i',str(p),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(dest)]
            proc=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            record['status']='STAGED_FOR_QA' if proc.returncode==0 and dest.exists() else 'FFMPEG_ERROR: '+proc.stderr[-120:]
        else:record['status']='FFMPEG_NOT_FOUND'
        if dest.exists():record['processed_path']=str(dest)
    new_rows.append(record)
new_audio_df=pd.DataFrame(new_rows,columns=['incoming_file','speaker_id','duration_min','consent_verified_in_register','status','processed_path'])
new_audio_df.to_csv(REPORT/'03_new_audio_intake.csv',index=False)
print(f'Відомих власних мовців: {len(known_speakers)}. Шаблон для нових: {speaker_template}')
print('Нових аудіофайлів у incoming:',len(incoming_audio),'; прийняті до staging:',int(new_audio_df.status.isin(['STAGED_FOR_QA','ALREADY_STAGED']).sum()))
new_stage_h=0.0
if len(new_audio_df):
    for path in new_audio_df.loc[new_audio_df.status.isin(['STAGED_FOR_QA','ALREADY_STAGED']),'processed_path']:
        if safe_str(path) and Path(path).is_file():new_stage_h+=(ffprobe_duration(path) or 0)/3600
print(f'Додаткове аудіо у staging: {new_stage_h:.3f} год (попередньо, потребує QA; НЕ додаємо до офіційного KPI).')

# %% [markdown]
# 3. Відео: реальні master, похідні маніпуляції, підготовка реєстру учасників
# Кодування не створює нових незалежних учасників. 6 додаткових фрагментів ≠ 6 нових людей.

# %%
video_root=DATA_ROOT/'02_video'
video_rows=[]
for folder in ['own_master','external_master','cutted','own_manipulated','manipulated']:
    base=video_root/folder
    if not base.exists():continue
    for p in sorted(base.rglob('*')):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            m=re.search(r'AVUA_(P\d+|S\d+|EXT\d+)',p.name,re.I)
            inferred=m.group(1).upper() if m else ''
            video_rows.append({'path':rel(p),'subset':folder,'filename':p.name,'person_or_family_hint':inferred})
video_df=pd.DataFrame(video_rows)
video_df.to_csv(REPORT/'04_video_inventory.csv',index=False)
print(video_df.groupby('subset').size().to_string())
own_v=video_df.loc[video_df.subset=='own_master'].copy()
unique_people=sorted(set(own_v.person_or_family_hint)-{''})
print('Власних master:',len(own_v),' ID людей, розпізнаних із назв:',len(unique_people),unique_people)

video_template=INCOMING/'new_video_register.csv'
if not video_template.exists():
    todo=max(0,24-len(unique_people))
    rows_video=[]
    for i in range(todo):
        rows_video.append({'person_id':'','video_filename':'','speaker_origin_group':'','script_profile':'neutral',
                           'consent_id':'','consent_status':'PENDING','can_create_manipulations':'NO','license':'','review_status':'PENDING'})
    pd.DataFrame(rows_video,columns=['person_id','video_filename','speaker_origin_group','script_profile','consent_id','consent_status','can_create_manipulations','license','review_status']).to_csv(video_template,index=False)
print('Шаблон реєстру відео:',video_template,'; додаткові унікальні особи орієнтовно:',max(0,24-len(unique_people)))

# Нові відео: лише валідація без фізичного дублювання великих master-файлів.
new_vreg=pd.read_csv(video_template).fillna('')
incoming_videos=[p for p in (INCOMING/'video').rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXT]
video_intake=[]
seen_people=set(unique_people)
for p in sorted(incoming_videos):
    match=new_vreg.loc[new_vreg.video_filename.astype(str).str.strip()==p.name]
    row=match.iloc[0].to_dict() if len(match) else {}
    sid=safe_str(row.get('person_id')).upper()
    consent=safe_str(row.get('consent_status')).upper() in {'YES','SIGNED','APPROVED','ТАК','ПІДПИСАНО'}
    permitted=safe_str(row.get('can_create_manipulations')).upper() in {'YES','TRUE','ТАК'}
    probe=ffprobe_duration(p)
    status=('NEED_REGISTER_ROW' if not len(match) else
            'NEED_UNIQUE_PERSON_ID' if not sid or sid in seen_people else
            'NEED_CONSENT_ID_AND_SIGNED_CONSENT' if not (consent and safe_str(row.get('consent_id'))) else
            'NEED_MANIPULATION_PERMISSION' if not permitted else
            'INVALID_MEDIA' if not probe else 'STAGED_FOR_VIDEO_QA')
    if status=='STAGED_FOR_VIDEO_QA':seen_people.add(sid)
    video_intake.append({'file_path':str(p),'person_id':sid,'duration_sec':probe,'status':status})
video_intake_df=pd.DataFrame(video_intake,columns=['file_path','person_id','duration_sec','status'])
video_intake_df.to_csv(REPORT/'04b_new_video_intake.csv',index=False)
print('Нових відео у incoming:',len(incoming_videos),'; попередньо готово до QA:',int((video_intake_df.status=='STAGED_FOR_VIDEO_QA').sum()))

# %% [markdown]
# 4. Перевірка зв'язків master → variant; перевірка, що маніпуляції справді мають вихідний файл.

# %%
maprows=[]
for folder in ['own_manipulated','manipulated']:
    f=video_root/folder/'original_to_variant.csv'
    if not f.exists():continue
    df=read_table(f)
    if df.empty:continue
    oc=choose_col(df,['original_path','master_path','source_path','original','parent_path'])
    vc=choose_col(df,['variant_path','output_path','file_path','variant','file_name'])
    cc=choose_col(df,['category','manipulation_type','type'])
    for _,r in df.iterrows():
        original=safe_str(r[oc]) if oc else ''
        variant=safe_str(r[vc]) if vc else ''
        def resolve_path(s):
            q=Path(s)
            return q if q.is_absolute() else DATA_ROOT/q
        good_original=resolve_path(original).is_file() if original else False
        good_variant=resolve_path(variant).is_file() if variant else False
        maprows.append({'map_file':rel(f),'original_path':original,'variant_path':variant,
                        'category':safe_str(r[cc]) if cc else folder,
                        'master_exists':good_original,'variant_exists':good_variant,
                        'status':'PASS' if good_original and good_variant else 'MISSING_PATH'})
map_df=pd.DataFrame(maprows,columns=['map_file','original_path','variant_path','category','master_exists','variant_exists','status'])
map_df.to_csv(REPORT/'05_video_master_variant_links.csv',index=False)
print('Пар master→variant:',len(map_df),'із помилками:',int((map_df.status!='PASS').sum()))

# Перевірка типів маніпуляцій за категоріями з вихідних маніфестів.
category_counts=map_df.groupby('category').size().reset_index(name='files') if len(map_df) else pd.DataFrame(columns=['category','files'])
category_counts.to_csv(REPORT/'05b_video_manipulations_from_mapping.csv',index=False)
print('Типи маніпуляцій з mapping:')
print(category_counts.to_string(index=False))

# %% [markdown]
# 5. Тексти: 60 UNLP + 30 WUNU як КАНДИДАТИ на рецензування
# Автоматично створюються 90 ID TXTUA_0121..0210 без дублювання існуючих ID.
# Це ще НЕ готові фактологічні RAG-бенчмарки: потрібна експертна перевірка джерел і перекладу.

# %%
text_root=DATA_ROOT/'03_text'
selected_metadata=read_table(text_root/'metadata_text.csv')
own_metadata=read_table(text_root/'own_cases/metadata_own.jsonl')
existing_ids=set()
for df in [selected_metadata,own_metadata]:
    if not df.empty:
        col=choose_col(df,['case_id','question_id','id'])
        if col:existing_ids.update(safe_str(x) for x in df[col] if safe_str(x))
existing_ids.update(p.stem for p in (text_root/'selected_cases').glob('TXTUA_*.json'))
existing_ids.discard('')
print('Ідентифікаторів існуючих кейсів:',len(existing_ids))

src_unlp=DATA_ROOT/'00_sources_and_licenses/text/unlp-2025-shared-task/data/techniques_classification/train.parquet'
src_wunu=DATA_ROOT/'00_sources_and_licenses/text/wunu-fake_trusted/dataset.csv'

# Якщо є підготовлений вхідний CSV для рецензування, НЕ перезаписуємо його.
staged_text_dir=STAGE/'text_review_candidates'; staged_text_dir.mkdir(exist_ok=True)

def find_text_col(df):
    exact=choose_col(df,['text','message','content','post','body','text_uk','news_text','article','title','claim'])
    if exact:return exact
    return next((c for c in df.columns if any(k in str(c).lower() for k in ['text','message','content','post'])),None)

def find_label_col(df):
    return choose_col(df,['label','labels','class','category','techniques','technique','target','fake','is_fake','source_label'])

def canonical_text(s):
    return re.sub(r'\s+',' ',str(s).casefold()).strip()

already_text=set()
for df in [selected_metadata, own_metadata]:
    if df.empty:continue
    col=find_text_col(df)
    if col:
        already_text.update(canonical_text(v) for v in df[col].dropna() if len(canonical_text(v))>=40)

text_staging=[]; used_text=set(already_text)
source_specs=[('UNLP_2025',src_unlp,60,'manipulation_detection'),('WUNU_FAKE_TRUSTED',src_wunu,30,'credibility_classification')]
source_status=[]
for source,p,quota,task_type in source_specs:
    if not p.is_file():
        source_status.append({'source':source,'available':False,'rows':0,'selected':0,'reason':'FILE_NOT_FOUND'});continue
    df=read_table(p)
    text_col=find_text_col(df); label_col=find_label_col(df)
    if df.empty or text_col is None:
        source_status.append({'source':source,'available':True,'rows':len(df),'selected':0,'reason':'TEXT_COLUMN_NOT_FOUND'});continue
    chosen=0
    # Детермінований вибір із різних частин набору; під час рецензування слід перевірити різноманітність.
    order=np.random.default_rng(2026).permutation(len(df))
    for idx in order:
        r=df.iloc[int(idx)]
        text=safe_str(r[text_col]); norm=canonical_text(text)
        if len(norm)<70 or norm in used_text:continue
        used_text.add(norm)
        label=safe_str(r[label_col]) if label_col else ''
        if source=='UNLP_2025':
            q_uk='Які маніпулятивні техніки присутні в наведеному повідомленні? Обґрунтуйте відповідь конкретними фрагментами тексту.'
            q_en='Which manipulative techniques appear in the Ukrainian-language message below? Support your answer with specific text spans.'
        else:
            q_uk='Оцініть фактологічну обґрунтованість наведеного повідомлення. Які твердження потребують незалежної перевірки?'
            q_en='Assess the factual grounding of the Ukrainian-language message below. Which claims require independent verification?'
        text_staging.append({'source_dataset':source,'original_file':rel(p),'source_row_number':int(idx)+2,
               'task_type':task_type,'text_uk':text,'question_uk':q_uk,'question_en':q_en,
               'source_label_raw':label,'reference_answer_uk':'','reference_answer_en':'',
               'evidence_urls':'','evidence_snapshot_ids':'','verdict_reviewed':'',
               'translator_verified':'NO','license_reviewed':'NO','review_status':'PENDING_EXPERT_REVIEW',
               'release_ready':False})
        chosen+=1
        if chosen>=quota:break
    source_status.append({'source':source,'available':True,'rows':len(df),'selected':chosen,'reason':'CANDIDATES_NOT_APPROVED'})

# ID 0121-0210 перед власними 0211-0300, резервуємо пропущені.
preferred=[f'TXTUA_{i:04d}' for i in range(121,211) if f'TXTUA_{i:04d}' not in existing_ids]
extra=[f'TXTUA_{i:04d}' for i in range(301,1000) if f'TXTUA_{i:04d}' not in existing_ids]
for r,cid in zip(text_staging,preferred+extra):r['case_id']=cid
stage_csv=STAGE/'TEXT_90_CANDIDATES_FOR_REVIEW.csv'
stage_jsonl=STAGE/'TEXT_90_CANDIDATES_FOR_REVIEW.jsonl'
should_write=bool(text_staging) and not stage_csv.exists()
if stage_csv.exists() and text_staging:
    try:
        previous=pd.read_csv(stage_csv).fillna('')
        pending_all=all(str(x).startswith('PENDING') for x in previous.get('review_status',pd.Series(dtype=str)).astype(str))
        if len(previous)<len(text_staging) and pending_all:
            backup=stage_csv.with_name('TEXT_CANDIDATES_BACKUP_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
            shutil.copy2(stage_csv,backup)
            print('Попередню неповну чернетку збережено:',backup)
            should_write=True
        elif len(previous)<len(text_staging):
            print('УВАГА: існує частково рецензована таблиця; не перезаписуємо її. Створено нову автоматичну чернетку.')
            fresh=stage_csv.with_name('TEXT_NEW_AUTO_DRAFT_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
            pd.DataFrame(text_staging).to_csv(fresh,index=False)
    except Exception as e:
        print('Не змінюємо існуючий файл рецензування:',str(e)[:100])
if should_write:
    pd.DataFrame(text_staging).to_csv(stage_csv,index=False)
    with stage_jsonl.open('w',encoding='utf-8') as f:
        for r in text_staging:f.write(json.dumps(r,ensure_ascii=False)+'\n')
else:
    print('Існуючий файл рецензування не перезаписано:',stage_csv)
source_status_df=pd.DataFrame(source_status)
source_status_df.to_csv(REPORT/'06_text_source_selection.csv',index=False)
print(source_status_df.to_string(index=False))
print('Кандидатів на експертну перевірку:',len(text_staging),'із 90; папка:',stage_csv)

# Автоматична перевірка заповнення обов'язкових полів у вже існуючих двомовних кейсах.
required=['question_uk','question_en','reference_answer_uk','reference_answer_en']
completeness=[]
for source,df in [('selected',selected_metadata),('own',own_metadata)]:
    if df.empty:continue
    for field in required:
        col=choose_col(df,[field])
        filled=sum(bool(safe_str(v)) for v in df[col]) if col else 0
        completeness.append({'source':source,'field':field,'rows':len(df),'filled':filled,'missing':len(df)-filled,'status':'CHECK' if col is None else ('PASS' if filled==len(df) else 'FAIL')})
completeness_df=pd.DataFrame(completeness)
completeness_df.to_csv(REPORT/'06b_text_required_field_completeness.csv',index=False)
print('Повнота двомовних полів:')
print(completeness_df.to_string(index=False) if len(completeness_df) else 'Метадані не знайдені')

# %% [markdown]
# 6. Побудова ЧИСТИХ маніфестів development / validation / hidden_test
# ЖОДНИХ копій media: тільки CSV зі шляхами. Усі похідні одного master мають ту саму групу.
# Якщо зв'язок походження не відомий — CHECK/UNASSIGNED, а не прихований PASS.

# %%
def make_splits(groups, seed=2026):
    groups=sorted(set(g for g in groups if g))
    rng=np.random.default_rng(seed)
    if not groups:return {}
    order=list(rng.permutation(len(groups)))
    n=len(order); n_dev=int(round(.60*n));n_val=int(round(.20*n))
    out={}
    for j,i in enumerate(order):
        s='development' if j<n_dev else ('validation' if j<n_dev+n_val else 'hidden_test')
        out[groups[int(i)]]=s
    return out

def speaker_from_audio(row):
    x=speaker_name(Path(row['path']).name)
    return x if x else ''

# Аудіо: спочатку за metadata_audio.csv; fallback — тільки якщо speaker ID є у назві.
audio_md=read_table(audio_root/'metadata_audio.csv')
audio_by_name={}
if not audio_md.empty:
    name_col=choose_col(audio_md,['working_path','file_name','filename'])
    sp_col=choose_col(audio_md,['speaker_id'])
    if name_col and sp_col:
        for _,r in audio_md.iterrows():
            name=Path(safe_str(r[name_col])).name
            sid=safe_str(r[sp_col])
            if name and sid and sid.lower() not in {'nan','none','unknown'}:
                audio_by_name[name]=sid

audio_manifest=[]
for _,r in audio.iterrows():
    sid=audio_by_name.get(Path(r.path).name) or speaker_from_audio(r)
    # У DRSSU ID у назві файлу НЕ гарантує реального мовця; групуємо за джерелом запису, якщо доступно.
    state='CHECK' if not sid else 'READY_TO_ASSIGN'
    audio_manifest.append({'file_path':r.path,'modality':'audio','source':r.source,'speaker_id':sid,
                           'group_id':('PERSON_'+str(sid).upper() if re.fullmatch(r'[SP]\d{3,4}',str(sid).upper()) else 'AUDIO_SPEAKER_'+str(sid) if sid else ''),'duration_sec':r.duration_sec,
                           'group_status':state})
am=pd.DataFrame(audio_manifest)
am['split']=am.group_id.map(make_splits(am.loc[am.group_status=='READY_TO_ASSIGN','group_id']))
am['split']=am['split'].fillna('UNASSIGNED')
am.to_csv(REPORT/'07_audio_new_split_manifest.csv',index=False)

# Відео: беремо підтверджене mapping, потім корінь EXTxxx/Pxxx/Sxxx із назви.
parent_of={}
for _,r in map_df.iterrows():
    if r.status=='PASS':parent_of[r.variant_path]=r.original_path

def video_group(path):
    parent=parent_of.get(path,path)
    m=re.search(r'AVUA_(P\d+|S\d+|EXT\d+)',Path(parent).name,re.I)
    if m:
        hint=m.group(1).upper()
        return ('PERSON_'+hint if re.fullmatch(r'[SP]\d{3,4}',hint) else 'VIDEO_'+hint), 'NAME_OR_MAPPING_CHECK'
    # Для невідомих master залишити без split до уточнення
    return '', 'UNRESOLVED_SOURCE'

vm=[]
for _,r in video_df.iterrows():
    group,status=video_group(r.path)
    vm.append({'file_path':r.path,'subset':r.subset,'group_id':group,
               'master_parent':parent_of.get(r.path,''),'group_status':status})
vm=pd.DataFrame(vm)
# Спільне присвоєння split між аудіо та відео: однаковий PERSON_... не може опинитися у різних частинах.
common_groups=make_splits(list(am.group_id)+list(vm.group_id))
am['split']=am.group_id.map(common_groups).fillna('UNASSIGNED')
am.to_csv(REPORT/'07_audio_new_split_manifest.csv',index=False)
vm['split']=vm.group_id.map(common_groups).fillna('UNASSIGNED')
vm.to_csv(REPORT/'08_video_new_split_manifest.csv',index=False)

# Текст: тільки вже наявні 210 явно ідентифікованих кейсів; нові 90 поки pending.
tm=[]
for src,df in [('selected',selected_metadata),('own',own_metadata)]:
    if df.empty:continue
    idc=choose_col(df,['case_id','question_id','id'])
    if not idc:continue
    for _,r in df.iterrows():
        case_id=safe_str(r[idc])
        if case_id:
            event=safe_str(r.get('event_id',''))
            tm.append({'case_id':case_id,'source':src,'group_id':'EVENT_'+event if event else 'CASE_'+case_id,
                       'group_confidence':'EVENT' if event else 'CASE_ONLY_NEED_EVENT_REVIEW'})
tm=pd.DataFrame(tm).drop_duplicates(subset=['case_id']) if tm else pd.DataFrame(columns=['case_id','source','group_id','group_confidence'])
tm['split']=tm.group_id.map(make_splits(tm.group_id)) if len(tm) else pd.Series(dtype=str)
tm.to_csv(REPORT/'09_text_new_split_manifest.csv',index=False)

# Перевірка групових конфліктів у нових маніфестах
for name,df in [('audio',am),('video',vm),('text',tm)]:
    grp=df[df['split']!='UNASSIGNED'].groupby('group_id').split.nunique() if len(df) else pd.Series(dtype=int)
    print(name,':',len(df),'рядків; UNASSIGNED=',int((df.split=='UNASSIGNED').sum()),'; group cross-split=',int((grp>1).sum()))

# %% [markdown]
# 7. Звіт точних розривів за ТЗ. CHECK означає, що кількість файлів не доводить якості.

# %%
def kpi(name,target,actual,category,auto=True,explanation=''):
    status='PASS' if auto and actual>=target else ('FAIL' if auto else 'CHECK')
    return {'category':category,'indicator':name,'target':target,'actual':round(float(actual),4),
            'gap':round(max(0,float(target)-float(actual)),4),'status':status,'comment':explanation}

kpis=[
 kpi('Common Voice, год',1.5,cv_h,'AUDIO'),
 kpi('FLEURS, год',0.5,fleurs_h,'AUDIO'),
 kpi('DRSSU synthetic, год',4.0,dr_h,'AUDIO'),
 kpi('New TTS synthetic, год',1.0,tts_h,'AUDIO'),
 kpi('Власне очищене аудіо, год',3.0,own_h,'AUDIO'),
 kpi('Власне аудіо, включно з новими у staging',3.0,own_h+new_stage_h,'AUDIO',auto=False,explanation='Не приймати як PASS до повної перевірки нових записів та дозволів'),
 kpi('Усього реального аудіо, год',5.0,real_h,'AUDIO'),
 kpi('Усього синтетичного аудіо, год',5.0,synthetic_h,'AUDIO'),
 kpi('Усього канонічного аудіо, год',10.0,real_h+synthetic_h,'AUDIO'),
 kpi('Власні мовці, унікальних ID',36,len(known_speakers),'AUDIO',auto=False,explanation='ID не дорівнює підтвердженій особі; регіони та згоди перевірити вручну'),
 kpi('Власні master-відео, файлів',24,len(own_v),'VIDEO'),
 kpi('Власні особи у відео, ID',24,len(unique_people),'VIDEO',auto=False,explanation='Необхідна звірка реальних осіб і дозволів'),
 kpi('Українські незалежні відеофрагменти',80,len(video_df.loc[video_df.subset.isin(['own_master','external_master','cutted'])]),'VIDEO',auto=False,explanation='cutted можуть походити з одних master і не обов’язково містити мову'),
 kpi('Маніпульовані відеофайли',80,len(video_df.loc[video_df.subset.isin(['own_manipulated','manipulated'])]),'VIDEO'),
 kpi('Підтверджені текстові кейси',300,len(tm),'TEXT',auto=False,explanation='210 існуючих кейсів + кандидати мають пройти експертне приймання'),
 kpi('Нові кандидати (UNLP+WUNU)',90,len(text_staging),'TEXT',auto=False,explanation='Чернетки не рахуються як verified RAG test cases'),
 kpi('Відео provenance mapping, справні зв’язки',len(map_df),int((map_df.status=='PASS').sum()),'QUALITY'),
]
summary=pd.DataFrame(kpis)
summary.to_csv(REPORT/'10_TZ_COMPLIANCE_AND_GAPS.csv',index=False)
print('\nРЕЗУЛЬТАТИ ПЕРЕВІРКИ:')
print(summary[['category','indicator','target','actual','gap','status']].to_string(index=False))

# %% [markdown]
# 8. Резюме, список ручних доробок, ZIP без вихідних медіафайлів
# Архів можна безпечно надіслати для аналізу (але перевірте текстові приклади перед поширенням).

# %%
readme=[
 '# AUA — виконання прогалин ТЗ (v2)', '',
 f'Час перевірки (UTC): {NOW}',f'Джерело: `{DATA_ROOT}`','',
 '## Підтверджені автоматикою обсяги','',
 f'- Common Voice: **{cv_h:.3f} год** / 1.5',
 f'- FLEURS: **{fleurs_h:.3f} год** / 0.5',
 f'- DRSSU: **{dr_h:.3f} год** / 4',
 f'- New TTS: **{tts_h:.3f} год** / 1',
 f'- Власне аудіо: **{own_h:.3f} год** / 3; у staging додатково **{new_stage_h:.3f} год**; нестача **{max(0,(3-own_h)*60):.1f} хв** без staging',
 f'- Власні video master: **{len(own_v)}** / 24; нестача **{max(0,24-len(own_v))}**',
 f'- Існуючі структуровані кейси: **{len(tm)}**; згенеровані кандидатури: **{len(text_staging)}**, *ще не прийняті*','',
 '## Дії, які не можна автоматично назвати виконаними','',
 '1. Залучити додаткових реальних мовців і завантажити записи з підтвердженою згодою в incoming/audio.',
 '2. Перевірити реальне походження та три регіональні групи мовців; не ототожнювати регіональність тексту з акцентом людини.',
 '3. Додати потрібні video master із дозволами; фрагментація наявних відео не збільшує кількість людей.',
 '4. Перевірити всі 90 запропонованих текстових кейсів, заповнити еталонні відповіді, докази, англомовну еквівалентність, ліцензії.',
 '5. За потреби перевірити непрямі дублікати, джерела/події й особистості в нових split. Самі CSV не доводять відсутності всіх витоків.',
 '6. НЕ переносити чужі дані в відкритий реліз без перевіреної ліцензії та згод.', '',
 '## Де файли', '',
 '- `reports/01_audio_by_file.csv`, `reports/02_audio_by_source.csv` — години по джерелах',
 '- `incoming/recording_register.csv` — заповнити дані й дозвіл на власні аудіозаписи',
 '- `incoming/audio/`, `incoming/video/` — сюди покласти нові файли',
 '- `incoming/new_video_register.csv` — окремо ID людей та дозволи',
 '- `staging/TEXT_90_CANDIDATES_FOR_REVIEW.csv` — додаткові 90 кандидатів',
 '- `reports/07_audio_new_split_manifest.csv` — аудіо split',
 '- `reports/08_video_new_split_manifest.csv` — video split',
 '- `reports/09_text_new_split_manifest.csv` — текстові split',
 '- `reports/10_TZ_COMPLIANCE_AND_GAPS.csv` — виконання ТЗ',
]
(OUT/'README_RESULTS.md').write_text('\n'.join(readme),encoding='utf-8')

zip_path=OUT.parent / f'AUA_GAP_CLOSURE_REPORTS_{datetime.now():%Y%m%d_%H%M%S}.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for folder in [REPORT,STAGE]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl','.md','.txt'}:
                # Обмежити архів звітами і кандидатами; без audio/video і секретних файлів.
                z.write(p,p.relative_to(OUT).as_posix())
    z.write(OUT/'README_RESULTS.md','README_RESULTS.md')
print('\nГОТОВО. Архів результатів:',zip_path,'; розмір MiB:',round(zip_path.stat().st_size/1024**2,2))
print('Оригінальні файли набору НЕ змінено.')
