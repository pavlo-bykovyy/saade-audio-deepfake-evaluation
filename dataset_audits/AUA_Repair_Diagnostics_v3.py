# %% [markdown]
# # AUA v3 — виправлення діагностики після Gap Closure v2
#
# **Мета:** з'ясувати, що можна виправити автоматично, і підготувати точні реєстри для ручного приймання.
# **Безпека:** початкові медіафайли та CSV у `AUA` НЕ змінюються. Усе записується в окрему теку `results/AUA_repair_v3`.
#
# Що робить ноутбук:
# 1. Виправляє *посилання* на зовнішні відео (originals → cutted) **лише коли файл з таким ім'ям дійсно існує**.
# 2. Розбирає структуру корпусу ЗУНУ, включно із вкладеними даними, і формує до 30 **кандидатів**, а не вигаданих еталонних відповідей.
# 3. Перевіряє 60 UNLP-кандидатів та формує пакет для ручної перевірки 90 англомовних полів власних кейсів.
# 4. Діагностує помилки групування у split (зокрема 1760 DRSSU в одній групі).
# 5. Формує зведений ZIP із CSV/JSON/TXT, без медіафайлів.
#
# Увага: згенеровані чернетки не вважаються готовими фактологічними RAG-кейсами. 
# Встановлені за назвами файлів групи не гарантують унікальних фізичних осіб.

# %% [markdown]
# ## 1. Параметри
# У типовій конфігурації достатньо натиснути **Run All Cells**.

# %%
from pathlib import Path
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime, timezone
import os, re, csv, io, json, ast, hashlib
import pandas as pd
import numpy as np

AUA_ROOT = os.environ.get('AUA_ROOT', '').strip()
if AUA_ROOT:
    ROOT = Path(AUA_ROOT).expanduser().resolve()
else:
    starts = [Path.cwd(), *Path.cwd().parents, Path.home(), Path('/home/jovyan/work')]
    candidates=[]
    for b in starts:
        for p in [b, b/'AUA']:
            if (p/'01_audio').is_dir() and (p/'02_video').is_dir() and (p/'03_text').is_dir():
                candidates.append(p.resolve())
    if not candidates:
        raise FileNotFoundError('AUA не знайдено. Вкажіть os.environ["AUA_ROOT"] = "/home/jovyan/work/AUA"')
    ROOT = sorted(set(candidates),key=lambda x:len(str(x)))[0]
assert (ROOT/'02_video').is_dir(), ROOT
OUT = Path(os.environ.get('AUA_REPAIR_OUT',str(ROOT.parent/'results'/'AUA_repair_v3'))).expanduser().resolve()
if OUT == ROOT or ROOT in OUT.parents:
    raise ValueError('Папка результатів повинна бути поза AUA')
OUT.mkdir(parents=True, exist_ok=True)
V2 = ROOT.parent/'results'/'AUA_gap_closure_v2'
print('Набір:',ROOT,'\nДіагностика:',OUT,'\nПопередній аудит:',V2)

# %% [markdown]
# ## 2. Відео: відновлення зв’язків між оригіналом і маніпуляцією
# В аудиторському архіві 85 посилань вказували на `02_video/originals`. Перевіримо, чи існують їхні оригінали у `02_video/cutted`.

# %%
def read_table(path):
    path=Path(path)
    try:
        if path.suffix.lower()=='.csv':return pd.read_csv(path,low_memory=False)
        if path.suffix.lower()=='.tsv':return pd.read_csv(path,sep='\t',low_memory=False)
        if path.suffix.lower()=='.jsonl':return pd.read_json(path,lines=True)
        if path.suffix.lower()=='.parquet':return pd.read_parquet(path)
        if path.suffix.lower()=='.json':
            obj=json.loads(path.read_text(encoding='utf-8'))
            if isinstance(obj,list):return pd.DataFrame(obj)
            if isinstance(obj,dict):
                for key in ('records','data','items','rows','cases'):
                    if isinstance(obj.get(key),list):return pd.DataFrame(obj[key])
                return pd.DataFrame([obj])
    except Exception as exc:
        print('Не прочитано',path.name,':',str(exc)[:150])
    return pd.DataFrame()

def pick(df, alternatives):
    lookup={str(c).strip().lower():c for c in df.columns}
    return next((lookup[x] for x in alternatives if x in lookup),None)

def as_path(value):
    p=Path(str(value))
    return p if p.is_absolute() else ROOT/p

def rel(p):
    try:return Path(p).resolve().relative_to(ROOT).as_posix()
    except Exception:return str(p)

mappings=[]
for dirname in ('own_manipulated','manipulated'):
    mpath=ROOT/'02_video'/dirname/'original_to_variant.csv'
    if not mpath.is_file(): continue
    df=read_table(mpath)
    oc=pick(df,['original_path','master_path','source_path','original','parent_path'])
    vc=pick(df,['variant_path','output_path','file_path','variant','file_name'])
    tc=pick(df,['category','manipulation_type','type'])
    if oc is None or vc is None:
        print('Невідомі назви колонок mapping:',mpath,list(df.columns));continue
    for row in df.to_dict('records'):
        orig=str(row.get(oc,'')).strip();var=str(row.get(vc,'')).strip()
        original=as_path(orig);variant=as_path(var)
        found=original.is_file()
        resolved=original
        method='ORIGINAL_EXISTS' if found else 'MISSING'
        if not found and orig:
            # дозволяємо тільки конкретну перевірювану підстановку шляху
            candidate=ROOT/'02_video'/'cutted'/Path(orig).name
            if candidate.is_file():
                resolved=candidate; found=True;method='MATCHED_CUTTED_BASENAME'
        mappings.append(dict(mapping_file=rel(mpath),original_path=orig,variant_path=var,
             resolved_original=rel(resolved) if found else '',
             manipulation_type=str(row.get(tc,'')) if tc else dirname,
             original_exists=found,variant_exists=variant.is_file(),resolution=method,
             status='PASS' if found and variant.is_file() else 'CHECK'))
links=pd.DataFrame(mappings)
links.to_csv(OUT/'VIDEO_LINKS_RESOLVED.csv',index=False)
if len(links):
    print('Пари:',len(links),'дійсні:',int((links.status=='PASS').sum()))
    print(links.groupby(['resolution','status']).size().to_string())
    if (links.status!='PASS').any():display(links[links.status!='PASS'].head(12))
else:print('Таблиці mapping не знайдені')

# %% [markdown]
# ## 3. Попередні split — контроль не лише конфліктів, але й перекосів
# Якщо весь DRSSU має один фіктивний speaker ID, відсутність cross-split *не означає*, що split правильний.

# %%
audio_path=V2/'reports'/'07_audio_new_split_manifest.csv'
video_path=V2/'reports'/'08_video_new_split_manifest.csv'
text_path=V2/'reports'/'09_text_new_split_manifest.csv'
source_reports={}
for kind,p in [('audio',audio_path),('video',video_path),('text',text_path)]:
    df=read_table(p) if p.exists() else pd.DataFrame()
    if df.empty:
        print(kind,': не знайдено попередній manifest',p);continue
    source_reports[kind]=df
    print('\n',kind.upper(),'рядків:',len(df),'UNASSIGNED:',int((df.split=='UNASSIGNED').sum()) if 'split' in df else 'n/a')
    src='source' if 'source' in df.columns else 'subset'
    if src in df.columns and 'split' in df.columns:
        pivot=pd.crosstab(df[src],df['split']); print(pivot.to_string())
        pivot.to_csv(OUT/f'{kind.upper()}_SOURCE_X_SPLIT.csv')
    if 'group_id' in df.columns and 'split' in df.columns:
        over=df.groupby(['group_id','split'],dropna=False).size().reset_index(name='files')
        over=over.sort_values('files',ascending=False)
        over.head(100).to_csv(OUT/f'{kind.upper()}_LARGEST_GROUPS.csv',index=False)
        print('Найбільші групи:');print(over.head(4).to_string(index=False))
        valid=df[df.group_id.fillna('').ne('') & df.split.ne('UNASSIGNED')]
        conflicts=(valid.groupby('group_id').split.nunique()>1).sum()
        print('Груп у кількох split:',int(conflicts))
        if 'source' in df.columns and 'duration_sec' in df.columns:
            hours=df.pivot_table(index='source',columns='split',values='duration_sec',aggfunc='sum',fill_value=0)/3600
            hours.round(3).to_csv(OUT/f'{kind.upper()}_SOURCE_HOURS_SPLIT.csv')

# %% [markdown]
# ## 4. Відеоманіпуляції — підготовка коректного групування
# Перетворення прив’язуємо до оригіналу з *перевіреного* mapping. Без підтвердженого оригіналу залишаємо `UNASSIGNED`.

# %%
old_video=source_reports.get('video',pd.DataFrame())
if len(old_video) and len(links):
    map_by_variant={str(r.variant_path):str(r.resolved_original) for r in links.itertuples() if r.status=='PASS'}
    updated=old_video.copy()
    newgrp=[];newparent=[];newstatus=[]
    for row in updated.itertuples():
        filepath=str(row.file_path)
        parent=map_by_variant.get(filepath,'')
        if parent:
            filename=Path(parent).name
            m=re.search(r'AVUA_(EXT\d+|P\d+|S\d+)',filename,re.I)
            if m:
                stem=m.group(1).upper()
                group='VIDEO_'+stem if stem.startswith('EXT') else 'PERSON_'+stem
                status='MAPPING_PARENT_FOUND_NAME_GROUP_NEEDS_QA'
            else:
                group='';status='MAPPING_PARENT_NO_ID'
        else:
            group=str(row.group_id) if pd.notna(row.group_id) else ''
            status=str(row.group_status) if pd.notna(row.group_status) else 'UNRESOLVED'
        newgrp.append(group);newparent.append(parent if parent else (str(row.master_parent) if pd.notna(row.master_parent) else ''));newstatus.append(status)
    updated['group_id']=newgrp
    updated['master_parent']=newparent
    updated['group_status']=newstatus
    # Без нового власноруч підтвердженого person_id не перепризначаємо split;
    # беремо існуючий розподіл для cutted/own master у межах ТОЇ САМОЇ групи.
    known=old_video[old_video.split.isin(['development','validation','hidden_test'])]
    candidates=known.groupby('group_id').split.agg(lambda x:set(x)).to_dict()
    assignment={g:next(iter(s)) for g,s in candidates.items() if isinstance(g,str) and g and len(s)==1}
    updated['split']=updated.apply(lambda r:assignment.get(r['group_id'],'UNASSIGNED'),axis=1)
    updated.to_csv(OUT/'VIDEO_LINKED_SPLIT_CANDIDATES.csv',index=False)
    print('\nВІДЕО після відновлення зв’язків:')
    print(pd.crosstab(updated['subset'],updated['split']).to_string())
    print('Невизначених:',int((updated.split=='UNASSIGNED').sum()))
    print('Примітка: це лише попередній split, потрібна звірка фізичних осіб і джерел.')

# %% [markdown]
# ## 5. Корпус ЗУНУ — діагностика структури
# Проблема попереднього коду: файл `wunu-fake_trusted/dataset.csv` прочитався як 2 рядки без колонки `text`. Перевіримо реальну схему, вкладені структури та сусідні файли. Відповідей не вигадуємо.

# %%
wunu_root=ROOT/'00_sources_and_licenses/text/wunu-fake_trusted'
text_keys=['text','article','article_text','body','news','news_text','content','message','post','claim','statement','material','full_text','sentence','paragraph','raw_text','publication','зміст','текст','повідомлення','новина','матеріал']
label_keys=['label','class','category','verdict','truth','fake','is_fake','source_label','target','тип','клас']

def maybe_decode(v):
    if not isinstance(v,str):return v
    s=v.strip()
    if len(s)<2 or s[0] not in '[{':return v
    if len(s)>2_000_000:return v
    for decode in (json.loads,ast.literal_eval):
        try:return decode(s)
        except Exception:pass
    return v

def good_cyrillic(s):
    if not isinstance(s,str):return False
    s=re.sub(r'\s+',' ',s).strip()
    if len(s)<90 or len(s)>100000:return False
    letters=re.findall(r'[A-Za-zА-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]',s)
    cyr=re.findall(r'[А-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]',s)
    return bool(letters) and len(cyr)/len(letters)>.62

def walk_records(x,path='',lab='',depth=0):
    if depth>6:return
    x=maybe_decode(x)
    if isinstance(x,dict):
        lower={str(k).lower():k for k in x}
        text_val='';label=lab
        for key in label_keys:
            if key in lower:
                v=x[lower[key]]
                if isinstance(v,(str,int,float)):
                    label=str(v);break
        for key in text_keys:
            if key in lower:
                v=x[lower[key]]
                if isinstance(v,str) and good_cyrillic(v):
                    text_val=v;break
        if text_val:
            yield (text_val,label,path)
        for k,v in x.items():
            # не дублюємо текст, якщо він уже виділений як окремий запис
            if str(k).lower() in text_keys and isinstance(v,str):continue
            if isinstance(maybe_decode(v),(list,dict)):
                yield from walk_records(v,f'{path}/{k}',label,depth+1)
    elif isinstance(x,list):
        for i,v in enumerate(x):
            yield from walk_records(v,f'{path}[{i}]',lab,depth+1)
    elif isinstance(x,str) and good_cyrillic(x) and ('[' in path or '/text' in path):
        yield (x,lab,path)

wunu_tables=[];wunu_samples=[];candidates=[];seen=set()
paths=[]
if wunu_root.exists():
    paths=[p for p in wunu_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl','.parquet','.tsv','.txt'}]
print('Файлів-кандидатів ЗУНУ:',len(paths))
for p in sorted(paths):
    if p.stat().st_size>300*1024*1024:
        wunu_tables.append(dict(file=rel(p),rows=None,columns='TOO_LARGE',size_mb=round(p.stat().st_size/1e6,2)))
        continue
    d=read_table(p) if p.suffix.lower()!='.txt' else pd.DataFrame()
    if not d.empty:
        wunu_tables.append(dict(file=rel(p),rows=len(d),columns=' | '.join(str(x) for x in d.columns)[:3000],size_mb=round(p.stat().st_size/1e6,2)))
        for i,row in d.iterrows():
            record={str(k):maybe_decode(v) for k,v in row.to_dict().items()}
            for value,label,loc in walk_records(record,f'row={i+1}'):
                h=hashlib.sha256(re.sub(r'\s+',' ',value).strip().casefold().encode()).hexdigest()
                if h not in seen:
                    seen.add(h);candidates.append(dict(source_dataset='WUNU_FAKE_TRUSTED',original_file=rel(p),source_record=loc,source_label_raw=label,text_uk=value,sha256_text=h))
                if len(wunu_samples)<12:
                    wunu_samples.append(dict(file=rel(p),record=loc,text_preview=value[:160],label=label))
    elif p.suffix.lower()=='.txt':
        try:
            s=p.read_text(encoding='utf-8')
            if good_cyrillic(s):
                h=hashlib.sha256(re.sub(r'\s+',' ',s).strip().casefold().encode()).hexdigest()
                if h not in seen:
                    seen.add(h);candidates.append(dict(source_dataset='WUNU_FAKE_TRUSTED',original_file=rel(p),source_record='whole_text_file',source_label_raw='',text_uk=s,sha256_text=h))
        except Exception:pass
pd.DataFrame(wunu_tables,columns=['file','rows','columns','size_mb']).to_csv(OUT/'WUNU_DATASET_SCHEMAS.csv',index=False)
pd.DataFrame(wunu_samples,columns=['file','record','text_preview','label']).to_csv(OUT/'WUNU_SAMPLE_RECORDS.csv',index=False)
print('Унікальних кандидатів ЗУНУ:',len(candidates),'(це не означає, що вони фактологічно перевірені)')
# Тільки чернетки; питання і метадані ще мають пройти ручне приймання.
stage=[]
for j,r in enumerate(candidates[:30],181):
    stage.append({**r,'case_id':f'TXTUA_{j:04d}',
                  'question_uk':'Які фактологічні твердження наведеного повідомлення можна незалежно перевірити?',
                  'question_en':'Which factual claims in the Ukrainian-language message can be independently verified?',
                  'reference_answer_uk':'','reference_answer_en':'','evidence_urls':'',
                  'license_reviewed':'NO','translator_verified':'NO','review_status':'PENDING_EXPERT_REVIEW','release_ready':False})
pd.DataFrame(stage,columns=['case_id','source_dataset','original_file','source_record','source_label_raw','text_uk','sha256_text','question_uk','question_en','reference_answer_uk','reference_answer_en','evidence_urls','license_reviewed','translator_verified','review_status','release_ready']).to_csv(OUT/'WUNU_30_CANDIDATES_FOR_REVIEW.csv',index=False)
print('Підготовлено нових кандидатів:',len(stage),'/ 30')
if len(stage)<30:print('ЗАЛИШИЛОСЯ',30-len(stage),'записів; потрібні коректні дані/інша схема, див. WUNU_DATASET_SCHEMAS.csv')

# %% [markdown]
# ## 6. Тексти — контроль якості UNLP та реєстр перекладів власних кейсів
# Порожні англомовні відповіді не можна вважати готовими до релізу. Усі запропоновані переклади потребують перевірки.

# %%
unlp_path=V2/'staging'/'TEXT_90_CANDIDATES_FOR_REVIEW.csv'
unlp=read_table(unlp_path)
if not unlp.empty:
    def uk_language_flag(x):
        s=str(x or '')
        ru=len(re.findall(r'[ыЫэЭёЁъЪ]',s));uk=len(re.findall(r'[іїєґІЇЄҐ]',s))
        return 'REVIEW_POSSIBLE_NON_UK' if ru>uk+1 else 'REVIEW_LANGUAGE' if ru else 'CHECK_UK_VERIFIED_MANUALLY'
    unlp['language_review_flag']=unlp['text_uk'].fillna('').map(uk_language_flag)
    unlp['needs_evidence']=unlp['evidence_urls'].fillna('').astype(str).str.strip().eq('') if 'evidence_urls' in unlp else True
    unlp['needs_reference_uk']=unlp['reference_answer_uk'].fillna('').astype(str).str.strip().eq('') if 'reference_answer_uk' in unlp else True
    unlp['needs_reference_en']=unlp['reference_answer_en'].fillna('').astype(str).str.strip().eq('') if 'reference_answer_en' in unlp else True
    unlp.to_csv(OUT/'UNLP_60_REVIEW_FLAGS.csv',index=False)
    print('UNLP рядків:',len(unlp));print(unlp['language_review_flag'].value_counts().to_string())
else:print('UNLP staging відсутній:',unlp_path)
own_path=ROOT/'03_text'/'own_cases'/'metadata_own.jsonl'
own=read_table(own_path)
if not own.empty:
    cols=[x for x in ('case_id','question_uk','question_en','reference_answer_uk','reference_answer_en','event_id','source_dataset') if x in own.columns]
    own_review=own[cols].copy()
    for f in ['question_en','reference_answer_en']:
        if f not in own_review:own_review[f]=''
    own_review['review_status']='PENDING_ENGLISH_TRANSLATION_REVIEW'
    own_review.to_csv(OUT/'OWN_90_ENGLISH_FIELDS_REVIEW.csv',index=False)
    print('Власних текстових записів:',len(own),'Порожніх question_en:',int(own_review['question_en'].fillna('').astype(str).str.strip().eq('').sum()),'answer_en:',int(own_review['reference_answer_en'].fillna('').astype(str).str.strip().eq('').sum()))

# %% [markdown]
# ## 7. Висновок і архів
# Звіти показують, де автоматизація відновила зв'язки, а де потрібні нові люди, дозволи або наукова верифікація.

# %%
report=['# AUA v3 — вузький аудит виправлень',f'Збережено (UTC): {datetime.now(timezone.utc).isoformat()}','',f'Набір: `{ROOT}`','']
if len(links):
    report.append(f'- Відеомапінг: {int((links.status=="PASS").sum())}/{len(links)} пар із перевіреним original та variant.')
    report.append(f'- З них відновлено через `cutted`: {int((links.resolution=="MATCHED_CUTTED_BASENAME").sum())}.')
report += [f'- Виявлено кандидатів ЗУНУ: {len(candidates)}; відібрано: {len(stage)}/30.',
           f'- UNLP перевірочних кандидатів: {len(unlp)}.',
           f'- Власних кейсів для англійського перегляду: {len(own)}.',
           '', '## Важливі обмеження',
           '- Це звіти, а не автоматично затверджені дані. Відсутність витоку за `group_id` не є доказом, якщо `speaker_id` або `original_path` вказані некоректно.',
           '- Поки DRSSU або FLEURS об’єднані у величезні групи, 60/20/20 не виконано за матеріалом.',
           '- Наявність текстових рядків не означає наявності перевірених evidence/answer і ліцензій.',
           '- Оригінали медіа та існуючі split не змінювалися.']
(OUT/'README_AUA_V3.md').write_text('\n'.join(report),encoding='utf-8')
zip_path=OUT.parent/f'AUA_REPAIR_V3_REPORTS_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.zip'
with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.glob('*')):
        if p.is_file() and p.suffix.lower() in {'.csv','.md','.txt','.json','.xlsx'}:
            z.write(p,arcname=p.name)
print('\nГотово. ZIP:',zip_path,'\nРозмір:',round(zip_path.stat().st_size/1024**2,2),'MiB')
print('Для наступного аналізу завантажте цей ZIP сюди.')
