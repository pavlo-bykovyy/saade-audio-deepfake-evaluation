from pathlib import Path
import sys, tempfile, hashlib
sys.path.insert(0,str(Path(__file__).parent))
import saade_v3_core as S
import saade_v3_data as D
import saade_v3_methods as M
import saade_v3_visuals as V
import numpy as np, pandas as pd, soundfile as sf
from dataclasses import replace

base=Path(tempfile.mkdtemp(prefix='saade_v3_fixture_'))
root=base/'AUA'; out=base/'results'
for d in ['01_audio/own_processed','01_audio/external_selected/common_voice','01_audio/external_selected/fleurs',
          '01_audio/synthetic/new-tts','01_audio/synthetic/drssu','00_sources_and_licenses/audio/drssu']:
 (root/d).mkdir(parents=True,exist_ok=True)
rows=[];own=[];tts=[];dr=[]
rng=np.random.default_rng(123)
def clip(folder,name,freq,seconds=4.3):
 p=root/folder/name
 t=np.arange(int(16000*seconds))/16000
 x=(.16*np.sin(2*np.pi*freq*t)+.04*np.sin(2*np.pi*(freq*2.13)*t)+rng.normal(scale=.008,size=len(t))).astype('float32')
 sf.write(p,x,16000)
 return p
for speaker in range(1,8):
 for i in range(1,4):
  name=f'AUA_S{speaker:04d}_READC_{i:03d}.wav';folder='01_audio/external_selected/common_voice';p=clip(folder,name,200+speaker*7+i)
  rows.append(dict(file_name=name,working_path=f'{folder}/{name}',source_dataset='common_voice',speaker_id=f'S{speaker:04d}',
       speaker_id_basis='common_voice_client_id',source_record_id=f'common_voice:{speaker}_{i}.mp3',source_sha256='',working_sha256='',mode='READC',license='CC0'))
for speaker in range(1,4):
 name=f'AUA_S{speaker:03d}_CONVERTED_001.wav';folder='01_audio/own_processed';p=clip(folder,name,230+speaker)
 rows.append(dict(file_name=name,working_path=f'{folder}/{name}',source_dataset='AUA_own',speaker_id=f'S{speaker:03d}',
    speaker_id_basis='own_speaker_register',source_record_id=f'original_{speaker}.wav',source_sha256='',working_sha256='',mode='CONVERTED'))
 own.append(dict(file_name=name,working_path=f'{folder}/{name}',speaker_id=f'S{speaker:03d}',source_record_id=f'original_{speaker}.wav',
                source_path='01_audio/own_master/original.wav'))
for i in range(1,4):
 name=f'AUA_S1108_READC_{i:03d}.wav';folder='01_audio/external_selected/fleurs';clip(folder,name,300+i)
 rows.append(dict(file_name=name,working_path=f'{folder}/{name}',source_dataset='fleurs',speaker_id='S1108',speaker_id_basis='source_group_unknown',source_record_id=f'fleurs:train_{i}.wav'))
for i in range(1,6):
 name=f'AUA_S1109_SYNTH_{i:04d}.wav';folder='01_audio/synthetic/drssu';clip(folder,name,450+i)
 sha=hashlib.sha256((root/folder/name).read_bytes()).hexdigest()
 rows.append(dict(file_name=name,working_path=f'{folder}/{name}',source_dataset='drssu',speaker_id='S1109',speaker_id_basis='source_group_unknown',source_record_id=f'drssu:{i}',source_sha256=sha))
 dr.append(dict(selection_rank=str(i),file_name=f'generated/{i}.wav',source_file=f'generated/{i}.wav',generated='True',sha256=sha))
for voice in ['A','B','C','D','E','F','G']:
 for i in range(1,4):
  name=f'gemini_{voice}_TTS_{i:03d}.wav';folder='01_audio/synthetic/new-tts';clip(folder,name,500+ord(voice)-65+i)
  rows.append(dict(file_name=name,working_path=f'{folder}/{name}',source_dataset='AUA_new_TTS',source_record_id=f'tts_{voice}_{i}',mode='TTS'))
  tts.append(dict(filename=name,model_name='Synthetic Model',voice_id=voice,model_version='1.0',input_text_id=f'text_{i}'))
pd.DataFrame(rows).to_csv(root/'01_audio/metadata_audio.csv',index=False)
pd.DataFrame(own).to_csv(root/'01_audio/own_processed/metadata_audio.csv',index=False)
pd.DataFrame(dr).to_csv(root/'00_sources_and_licenses/audio/drssu/metadata.csv',index=False)
pd.DataFrame(tts).to_csv(root/'01_audio/synthetic/new-tts/metadata_new-tts.csv',index=False)
cfg=S.Config(root=str(root),out=str(out),max_items=35,min_seconds=2.0,epochs=1,
             models=('mfcc_svm',),scenario_profile='v2_5level',bootstraps=20)
x=D.build_verified_catalog(cfg)
assert x.label.notna().all(),x.loc[x.label.isna()].to_string()
assert x[x.source=='DRSSU'].identity_status.eq('unknown_identity').all()
assert x[x.source=='FLEURS'].identity_status.eq('unknown_identity').all()
assert x[x.source=='CommonVoice'].identity_status.eq('speaker_confirmed').all()
assert x[x.source=='AUA_new_TTS'].identity_status.eq('synthetic_voice_confirmed').all()
parents,summary=D.prepare_parent_splits(x,cfg,max_parent_files=35,unknown_train_cap=6)
assert (parents[parents.source.isin(['DRSSU','FLEURS'])].split=='train').all()
readiness=D.data_readiness_report(x,parents,cfg,'exploratory')
seg,_,r=M.expand_nonoverlapping_segments(parents,cfg,target=45)
print('Smoketest segments',len(seg),'labels',seg.groupby(['split','label']).size())
for subset in ['train','validation','hidden_test']:
 assert seg[seg.split==subset].label.nunique()==2
S.lock_hidden_test(seg,cfg)
examples=V.pick_examples(seg,'validation')
assert len(examples)>0
for ex in examples:
 wave=V.waves_for_record(ex,cfg)
 assert len(wave)==21,(len(wave),wave.keys())
model,meta=S.train_one(seg,cfg,'mfcc_svm','static')
base=S.run_scenario_predictions(seg,model,'mfcc_svm','static',cfg,threshold=meta['threshold'],split='validation')
assert len(base)==len(seg[seg.split=='validation'])*21
weights=S.validation_difficulty(base)
model2,meta2=S.train_one(seg,cfg,'mfcc_svm','random')
model3,meta3=S.train_one(seg,cfg,'mfcc_svm','adaptive',weights)
a=S.run_scenario_predictions(seg,model2,'mfcc_svm','random',cfg,meta2['threshold'],split='validation')
b=S.run_scenario_predictions(seg,model3,'mfcc_svm','adaptive',cfg,meta3['threshold'],split='validation')
preds=pd.concat([base,a,b],ignore_index=True)
_,asri,coverage=V.report_scenario_tests(preds,cfg,tag='smoketest',show=False)
print(asri)
assert not asri.empty
print('SMOKE PASS:',root,out)
