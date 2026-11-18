from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import sys, csv, wave, struct
sys.path.insert(0, str(Path(__file__).resolve().parent))

import saade_final_data as D

def writecsv(path, head, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=head);w.writeheader();w.writerows(rows)
def wav(path,offset):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(struct.pack('<'+ 'h'*16000, *([offset]*16000)))
with TemporaryDirectory() as tmp:
    root=Path(tmp)/'AUA'
    rows=[]
    for i,(rank,sha) in enumerate([(1,'abc'),(1,'abc'),(2,'not_matches')],1):
        name=f'AUA_S1109_SYNTH_{i:04d}.wav'
        p=root/'01_audio/synthetic/drssu'/name
        wav(p,i)
        rows.append(dict(file_name=name,working_path=str(p.relative_to(root)),source_dataset='drssu',source_record_id=f'drssu:{rank}',source_sha256=sha,working_sha256=f'w{i}',speaker_id='S1109',speaker_id_basis='source_group_unknown'))
    tname='omnivoice_TTS_001.wav'
    wav(root/'01_audio/synthetic/new-tts'/tname,2)
    writecsv(root/'01_audio/metadata_audio.csv',list(rows[0]),rows)
    writecsv(root/'00_sources_and_licenses/audio/drssu/metadata.csv',['selection_rank','file_name','generated','sha256'],[
        dict(selection_rank='1',file_name='generated/4308.wav',generated='True',sha256='abc'),
        dict(selection_rank='2',file_name='generated/4309.wav',generated='True',sha256='mismatch'),
    ])
    writecsv(root/'01_audio/synthetic/new-tts/metadata_new-tts.csv',['filename','model_name','voice_id'],[
        dict(filename=tname,model_name='OmniVoice',voice_id='')
    ])
    x=D.build_verified_catalog(SimpleNamespace(root=root,out=Path(tmp)/'out',min_seconds=0.5))
    print(x[['relative_path','label','label_source','usable','warning']].to_string(index=False))
    assert len(x)==4
    dr=x[x.source=='DRSSU'].sort_values('relative_path')
    assert dr.label.notna().sum()==2
    assert dr.usable.sum()==1
    assert 'DRSSU_DUPLICATE_SOURCE_SHA' in dr.iloc[1].warning
    assert 'DRSSU_SOURCE_SHA_CONFLICT' in dr.iloc[2].warning
    tt=x[x.source=='AUA_new_TTS'].iloc[0]
    assert tt.label==1 and tt.usable and tt.split_role=='train_only'
    print('ALL TESTS PASS')
