"""Engineering-only fixture for REAL-TTS paired splits: synthetic waves NOT TTS evidence."""
from pathlib import Path
import tempfile,sys,hashlib,os
import numpy as np,pandas as pd,soundfile as sf
sys.path.insert(0,str(Path(__file__).parent))
import saade_final_core as S
import saade_final_methods as M
import saade_final_adaptive as A

def test_pairs():
 with tempfile.TemporaryDirectory(prefix='saade_v5_pairs_') as tmp:
  base=Path(tmp);root=base/'AUA';root.mkdir();out=base/'results';fake_dir=out/'paired_tts_generated';fake_dir.mkdir(parents=True)
  cfg=S.Config(root=str(root),out=str(out),seconds=1.,min_seconds=.8,scenario_profile='v2_5level')
  original=[];receipts=[];t=np.arange(16000)/16000
  for split,n in [('train',10),('validation',3),('hidden_test',3)]:
   for i in range(n):
    pid=f'PAIR_{split}_{i}';speech=f'Український тестовий текст номер {split} {i}.'
    p=root/f'good_{pid}.wav';f=fake_dir/f'fake_{pid}.wav'
    sf.write(p,.08*np.sin(2*np.pi*(200+i+{'train':0,'validation':1000,'hidden_test':2000}[split])*t),16000)
    sf.write(f,.08*np.sin(2*np.pi*(300+i+{'train':0,'validation':1000,'hidden_test':2000}[split])*t),16000)
    h=hashlib.sha256(p.read_bytes()).hexdigest();hf=hashlib.sha256(f.read_bytes()).hexdigest()
    original.append(dict(path=str(p),relative_path=p.name,group_id='SP::'+split+str(i),
      speaker_id='SP'+split+str(i),source='CommonVoice',generator='',voice_id='',
      source_sha256=h,working_sha256=h,split_role='eligible',label=0,split=split,
      identity_status='speaker_confirmed',duration_sec=1.,parent_file_id=p.name,
      transcription=speech,source_record_id=p.name,group_confidence='metadata_client_id'))
    receipts.append(dict(pair_id=pid,original_path=str(p),generated_path=str(f),
      split=split,voice_id='VOICE_'+split+str(i),model_id='piper_fixture',sha256=hf,text=speech))
  rfile=out/'receipts.csv';pd.DataFrame(receipts).to_csv(rfile,index=False)
  pairs=A.integrate_tts_pairs(pd.DataFrame(original),rfile,cfg,min_pairs_per_split=3)
  assert pairs.pair_id.nunique()==16 and pairs.groupby('pair_id').label.nunique().eq(2).all()
  folds=A.create_matched_adaptation_fold(pairs,cfg,seed=42,target_fraction=.32)
  assert folds.groupby('pair_id').split.nunique().eq(1).all()
  seg,_,_=M.expand_nonoverlapping_segments(folds,cfg,target=800)
  balanced=A.balance_pairs(seg,cfg)
  assert balanced.groupby(['split','label']).size().unstack().pipe(lambda x:x[0].eq(x[1]).all())
  info,ready=A.research_readiness(balanced,cfg,mode='matched_tts',trained_seeds=(42,123,777))
  print('readiness',ready)
  assert ready['candidate_for_strict_review'],ready
  # A TTS file with a mismatched transcript must fail.
  bad=pd.DataFrame(receipts)
  bad.loc[0,'text']='Це зовсім інший текст.'
  badfile=out/'bad_receipts.csv';bad.to_csv(badfile,index=False)
  try:A.integrate_tts_pairs(pd.DataFrame(original),badfile,cfg)
  except ValueError as e:assert 'does not match' in str(e)
  else:raise AssertionError('Mismatched TTS transcript was not blocked')
  # Attempt to open hidden without explicit authorization must fail.
  try:A.run_hidden_once(balanced,cfg,authorization=False)
  except PermissionError:pass
  else:raise AssertionError('Hidden opened without authorization')
  assert not (out/'V5_HIDDEN_TEST_STARTED.json').exists()
  if os.environ.get('SAADE_V5_LONG_HIDDEN_FIXTURE')=='1':
    # Optional exhaustive fixture. Deliberately not part of fast smoke tests.
    predictions,_=A.evaluate_all_seeds(balanced,cfg,models=('mfcc_svm',),
                         seeds=(42,123,777),epochs=1,update_every=1,adapt_max_per_class=2)
    _,asri,_,_=A.metric_tables(predictions,cfg)
    assert len(asri)==9 and asri.complete_cells.eq(20).all()
    protocol=A.freeze_for_hidden_once(balanced,cfg,models=('mfcc_svm',),
                          seeds=(42,123,777),epochs=1,mode='matched_tts')
    assert protocol.is_file()
    result=A.run_hidden_once(balanced,cfg,authorization=True)
    assert len(result)==9 and (out/'V5_HIDDEN_TEST_STARTED.json').is_file()
    try:A.run_hidden_once(balanced,cfg,authorization=True)
    except FileExistsError:pass
    else:raise AssertionError('Second hidden-test execution was not blocked')
  print('PASS: paired case, independent components, 50/50, TTS receipts, hidden authorization gate')
if __name__=='__main__':test_pairs()
