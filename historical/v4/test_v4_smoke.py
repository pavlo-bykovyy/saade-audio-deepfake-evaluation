"""Synthetic audio fixture checks engineering execution only, NOT dissertation evidence."""
from pathlib import Path
import tempfile,sys
import numpy as np,pandas as pd,soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parent))
import saade_v3_core as S
import saade_v4_adaptive as A
import torch
torch.set_num_threads(2)

def main():
  with tempfile.TemporaryDirectory(prefix='saade_v4_check_') as t:
    base=Path(t);aud=base/'AUA'/'audio';aud.mkdir(parents=True);cfg=S.Config(root=str(base/'AUA'),out=str(base/'results'),
       scenario_profile='v2_5level',seconds=1.,min_seconds=.8,sample_rate=16000,
       epochs=2,augmentation_probability=.5,use_cache=True)
    rr=[]
    for label in (0,1):
      for i in range(5):
        p=aud/f'{label}_{i}.wav'
        x=np.sin(2*np.pi*(150+label*100+i*6)*np.arange(16000)/16000).astype('float32')*.08
        sf.write(p,x,16000)
        sp='train' if i<3 else ('validation' if i==3 else 'hidden_test')
        rr.append(dict(path=str(p),relative_path=p.name,group_id=f'SP::{label}::{i}',
           speaker_id=f'SP{label}{i}',source='fixture',generator='demo',
           voice_id=f'SP{label}{i}',source_sha256='hash-'+str(label)+str(i),
           working_sha256='hash-'+str(label)+str(i),split_role='eligible',
           label=label,split=sp,identity_status='speaker_confirmed',
           duration_sec=1.,parent_file_id=p.name,segment_start_sec=0.,
           transcription='Тестовий аудіозапис для експериментального контролю.',
           source_record_id=p.name))
    parents=pd.DataFrame(rr)
    pf=A.create_adaptation_fold(parents,cfg,target_fraction=.22)
    assert (pf.groupby('split').label.nunique()>=2).all()
    # Add deliberately skewed number of segments within each fold, then balance.
    df=pf.copy()
    df=pd.concat([df,df[(df.split=='train')&(df.label==0)].copy().assign(relative_path=lambda x:x.relative_path+'::duplicate_window')],ignore_index=True)
    final,info=A.balance_per_partition(df,cfg)
    assert info.real_after.eq(info.fake_after).all()
    assert final.groupby('group_id').split.nunique().max()==1
    # End-to-end evaluation, 21 × 3 regimes; one epoch is merely a smoke test.
    preds,adapt=A.evaluate_all_seeds(final,cfg,models=('mfcc_svm',),seeds=(42,),epochs=2,
                                    update_every=1,adapt_max_per_class=1)
    metrics,asri,paired,stats=A.metric_tables(preds,cfg)
    expected=1*1*3*2*21
    print('rows:',len(preds),'expected:',expected)
    assert len(preds)==expected
    assert set(preds.regime)=={'static','random','adaptive'}
    assert len(asri)==3 and asri.complete_cells.eq(20).all()
    bs=A.stratified_voice_bootstrap(preds,cfg,B=30,seed=14)
    assert bs.status.eq('INSUFFICIENT_INDEPENDENT_VOICES').all()
    hist=base/'results'/'models'/'mfcc_svm_adaptive_seed42_policy_history.csv'
    assert hist.is_file(),'No online adaptive updates recorded'
    assert not list(base.glob('Melnyk_dus/**'))
    cnn,cnn_meta,_=A.train_curriculum_model(final,cfg,'raw_wave_cnn','random',42,epochs=1,adapt_max_per_class=1)
    assert cnn_meta['training_draws']>0
    spec,sm,_=A.train_curriculum_model(final,cfg,'spectrogram_cnn','random',42,epochs=1,adapt_max_per_class=1)
    assert sm['training_draws']>0
    print('SMOKE PASS: all 21 scenarios, dynamic policy update, 50/50 balance, 3 regimes, raw CNN, untouched hidden test')

if __name__=='__main__':main()
