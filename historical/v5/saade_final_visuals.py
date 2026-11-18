"""SAADE v2 visual and audio companion for the user-supplied AUA research notebook.

Usage:
    import saade_visuals as V
    V.show_audio_gallery(splits.iloc[0], cfg, show_players=True)
    V.report_scenario_tests(validation_predictions, cfg, tag='validation')

No source data are modified. Only figures/CSV are written to <cfg.out>/visuals/<tag>/.
No raw audio is exported automatically; notebook playback remains local to JupyterLab.
This module never opens hidden-test data unless the caller passes those predictions.
"""
from __future__ import annotations
import math, re, hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import saade_final_core as S

CATEGORY_UA={
    'baseline':'Оригінал',
    'noise':'Гаусівський шум',
    'compression':'MP3-компресія',
    'reverberation':'Реверберація (синтетична)',
    'lowpass':'Низькочастотний фільтр',
}
CATEGORY_EN={'baseline':'Baseline', 'noise':'Noise', 'compression':'MP3',
             'reverberation':'Reverb', 'lowpass':'Low-pass'}
REGIME_EN={'static':'E1 · Static','random':'E2 · Random','adaptive':'E3 · Adaptive'}
METRIC_INFO={
    'balanced_accuracy':('Balanced Accuracy','0–1; більше = краще',0,1),
    'f1_fake':('F1 для FAKE','0–1; більше = краще',0,1),
    'roc_auc':('ROC-AUC','0–1; більше = краще',0,1),
    'eer':('EER','0–1; менше = краще',0,1),
    'false_alarm_rate':('False Alarm Rate','частка REAL помилково як FAKE',0,1),
    'miss_rate':('Miss Rate','частка FAKE помилково як REAL',0,1),
}


def scenario_keys(cfg):
    return [('baseline',0)]+[(c,k) for c in S.CATEGORIES for k in S.available_levels(cfg.scenario_profile)]


def label_for_scenario(category, level, cfg):
    if category=='baseline':return '00 · Original / baseline'
    severity=S.scenario_severity(category,int(level),cfg.scenario_profile)
    unit={'noise':'dB SNR','compression':'kbps MP3','reverberation':'s RT60*','lowpass':'Hz'}[category]
    return f'{CATEGORY_EN[category]} L{int(level)} · {severity:g} {unit}'


def all_scenario_metadata(cfg):
    rows=[]
    for i,(cat,lev) in enumerate(scenario_keys(cfg)):
        value=S.scenario_severity(cat,lev,cfg.scenario_profile)
        rows.append(dict(order=i,category=cat,category_ua=CATEGORY_UA[cat],level=int(lev),
                         parameter=value,description=label_for_scenario(cat,lev,cfg),
                         scenario_key=f'{cat}::{lev}',
                         scenario_profile=cfg.scenario_profile))
    return pd.DataFrame(rows)


def _out(cfg,tag='validation'):
    folder=Path(cfg.out)/'visuals'/S.safe_key(tag)
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def _save(fig,path,show=True):
    fig.savefig(path,dpi=160,bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def _row_as_series(record):
    if isinstance(record,pd.DataFrame):return record.iloc[0]
    if isinstance(record,pd.Series):return record
    if isinstance(record,dict):return pd.Series(record)
    return pd.Series(record._asdict())


def pick_examples(splits, split='validation'):
    """Picks 1 genuine and 1 fake only from requested split (default VALIDATION)."""
    d=splits[splits.split.eq(split)].copy()
    if d.empty:raise ValueError(f'Empty split: {split}')
    results=[]
    for label in [0,1]:
        x=d[d.label.eq(label)].sort_values(['source','relative_path'])
        if not x.empty:results.append(x.iloc[0])
    return results


def waves_for_record(record,cfg):
    """Generate the exact scenario audio that is fed to the models, respecting profile and cache."""
    r=_row_as_series(record)
    required=['path','relative_path']
    missing=[k for k in required if k not in r or not str(r[k]).strip()]
    if missing:raise ValueError('Audio row missing: '+','.join(missing))
    def attr(name,default=''):
        return r.get(name,default)
    class R:
        pass
    waves={}
    for cat,lev in scenario_keys(cfg):
        rr=R()
        rr.path=str(attr('path'))
        rr.segment_start_sec=attr('segment_start_sec',None) # V2_SEGMENT_VISUAL
        rr.relative_path=str(attr('relative_path'))
        rr.category=cat
        rr.level=lev
        rr.label=int(attr('label',0))
        rr.source=str(attr('source'))
        rr.group_id=str(attr('group_id'))
        rr.speaker_id=str(attr('speaker_id'))
        rr.generator=str(attr('generator'))
        rr.split=str(attr('split','validation'))
        waves[(cat,lev)]=S.get_scenario_wave(rr,cfg)
    return waves


def _spec(x,sr):
    _,_,z=signal.stft(x,fs=sr,nperseg=min(512,len(x)),noverlap=min(384,max(0,len(x)//2)),
                      boundary=None)
    return 20*np.log10(np.maximum(1e-7,np.abs(z)))


def _wave_stats(waves,cfg):
    baseline=np.asarray(waves[('baseline',0)],float)
    x0=baseline[:len(baseline)]
    rows=[]
    for order,(cat,lev) in enumerate(scenario_keys(cfg)):
        x=np.asarray(waves[(cat,lev)],float)
        rms=float(np.sqrt(np.mean(x*x)+1e-12))
        error=x-x0
        diff_rms=float(np.sqrt(np.mean(error*error)+1e-12))
        corr=float(np.corrcoef(x,x0)[0,1]) if np.std(x)>1e-8 and np.std(x0)>1e-8 else np.nan
        rows.append(dict(order=order,category=cat,level=lev,
             description=label_for_scenario(cat,lev,cfg),duration_sec=len(x)/cfg.sample_rate,
             rms=rms,peak=float(np.max(np.abs(x))),change_rms=diff_rms,
             correlation_with_original=corr,
             inferred_snr_db=(10*np.log10((np.mean(x0*x0)+1e-12)/(np.mean(error*error)+1e-12))
                if cat=='noise' else np.nan)))
    return pd.DataFrame(rows)


def show_audio_gallery(record,cfg,show_players=True,show_figures=True,tag='validation',max_players=None):
    """All 13 (v1) or 21 (v2) real playable scenario audio items and figures.

    Audio player labels represent local transforms. Normalization is ONLY for audition;
    the exact waveform fed to a model remains unaltered in generated figures and scores.
    """
    r=_row_as_series(record)
    label=int(r.get('label',0)); group=str(r.get('group_id',''))
    ident=hashlib.sha256(str(r.get('relative_path','')).encode()).hexdigest()[:10]
    print(f'КЛАС: {"FAKE / synthetic" if label==1 else "REAL / bona fide"}; '
          f'джерело: {r.get("source", "?")}; фрагмент ID: {ident}; '
          f'профіль: {cfg.scenario_profile}')
    print('Відтворення сегмента:',r.get('segment_start_sec','центр запису'),'с; тривалість',cfg.seconds,'с')
    waves=waves_for_record(r,cfg)
    info=_wave_stats(waves,cfg)
    path=_out(cfg,tag)
    info.to_csv(path/f'audio_scenario_stats_{ident}.csv',index=False)
    from IPython.display import display, Audio, Markdown
    display(info[['description','rms','peak','change_rms','correlation_with_original','inferred_snr_db']].round(4))

    keys=scenario_keys(cfg)
    ncol=3
    nrow=math.ceil(len(keys)/ncol)
    if show_figures:
        t=np.arange(int(cfg.seconds*cfg.sample_rate))/cfg.sample_rate
        fig,axs=plt.subplots(nrow,ncol,figsize=(16,2.3*nrow),sharex=True,sharey=True,squeeze=False)
        for ax,(cat,lev) in zip(axs.flat,keys):
            x=waves[(cat,lev)]
            ax.plot(t[:len(x)][::32],x[::32],linewidth=.55)
            ax.set_title(label_for_scenario(cat,lev,cfg),fontsize=9)
            ax.grid(alpha=.2)
        for ax in axs.flat[len(keys):]:ax.axis('off')
        fig.suptitle(f'ВСІ {len(keys)} СЦЕНАРІЇ · waveform · ID {ident}',y=1.003,fontsize=15)
        fig.supxlabel('Час, с');fig.supylabel('Амплітуда PCM');fig.tight_layout()
        _save(fig,path/f'audio_all_waveforms_{ident}.png',show=True)

        fig,axs=plt.subplots(nrow,ncol,figsize=(16,2.75*nrow),sharex=True,sharey=True,squeeze=False)
        for ax,(cat,lev) in zip(axs.flat,keys):
            x=waves[(cat,lev)]
            f,tt,zz=signal.stft(x,fs=cfg.sample_rate,nperseg=512,noverlap=384)
            ax.pcolormesh(tt,f/1000,20*np.log10(np.maximum(1e-7,np.abs(zz))),
                          shading='auto',cmap='magma',vmin=-85,vmax=-20)
            ax.set_ylim(0,8);ax.set_title(label_for_scenario(cat,lev,cfg),fontsize=9)
        for ax in axs.flat[len(keys):]:ax.axis('off')
        fig.suptitle(f'ВСІ {len(keys)} СЦЕНАРІЇ · спектрограми STFT · ID {ident}',y=1.003,fontsize=15)
        fig.supxlabel('Час, с');fig.supylabel('Частота, кГц');fig.tight_layout()
        _save(fig,path/f'audio_all_spectrograms_{ident}.png',show=True)

    if show_players:
        nshow=len(keys) if max_players is None else max(1,min(int(max_players),len(keys)))
        print(f'▶ ПРОСЛУХОВУВАННЯ: {nshow}/{len(keys)} сценаріїв. Звук нормалізується тільки у плеєрі.')
        for cat,lev in keys[:nshow]:
            display(Markdown('**'+label_for_scenario(cat,lev,cfg)+'**'))
            display(Audio(waves[(cat,lev)],rate=cfg.sample_rate,normalize=True))
    return info


def draw_scenario_guide(cfg,tag='validation'):
    guide=all_scenario_metadata(cfg)
    out=_out(cfg,tag)
    guide.to_csv(out/'00_scenario_guide.csv',index=False)
    print(f'Протокол {cfg.scenario_profile}: {len(guide)} умов; {len(guide)-1} акустичних впливів.')
    from IPython.display import display
    display(guide[['description','category_ua','level','parameter']])
    fig,axs=plt.subplots(1,4,figsize=(16,3),sharex=False)
    for ax,c in zip(axs,S.CATEGORIES):
        g=guide[guide.category.eq(c)];x=g.level.to_numpy();y=g.parameter.to_numpy(dtype=float)
        ax.plot(x,y,marker='o');ax.grid(alpha=.3)
        ax.set_title(CATEGORY_UA[c]);ax.set_xlabel('Рівень впливу');
        unit={'noise':'SNR, dB ↓','compression':'MP3, kbps ↓','reverberation':'RT60*, c ↑','lowpass':'Частота зрізу, Hz ↓'}[c]
        ax.set_ylabel(unit);ax.set_xticks(x)
    fig.suptitle('Параметри 4 типів спотворень (ordered v2 / legacy v1)')
    fig.tight_layout();_save(fig,out/'00_scenario_levels.png')
    return guide


def _with_scenarios(df,cfg):
    a=df.copy()
    a['category']=a['category'].astype(str)
    a['level']=pd.to_numeric(a['level'],errors='coerce').astype('Int64')
    order={k:i for i,k in enumerate(scenario_keys(cfg))}
    a['key']=list(zip(a.category,a.level.fillna(-1).astype(int)))
    a['scenario_order']=a['key'].map(order)
    a['scenario_label']=[label_for_scenario(c,l,cfg) if (c,l) in order else f'{c} L{l}'
                         for c,l in zip(a.category,a.level.fillna(-1))]
    return a


def _coerce_predictions(preds,cfg):
    if preds is None or len(preds)==0:raise ValueError('Empty prediction table; run E1/E2/E3 or load CSV.')
    needed={'scenario_id','label','score','threshold','model','regime','category','level','split'}
    missing=needed-set(preds.columns)
    if missing:raise ValueError(f'Predictions missing columns: {sorted(missing)}')
    x=preds.copy()
    for k in ['label','score','threshold','level']:
        x[k]=pd.to_numeric(x[k],errors='coerce')
    x=x[x.label.isin([0,1]) & x.score.notna() & x.threshold.notna() & x.level.notna()].copy()
    x['level']=x['level'].astype(int)
    x['label']=x['label'].astype(int)
    x['pred']=(x.score>=x.threshold).astype(int)
    x['correct']=(x.pred==x.label).astype(int)
    x=_with_scenarios(x,cfg)
    if x.scenario_order.isna().any():
        unknown=x.loc[x.scenario_order.isna(),['category','level']].drop_duplicates()
        raise ValueError('Incompatible scenario protocol. Unexpected (category,level): '+str(unknown.to_dict('records')))
    return x


def scenario_metric_table(predictions,cfg):
    """One row per model, regime, category, severity, with BOTH classes required.
    Uses threshold locked for each model from clean validation, not reoptimized per scenario.
    """
    p=_coerce_predictions(predictions,cfg)
    rows=[]
    for (model,regime,cat,level),g in p.groupby(['model','regime','category','level'],sort=False):
        meta=dict(model=model,regime=regime,category=cat,level=int(level),
                  scenario_label=label_for_scenario(cat,level,cfg),
                  n=len(g),n_real=int((g.label==0).sum()),n_fake=int((g.label==1).sum()),
                  n_groups=int(g.group_id.nunique()) if 'group_id' in g.columns else np.nan,
                  split=' | '.join(sorted(g.split.astype(str).unique())))
        if g.label.nunique()!=2:
            meta.update(status='CHECK_ONE_CLASS',**{k:np.nan for k in METRIC_INFO})
        elif g.score.nunique()<2:
            values=S.metric_row(g.label,g.score,float(g.threshold.iloc[0]))
            meta.update(status='CHECK_CONSTANT_SCORES',**values)
        else:
            meta.update(status='OK',**S.metric_row(g.label,g.score,float(g.threshold.iloc[0])))
        rows.append(meta)
    r=pd.DataFrame(rows)
    r=_with_scenarios(r,cfg).sort_values(['model','regime','scenario_order'])
    return r


def plot_source_class_balance(predictions,cfg,tag='validation'):
    p=_coerce_predictions(predictions,cfg)
    x=p[p.category.eq('baseline')].drop_duplicates(['scenario_id'])
    group=x.groupby(['source','label']).size().unstack(fill_value=0).reindex(columns=[0,1],fill_value=0)
    group.columns=['REAL','FAKE']
    out=_out(cfg,tag)
    group.to_csv(out/'source_class_balance.csv')
    fig,ax=plt.subplots(figsize=(max(8,min(16,len(group)*1.5)),4))
    group.plot.bar(stacked=True,ax=ax,rot=35)
    ax.set_xlabel('Джерело');ax.set_ylabel('Незалежні базові фрагменти')
    ax.set_title('Баланс REAL/FAKE за джерелом · перевірка source confounding')
    ax.legend(title='Клас');fig.tight_layout();_save(fig,out/'source_class_balance.png')
    return group.reset_index()


def plot_scenario_heatmaps(metrics,cfg,tag='validation',show=True):
    """Show ALL scenarios for ALL models x regimes for accuracy, EER, F1, miss/FA rates."""
    out=_out(cfg,tag)
    profiles=[]
    order=[label_for_scenario(c,l,cfg) for c,l in scenario_keys(cfg)]
    for metric in ['balanced_accuracy','eer','f1_fake','roc_auc','false_alarm_rate','miss_rate']:
        x=metrics.copy()
        x['configuration']=x['model']+' · '+x['regime'].map(lambda s:REGIME_EN.get(s,s))
        mat=x.pivot_table(index='configuration',columns='scenario_label',values=metric,aggfunc='first',dropna=False)
        mat=mat.reindex(columns=order)
        mat.to_csv(out/f'table_ALL_scenarios_{metric}.csv')
        fig,ax=plt.subplots(figsize=(max(13,0.75*len(order)),max(4,0.48*len(mat)+2)))
        image=ax.imshow(mat.values,aspect='auto',cmap='viridis' if metric not in ('eer','miss_rate','false_alarm_rate') else 'magma_r',vmin=0,vmax=1)
        ax.set_xticks(range(len(order)),order,rotation=55,ha='right',fontsize=8)
        ax.set_yticks(range(len(mat)),mat.index,fontsize=9)
        ax.set_title(f'{METRIC_INFO[metric][0]} за ВСІМА {len(order)} сценаріями · {tag}')
        ax.set_xlabel('Акустичний сценарій');ax.set_ylabel('Архітектура / режим')
        for i in range(len(mat)):
            for j in range(len(order)):
                v=mat.iloc[i,j]
                if pd.notna(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,
                                     color='white' if (.2<v<.7) else 'black')
        fig.colorbar(image,ax=ax,label=METRIC_INFO[metric][1],fraction=.02,pad=.01)
        fig.tight_layout(); _save(fig,out/f'heatmap_ALL_scenarios_{metric}.png',show)
        profiles.append(mat)
    return profiles


def plot_per_category_curves(metrics,cfg,tag='validation',show=True):
    """For each acoustic category: lines for E1 E2 E3, separately for each model."""
    out=_out(cfg,tag)
    for metric in ['balanced_accuracy','eer']:
        for c in S.CATEGORIES:
            d=metrics[metrics.category.eq(c)].copy()
            if d.empty:continue
            names=sorted(d.model.unique())
            fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False,sharey=True)
            for ax,name in zip(axes[0],names):
                z=d[d.model.eq(name)]
                for regime in ['static','random','adaptive']:
                    g=z[z.regime.eq(regime)].sort_values('level')
                    if not g.empty:ax.plot(g.level,g[metric],marker='o',label=REGIME_EN[regime])
                ax.set_title(name);ax.set_xticks(S.available_levels(cfg.scenario_profile))
                ax.set_xlabel('Рівень акустичного впливу');ax.set_ylim(0,1);ax.grid(alpha=.25)
            axes[0][0].set_ylabel(METRIC_INFO[metric][0]+' [0–1]')
            axes[0][-1].legend(fontsize=8,loc='best')
            fig.suptitle(f'{CATEGORY_UA[c]} · {METRIC_INFO[metric][0]} · {tag}',y=1.02)
            fig.tight_layout();_save(fig,out/f'curves_{c}_{metric}.png',show)


def plot_baseline_degradation(metrics,cfg,tag='validation',show=True):
    """All scenarios relative to SAME model/regime baseline on SAME evaluation split."""
    out=_out(cfg,tag)
    d=metrics.copy()
    d['configuration']=d.model+' · '+d.regime
    base=d[d.category.eq('baseline')][['configuration','balanced_accuracy']].rename(columns={'balanced_accuracy':'ba_baseline'})
    d=d.merge(base,on='configuration',how='left',validate='many_to_one')
    d['delta_ba_vs_clean']=d.balanced_accuracy-d.ba_baseline
    d=d[d.category.ne('baseline')]
    d.to_csv(out/'table_all_degradation_vs_clean.csv',index=False)
    scenarios=[label_for_scenario(c,l,cfg) for c,l in scenario_keys(cfg)[1:]]
    mat=d.pivot_table(index='configuration',columns='scenario_label',values='delta_ba_vs_clean',aggfunc='first')
    mat=mat.reindex(columns=scenarios)
    fig,ax=plt.subplots(figsize=(max(12,.7*len(scenarios)),max(4,.45*len(mat)+2)))
    extent=np.nanmax(abs(mat.to_numpy(dtype=float))) if np.isfinite(mat.to_numpy(dtype=float)).any() else .5
    extent=max(.05,float(extent))
    im=ax.imshow(mat.values,aspect='auto',cmap='coolwarm_r',vmin=-extent,vmax=extent)
    ax.set_xticks(range(len(scenarios)),scenarios,rotation=55,ha='right',fontsize=8)
    ax.set_yticks(range(len(mat)),mat.index)
    ax.set_title('Втрата / приріст Balanced Accuracy відносно CLEAN baseline')
    for i in range(len(mat)):
        for j in range(len(scenarios)):
            v=mat.iloc[i,j]
            if pd.notna(v):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=ax,label='Δ Balanced Accuracy (scenario − baseline)',fraction=.02,pad=.01)
    fig.tight_layout();_save(fig,out/'heatmap_delta_BA_vs_baseline.png',show)
    return d


def plot_asri(asri,cfg,tag='validation',show=True):
    out=_out(cfg,tag)
    if asri is None or asri.empty:return
    table=asri.pivot(index='model',columns='regime',values='asri')
    table=table.reindex(columns=[k for k in ['static','random','adaptive'] if k in table.columns])
    table.to_csv(out/'table_asri.csv')
    fig,ax=plt.subplots(figsize=(max(7,len(table)*2.5),4.3))
    table.plot.bar(ax=ax,rot=0,ylim=(0,1))
    ax.set_title(f'ASRI за {4*len(S.available_levels(cfg.scenario_profile))} акустичними сценаріями · {tag}')
    ax.set_ylabel('ASRI [0,1]');ax.set_xlabel('Архітектура');ax.grid(axis='y',alpha=.25)
    fig.tight_layout();_save(fig,out/'ASRI_comparison.png',show)


def plot_error_confusions(predictions,cfg,model='spectrogram_cnn',regime='adaptive',tag='validation',show=True):
    """For ONE selected model/regime show confusion matrices for ALL 13/21 scenarios."""
    p=_coerce_predictions(predictions,cfg)
    d=p[p.model.eq(model)&p.regime.eq(regime)]
    if d.empty:
        print('No model/regime predictions:',model,regime);return pd.DataFrame()
    keys=scenario_keys(cfg)
    ncols=4;nrows=math.ceil(len(keys)/ncols)
    fig,axs=plt.subplots(nrows,ncols,figsize=(14,3.2*nrows),squeeze=False)
    rows=[]
    for ax,(cat,lev) in zip(axs.flat,keys):
        z=d[d.category.eq(cat)&d.level.eq(lev)]
        tn=int(((z.label==0)&(z.pred==0)).sum());fp=int(((z.label==0)&(z.pred==1)).sum())
        fn=int(((z.label==1)&(z.pred==0)).sum());tp=int(((z.label==1)&(z.pred==1)).sum())
        cm=np.array([[tn,fp],[fn,tp]])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,cm.max()))
        ax.set_xticks([0,1],['REAL','FAKE']);ax.set_yticks([0,1],['REAL','FAKE'])
        ax.set_xlabel('Прогноз');ax.set_ylabel('Еталон')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11)
        ax.set_title(label_for_scenario(cat,lev,cfg),fontsize=9)
        rows.append(dict(category=cat,level=lev,n=len(z),tn=tn,fp=fp,fn=fn,tp=tp))
    for ax in axs.flat[len(keys):]:ax.axis('off')
    fig.suptitle(f'ВСІ confusion matrices: {model} / {regime} / {tag}',y=1.001,fontsize=15)
    fig.tight_layout();_save(fig,_out(cfg,tag)/f'confusion_ALL_{S.safe_key(model)}_{S.safe_key(regime)}.png',show)
    result=pd.DataFrame(rows)
    result.to_csv(_out(cfg,tag)/f'confusion_ALL_{S.safe_key(model)}_{S.safe_key(regime)}.csv',index=False)
    return result


def show_one_record_decisions(predictions,record,cfg,tag='validation',show=True):
    """All scenario predictions for ALL model/regime configurations for ONE source file."""
    p=_coerce_predictions(predictions,cfg);r=_row_as_series(record)
    ref=str(r.relative_path)
    z=p[p.relative_path.astype(str).eq(ref)]
    if z.empty:
        print('No predictions for this sample in this evaluation split.');return pd.DataFrame()
    colorder=[label_for_scenario(c,l,cfg) for c,l in scenario_keys(cfg)]
    z=z.copy();z['config']=z.model+' · '+z.regime
    pivot=z.pivot_table(index='config',columns='scenario_label',values='correct',aggfunc='first').reindex(columns=colorder)
    nrows=max(3,int(.6*len(pivot)+1))
    fig,ax=plt.subplots(figsize=(max(13,.65*len(colorder)),nrows))
    im=ax.imshow(pivot.values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1)
    ax.set_xticks(range(len(colorder)),colorder,rotation=65,ha='right',fontsize=8)
    ax.set_yticks(range(len(pivot)),pivot.index,fontsize=9)
    for i in range(len(pivot)):
        for j in range(len(colorder)):
            v=pivot.iloc[i,j]
            if pd.notna(v):ax.text(j,i,'✓' if int(v)==1 else '✗',ha='center',va='center',fontsize=11)
    ax.set_title(f'Правильність прогнозу для ОДНОГО аудіозапису · {tag}\n'
                 'Зелений = правильно · червоний = помилка; score не зіставляти між архітектурами')
    fig.tight_layout();id10=hashlib.sha256(ref.encode()).hexdigest()[:10]
    _save(fig,_out(cfg,tag)/f'one_audio_{id10}_model_decisions.png',show)
    table=z[['model','regime','scenario_label','label','score','threshold','pred','correct']].sort_values(['model','regime','scenario_label'])
    table.to_csv(_out(cfg,tag)/f'one_audio_{id10}_scenario_decisions.csv',index=False)
    from IPython.display import display
    display(table.head(40).round(4))
    return table


def scenario_failure_examples(predictions,cfg,tag='validation',max_examples=150):
    """Hard examples (score crosses threshold incorrectly); path is kept server-local, never audio."""
    p=_coerce_predictions(predictions,cfg)
    p['wrong_margin']=np.where(p.correct.eq(0),abs(p.score-p.threshold),0)
    failures=p[p.correct.eq(0)].sort_values('wrong_margin',ascending=False)
    cols=['model','regime','category','level','relative_path','source','generator','group_id','label','pred','score','threshold','wrong_margin']
    cols=[c for c in cols if c in failures.columns]
    fail=failures[cols].head(max_examples)
    fail.to_csv(_out(cfg,tag)/'failure_examples_for_review.csv',index=False)
    return fail


def report_scenario_tests(predictions,cfg,tag='validation',show=True,confusion_model=None,
                          confusion_regime='adaptive',include_source_plot=True):
    """Report scenario-wise results for every model and regime on current split.

    Does not train models or open hidden-test audio. All charts and tables are saved to
    a tag-specific folder; validation and hidden cannot overwrite one another.
    """
    p=_coerce_predictions(predictions,cfg)
    observed=sorted(p.split.astype(str).unique())
    print('ТИП ОЦІНЮВАННЯ:', tag, '| фактичні split:', observed)
    if tag=='validation' and observed!=['validation']:
        warnings.warn('The validation dashboard received non-validation rows. Review before interpreting.')
    if tag=='hidden_test' and observed!=['hidden_test']:
        warnings.warn('The hidden dashboard received non-hidden data. Review before final publication.')
    out=_out(cfg,tag)
    metrics=scenario_metric_table(p,cfg)
    metrics.to_csv(out/'ALL_SCENARIO_METRICS.csv',index=False)
    asri=S.asri_table(p,expected_levels=len(S.available_levels(cfg.scenario_profile)))
    asri.to_csv(out/'ASRI_BY_MODEL_REGIME.csv',index=False)
    required=len(scenario_keys(cfg))
    counts=metrics.groupby(['model','regime']).agg(
        scenarios=('scenario_label','nunique'),
        min_n=('n','min'),max_n=('n','max'),
        min_real=('n_real','min'),min_fake=('n_fake','min'),
        n_failed_cells=('status',lambda x:(x=='CHECK_ONE_CLASS').sum()),
        n_constant_score_warnings=('status',lambda x:(x=='CHECK_CONSTANT_SCORES').sum())).reset_index()
    counts['expected_scenarios']=required
    counts['complete']=(counts.scenarios.eq(required)&counts.min_real.gt(0)&counts.min_fake.gt(0)&counts.n_failed_cells.eq(0))
    counts.to_csv(out/'ALL_SCENARIO_COVERAGE.csv',index=False)
    from IPython.display import display
    print(f'Покриття ВСІХ сценаріїв ({required}) для кожної моделі × режиму:')
    display(counts)
    print('ASRI:')
    display(asri.sort_values(['model','regime']))
    summary=metrics[metrics.category.ne('baseline')].groupby(['model','regime']).agg(
        mean_balanced_accuracy=('balanced_accuracy','mean'),worst_balanced_accuracy=('balanced_accuracy','min'),
        mean_EER=('eer','mean'),mean_F1_fake=('f1_fake','mean'),mean_ROC_AUC=('roc_auc','mean')).reset_index()
    display(summary.round(4));summary.to_csv(out/'MODEL_SUMMARY_ALL_SCENARIOS.csv',index=False)
    plot_asri(asri,cfg,tag,show)
    plot_scenario_heatmaps(metrics,cfg,tag,show)
    plot_per_category_curves(metrics,cfg,tag,show)
    plot_baseline_degradation(metrics,cfg,tag,show)
    if include_source_plot:plot_source_class_balance(p,cfg,tag)
    if confusion_model is None:
        confusion_model='spectrogram_cnn' if 'spectrogram_cnn' in p.model.unique() else str(p.model.iloc[0])
    if confusion_regime not in p.regime.unique():confusion_regime=str(p.regime.iloc[0])
    plot_error_confusions(p,cfg,confusion_model,confusion_regime,tag,show)
    failures=scenario_failure_examples(p,cfg,tag)
    print(f'Помилкових рішень (усі сценарії × моделі): {(p.correct==0).sum():,}. '
          f'Реєстр прикладів: {len(failures)} рядків; аудіофайли не копіюються.')
    print('Графіки / таблиці:',out)
    return metrics,asri,counts


def plot_roc_for_selected(predictions,cfg,model='spectrogram_cnn',regime='adaptive',tag='validation',show=True):
    """ROC baseline vs strongest severity in each of 4 groups; labels need two classes."""
    from sklearn.metrics import roc_curve,roc_auc_score
    p=_coerce_predictions(predictions,cfg)
    z=p[p.model.eq(model)&p.regime.eq(regime)]
    if z.empty:return
    max_level=max(S.available_levels(cfg.scenario_profile))
    fig,axs=plt.subplots(2,2,figsize=(12,9),squeeze=False)
    for ax,cat in zip(axs.flat,S.CATEGORIES):
        for scenario,level in [('baseline',0),(cat,max_level)]:
            d=z[z.category.eq(scenario)&z.level.eq(level)]
            if d.label.nunique()!=2:continue
            fpr,tpr,_=roc_curve(d.label,d.score)
            auc=roc_auc_score(d.label,d.score)
            ax.plot(fpr,tpr,label=f'{scenario} L{level} · AUC {auc:.3f}')
        ax.plot([0,1],[0,1],linestyle='--',alpha=.4)
        ax.set_title(CATEGORY_UA[cat]);ax.set_xlabel('FPR');ax.set_ylabel('TPR');ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f'ROC: baseline проти найсильнішого рівня · {model}/{regime}/{tag}')
    fig.tight_layout();_save(fig,_out(cfg,tag)/f'roc_baseline_vs_worst_{S.safe_key(model)}_{S.safe_key(regime)}.png',show)


def make_interactive_player(splits,cfg,split='validation'):
    """Optional ipywidgets explorer: select sample, category, level; play exact transformed wave."""
    try:
        import ipywidgets as W
        from IPython.display import Audio, display, clear_output
    except ImportError:
        print('Для інтерактивного меню встановіть `%pip install ipywidgets`. '
              'Без нього функція show_audio_gallery() уже показує всі плеєри.')
        return None
    d=splits[splits.split.eq(split)].copy().sort_values(['label','source','relative_path'])
    if d.empty:raise ValueError(f'No records in {split}')
    opts=[(f'{"FAKE" if int(r.label) else "REAL"} · {r.source} · {str(r.relative_path).split("/")[-1][:65]}',int(i))
         for i,r in d.iterrows()]
    clip=W.Dropdown(options=opts,description='Файл:',layout=W.Layout(width='90%'))
    scenarios=scenario_keys(cfg)
    scenario_opts=[(label_for_scenario(cat,lev,cfg),(cat,lev)) for cat,lev in scenarios]
    case=W.Dropdown(options=scenario_opts,description='Сценарій:',layout=W.Layout(width='90%'))
    go=W.Button(description='▶ Показати й відтворити',button_style='info')
    output=W.Output(layout=W.Layout(border='1px solid #ddd',padding='6px'))
    def render(_):
        with output:
            clear_output(wait=True)
            row=d.loc[clip.value]
            cat,level=case.value
            waves=waves_for_record(row,cfg)
            original=waves[('baseline',0)]
            changed=waves[(cat,level)]
            print(f'ID: {row.relative_path}; REAL/FAKE: {int(row.label)}; scenario={label_for_scenario(cat,level,cfg)}')
            display(Audio(changed,rate=cfg.sample_rate,normalize=True))
            t=np.arange(len(changed))/cfg.sample_rate
            fig,axes=plt.subplots(2,1,figsize=(12,5))
            axes[0].plot(t[::24],original[::24],label='Baseline',alpha=.8)
            axes[0].plot(t[::24],changed[::24],label='Scenario',alpha=.7)
            axes[0].set_title('Waveform: original vs scenario');axes[0].legend()
            f,tt,zz=signal.stft(changed,fs=cfg.sample_rate,nperseg=512,noverlap=384)
            axes[1].pcolormesh(tt,f/1000,20*np.log10(np.maximum(1e-7,abs(zz))),
                                shading='auto',cmap='magma')
            axes[1].set_ylim(0,8);axes[1].set_xlabel('Time, s');axes[1].set_ylabel('Frequency, kHz')
            axes[1].set_title('Spectrogram of selected scenario')
            fig.tight_layout();plt.show();plt.close(fig)
    go.on_click(render)
    display(W.VBox([clip,case,go,output]));return (clip,case,go,output)
