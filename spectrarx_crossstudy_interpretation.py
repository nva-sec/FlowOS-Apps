#!/usr/bin/env python3
from __future__ import annotations

import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import spectrarx_crossstudy_harmonized as h

OUT=Path('outputs'); RAW=Path('work/raw'); MAT=Path('work/matrices')


def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]
    q=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p)); out[order]=np.clip(q,0,1); return out


def bootstrap_ci(v,seed=0,n=5000):
    v=np.asarray(v,float); v=v[np.isfinite(v)]; rng=np.random.default_rng(seed)
    means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(n)])
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def main():
    gdsc=pd.read_pickle(MAT/'GDSC2.pkl')
    r=pd.read_csv(RAW/'PRISM_readout.csv',low_memory=False)
    t=pd.read_csv(RAW/'PRISM_treatment.csv',low_memory=False)
    c=pd.read_csv(RAW/'PRISM_cells.csv',low_memory=False)
    model=pd.read_csv(RAW/'model_list_latest.csv.gz',low_memory=False)
    r=r.rename(columns={r.columns[0]:'depmap_id'}).set_index('depmap_id').apply(pd.to_numeric,errors='coerce')
    prism,ann=h.build_prism_named(r,t,c,model)
    gdsc.index=[str(x).upper().strip() for x in gdsc.index]
    gdsc.columns=[h.norm(x) for x in gdsc.columns]
    gdsc=gdsc.groupby(level=0).mean().T.groupby(level=0).mean().T
    cells=sorted(set(gdsc.index)&set(prism.index)); drugs=sorted(set(gdsc.columns)&set(prism.columns))
    G=gdsc.loc[cells,drugs]; P=prism.loc[cells,drugs]
    min_drugs=max(3,int(math.ceil(.5*len(drugs))))
    keep=(G.notna().sum(1)>=min_drugs)&(P.notna().sum(1)>=min_drugs); G=G.loc[keep]; P=P.loc[keep]
    drugs=[d for d in drugs if (G[d].notna()&P[d].notna()).sum()>=30]
    G=G[drugs]; P=P[drugs]
    ug,sg,vg,_=h.spectral_parts(G,k=min(5,len(drugs)-1),seed=910)
    up,sp,vp,_=h.spectral_parts(P,k=min(5,len(drugs)-1),seed=920)
    k=min(vg.shape[0],vp.shape[0]); C=np.corrcoef(vg[:k],vp[:k])[:k,k:]
    rr,cc=linear_sum_assignment(-np.abs(C)); signs=np.sign(C[rr,cc])
    annmap=ann.set_index('drug').to_dict('index') if len(ann) else {}
    load_rows=[]
    for pair,(i,j,sgn) in enumerate(zip(rr,cc,signs),1):
        consensus=(vg[i]+sgn*vp[j])/2
        for ix,d in enumerate(drugs):
            load_rows.append({'pair':pair,'gdsc_factor':int(i+1),'prism_factor':int(j+1),'drug':d,'consensus_loading':float(consensus[ix]),'absolute_loading':float(abs(consensus[ix])),'gdsc_loading':float(vg[i,ix]),'prism_loading_aligned':float(sgn*vp[j,ix]),**annmap.get(d,{})})
    L=pd.DataFrame(load_rows); L.to_csv(OUT/'factor_recurrence_all_loadings.csv',index=False)
    # Mechanism enrichment using absolute consensus loadings and matched-size drug-label permutations.
    rng=np.random.default_rng(1201); enrich=[]
    for pair,g in L.groupby('pair'):
        labels={}
        for idx,row in g.iterrows():
            moa=str(row.get('moa',''))
            if moa.lower()=='nan': continue
            for token in re.split(r'[,;/]',moa):
                token=token.strip().lower()
                if token: labels.setdefault(token,[]).append(idx)
        scores=g.absolute_loading.to_numpy(); index_to_pos={idx:p for p,idx in enumerate(g.index)}
        for mechanism,idxs in labels.items():
            pos=np.array([index_to_pos[x] for x in sorted(set(idxs)) if x in index_to_pos],int)
            if len(pos)<3 or len(pos)>len(g)-3: continue
            mask=np.zeros(len(g),bool); mask[pos]=1
            observed=float(scores[mask].mean()-scores[~mask].mean())
            null=np.empty(5000)
            for b in range(5000):
                sel=rng.choice(len(g),len(pos),replace=False); m=np.zeros(len(g),bool); m[sel]=1
                null[b]=scores[m].mean()-scores[~m].mean()
            enrich.append({'pair':int(pair),'mechanism':mechanism,'n_drugs':int(len(pos)),'mean_absolute_loading_in':float(scores[mask].mean()),'mean_absolute_loading_out':float(scores[~mask].mean()),'enrichment_difference':observed,'permutation_p':float((1+(null>=observed).sum())/(1+len(null)))})
    E=pd.DataFrame(enrich)
    if len(E): E['q_bh']=bh(E.permutation_p)
    E.sort_values(['q_bh','enrichment_difference'],ascending=[True,False]).to_csv(OUT/'factor_moa_enrichment.csv',index=False)
    # Independent-drug inference for transfer metrics.
    D=pd.read_csv(OUT/'crossstudy_transfer_per_drug.csv')
    stats=[]
    for comparator in ['PRISM_mean','raw_GDSC_ridge']:
        for metric,lower in [('rmse',True),('spearman',False),('ndcg',False),('ap',False),('ef',False)]:
            W=D.pivot_table(index=['drug','fold'],columns='method',values=metric).dropna()
            if comparator not in W or 'SpectraRx_factor_transfer' not in W: continue
            diff=(W[comparator]-W['SpectraRx_factor_transfer']) if lower else (W['SpectraRx_factor_transfer']-W[comparator])
            bydrug=diff.groupby(level='drug').mean(); lo,hi=bootstrap_ci(bydrug.values,seed=1301)
            try: p=float(wilcoxon(bydrug,alternative='greater').pvalue)
            except Exception: p=1.0
            stats.append({'comparator':comparator,'metric':metric,'n_drugs':int(len(bydrug)),'mean_advantage':float(bydrug.mean()),'ci_low':lo,'ci_high':hi,'fraction_drugs_improved':float((bydrug>0).mean()),'wilcoxon_p_one_sided':p})
    S=pd.DataFrame(stats)
    if len(S): S['q_bh']=bh(S.wilcoxon_p_one_sided)
    S.to_csv(OUT/'crossstudy_transfer_independent_drug_tests.csv',index=False)
    top_enrich=E.iloc[0].to_dict() if len(E) else None
    status={'status':'completed','n_cells':int(len(G)),'n_drugs':int(len(drugs)),'n_factor_pairs':int(k),'top_mechanism_enrichment':top_enrich,'significant_mechanism_tests':int(((E.q_bh<.05)&(E.enrichment_difference>0)).sum()) if len(E) else 0,'significant_transfer_tests':int(((S.q_bh<.05)&(S.mean_advantage>0)).sum()) if len(S) else 0}
    Path(OUT/'INTERPRETATION_STATUS.json').write_text(json.dumps(status,indent=2,default=float))
    print(json.dumps(status,indent=2,default=float))

if __name__=='__main__': main()
