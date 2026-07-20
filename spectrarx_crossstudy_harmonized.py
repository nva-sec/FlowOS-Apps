#!/usr/bin/env python3
from __future__ import annotations

import json, math, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd

OUT=Path('outputs'); RAW=Path('work/raw'); MAT=Path('work/matrices')
MODEL_URL='https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz'
EPS=1e-12

def dl(url,p):
 p=Path(p)
 if p.exists() and p.stat().st_size>1000:return
 subprocess.run(['curl','-L','--fail','--retry','8','--retry-delay','5','--connect-timeout','60','--max-time','3600','-o',str(p)+'.part',url],check=True)
 q=Path(str(p)+'.part'); assert q.stat().st_size>1000; q.replace(p)

def norm(x):
 s=re.sub(r'[^A-Z0-9]','',str(x).upper())
 return {'PALBOCICLIB':'PD0332991','PD0332991':'PD0332991','NUTLIN3A':'NUTLIN3','DORAMAPIMOD':'BIRB0796','TRAMETINIB':'GSK1120212','SELUMETINIB':'AZD6244'}.get(s,s)

def findcol(d,terms):
 for c in d.columns:
  z=c.lower()
  if any(t.lower()==z for t in terms):return c
 for c in d.columns:
  z=c.lower()
  if any(t.lower() in z for t in terms):return c
 return None

def bias(y,m,it=20,lam=5.0):
 mu=float(y[m].mean()); a=np.zeros(y.shape[0]); b=np.zeros(y.shape[1])
 for _ in range(it):
  nr=m.sum(1); a=np.divide(np.where(m,y-mu-b[None,:],0).sum(1),nr+lam,out=np.zeros_like(a),where=nr+lam>0)
  nc=m.sum(0); b=np.divide(np.where(m,y-mu-a[:,None],0).sum(0),nc+lam,out=np.zeros_like(b),where=nc+lam>0)
 return mu+a[:,None]+b[None,:]

def spectral_parts(df,k=5,seed=0):
 y=df.to_numpy(float); m=np.isfinite(y); base=bias(y,m); r=np.where(m,y-base,0.0)
 kk=max(1,min(k,min(r.shape)-1)); u,s,vt=randomized_svd(r,n_components=kk,n_iter=5,random_state=seed)
 return u,s,vt,base

def build_prism_named(r,t,c,model):
 # map PRISM response columns to canonical compound names
 column_col=findcol(t,['column_name','column']); name_col=findcol(t,['name','compound_name','drug_name']); broad_col=findcol(t,['broad_id'])
 cmap={}
 if column_col and name_col:
  for _,z in t[[column_col,name_col]].dropna().iterrows():cmap[str(z[column_col])]=norm(z[name_col])
 if broad_col and name_col:
  bmap=dict(zip(t[broad_col].astype(str),t[name_col].astype(str)))
  for x in r.columns:
   if str(x) not in cmap:
    for b,n in bmap.items():
     if b in str(x):cmap[str(x)]=norm(n);break
 groups={}
 for x in r.columns:
  key=cmap.get(str(x),norm(str(x).split('::')[0])); groups.setdefault(key,[]).append(x)
 q=pd.DataFrame(index=r.index)
 for key,cols in groups.items():q[key]=r[cols].median(axis=1,skipna=True)
 # official SIDM <-> ACH bridge
 mid=findcol(model,['model_id']); bid=findcol(model,['broad_id']); assert mid and bid,(model.columns.tolist())
 ach_to_sidm=dict(zip(model[bid].astype(str),model[mid].astype(str)))
 q.index=[ach_to_sidm.get(str(x),str(x)) for x in q.index]
 q.index=[str(x).upper().strip() for x in q.index]
 q=q.groupby(level=0).mean().T.groupby(level=0).mean().T
 # annotation keyed by canonical name
 ann=[]
 target=findcol(t,['target','targets']); moa=findcol(t,['moa','mechanism_of_action']); disease=findcol(t,['indication','disease_area'])
 use=[x for x in [name_col,target,moa,disease] if x]
 if name_col:
  for _,z in t[use].drop_duplicates().iterrows():
   row={'drug':norm(z[name_col]),'display_name':str(z[name_col])}
   if target:row['target']=str(z[target])
   if moa:row['moa']=str(z[moa])
   if disease:row['indication']=str(z[disease])
   ann.append(row)
 return q,pd.DataFrame(ann).drop_duplicates('drug')

def metric_rows(y,pred,names):
 rows=[]
 for j,n in enumerate(names):
  truth=y[:,j]; est=pred[:,j]; ok=np.isfinite(truth)&np.isfinite(est)
  if ok.sum()<20:continue
  a=truth[ok]; z=est[ok]; k=max(1,int(math.ceil(.2*len(a)))); lab=np.zeros(len(a),int);lab[np.argsort(a)[:k]]=1;score=-z;top=np.argsort(z)[:k]
  rows.append({'drug':n,'n':len(a),'rmse':float(np.sqrt(np.mean((a-z)**2))),'spearman':float(spearmanr(a,z).statistic),'ndcg':float(ndcg_score(lab[None,:],score[None,:],k=k)),'ap':float(average_precision_score(lab,score)),'ef':float(lab[top].mean()/max(lab.mean(),EPS))})
 return rows

def main():
 dl(MODEL_URL,RAW/'model_list_latest.csv.gz')
 gdsc=pd.read_pickle(MAT/'GDSC2.pkl'); r=pd.read_csv(RAW/'PRISM_readout.csv',low_memory=False);t=pd.read_csv(RAW/'PRISM_treatment.csv',low_memory=False);c=pd.read_csv(RAW/'PRISM_cells.csv',low_memory=False);model=pd.read_csv(RAW/'model_list_latest.csv.gz',low_memory=False)
 r=r.rename(columns={r.columns[0]:'depmap_id'}).set_index('depmap_id').apply(pd.to_numeric,errors='coerce')
 prism,ann=build_prism_named(r,t,c,model)
 gdsc.index=[str(x).upper().strip() for x in gdsc.index];gdsc.columns=[norm(x) for x in gdsc.columns];gdsc=gdsc.groupby(level=0).mean().T.groupby(level=0).mean().T
 cells=sorted(set(gdsc.index)&set(prism.index));drugs=sorted(set(gdsc.columns)&set(prism.columns))
 pd.DataFrame({'cell':cells}).to_csv(OUT/'common_cells_harmonized.csv',index=False);pd.DataFrame({'drug':drugs}).to_csv(OUT/'common_drugs_harmonized.csv',index=False)
 status={'common_cells':len(cells),'common_drugs':len(drugs),'mapping_source':MODEL_URL}
 if len(cells)<30 or len(drugs)<3:
  status['status']='insufficient_overlap';Path(OUT/'CROSSSTUDY_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));return
 G=gdsc.loc[cells,drugs];P=prism.loc[cells,drugs]
 min_drugs=max(3,int(math.ceil(.5*len(drugs)))); keep=(G.notna().sum(1)>=min_drugs)&(P.notna().sum(1)>=min_drugs);G=G.loc[keep];P=P.loc[keep]
 # retain drugs with enough paired observations
 keepd=[]
 for d in drugs:
  if (G[d].notna()&P[d].notna()).sum()>=30:keepd.append(d)
 G=G[keepd];P=P[keepd]; drugs=keepd
 status.update({'analysis_cells':len(G),'analysis_drugs':len(drugs)})
 if len(G)<30 or len(drugs)<3:
  status['status']='insufficient_dense_overlap';Path(OUT/'CROSSSTUDY_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));return
 # same bias-residual spectral construction in each study
 ug,sg,vg,_=spectral_parts(G,k=min(5,len(drugs)-1),seed=910);up,sp,vp,_=spectral_parts(P,k=min(5,len(drugs)-1),seed=920);k=min(vg.shape[0],vp.shape[0]);C=np.corrcoef(vg[:k],vp[:k])[:k,k:];rr,cc=linear_sum_assignment(-np.abs(C));matched=np.abs(C[rr,cc]);signs=np.sign(C[rr,cc]);
 rng=np.random.default_rng(930);null=[]
 for _ in range(5000):
  X=np.corrcoef(vg[:k],vp[:k,rng.permutation(len(drugs))])[:k,k:];a,b=linear_sum_assignment(-np.abs(X));null.append(float(np.abs(X[a,b]).mean()))
 null=np.asarray(null);obs=float(matched.mean());pval=float((1+(null>=obs).sum())/(1+len(null)))
 pairs=[];load=[]
 annmap=ann.set_index('drug').to_dict('index') if len(ann) else {}
 for rank,(i,j,sgn,cor) in enumerate(zip(rr,cc,signs,matched),1):
  pairs.append({'pair':rank,'gdsc_factor':int(i+1),'prism_factor':int(j+1),'absolute_correlation':float(cor),'signed_correlation':float(C[i,j])})
  consensus=(vg[i]+sgn*vp[j])/2
  for ix in np.argsort(np.abs(consensus))[::-1][:15]:
   d=drugs[ix]; row={'pair':rank,'drug':d,'consensus_loading':float(consensus[ix]),'gdsc_loading':float(vg[i,ix]),'prism_loading_aligned':float(sgn*vp[j,ix]),**annmap.get(d,{})};load.append(row)
 pd.DataFrame(pairs).to_csv(OUT/'factor_recurrence_pairs.csv',index=False);pd.DataFrame(load).to_csv(OUT/'factor_recurrence_top_drugs.csv',index=False)
 recurrence={'mean_matched_absolute_correlation':obs,'matched_absolute_correlations':matched.tolist(),'null_mean':float(null.mean()),'permutation_p':pval,'n_permutations':len(null),'n_factors':k}
 Path(OUT/'factor_recurrence_harmonized.json').write_text(json.dumps(recurrence,indent=2))
 # direct five-fold GDSC -> PRISM transfer on paired observed cells/drugs, z-scored using train folds only
 rows=[];perdrug=[];kf=KFold(5,shuffle=True,random_state=940)
 for fold,(tr,te) in enumerate(kf.split(G)):
  Gtr=G.iloc[tr];Ptr=P.iloc[tr];Gte=G.iloc[te];Pte=P.iloc[te]
  gm=Gtr.mean();gsd=Gtr.std(ddof=0).replace(0,1);pm=Ptr.mean();psd=Ptr.std(ddof=0).replace(0,1)
  Xtr=((Gtr-gm)/gsd).fillna(0).to_numpy();Xte=((Gte-gm)/gsd).fillna(0).to_numpy();Ytr=((Ptr-pm)/psd).fillna(0).to_numpy();Yte=((Pte-pm)/psd).to_numpy()
  u,s,vt=randomized_svd(Xtr,n_components=min(5,Xtr.shape[1]-1,Xtr.shape[0]-1),n_iter=5,random_state=950+fold);Ztr=u*s;Zte=Xte@vt.T
  methods={'PRISM_mean':np.zeros_like(Yte),'raw_GDSC_ridge':Ridge(alpha=10).fit(Xtr,Ytr).predict(Xte),'SpectraRx_factor_transfer':Ridge(alpha=10).fit(Ztr,Ytr).predict(Zte)}
  for m,pred in methods.items():
   ok=np.isfinite(Yte);rmse=float(np.sqrt(np.mean((Yte[ok]-pred[ok])**2)));rho=float(spearmanr(Yte[ok],pred[ok]).statistic);mr=metric_rows(Yte,pred,drugs);rows.append({'fold':fold,'method':m,'rmse':rmse,'spearman':rho,'ndcg':float(np.mean([x['ndcg'] for x in mr])),'ap':float(np.mean([x['ap'] for x in mr])),'ef':float(np.mean([x['ef'] for x in mr]))})
   for x in mr:perdrug.append({'fold':fold,'method':m,**x})
 T=pd.DataFrame(rows);T.to_csv(OUT/'crossstudy_transfer_harmonized.csv',index=False);pd.DataFrame(perdrug).to_csv(OUT/'crossstudy_transfer_per_drug.csv',index=False)
 agg=T.groupby('method',as_index=False).agg(rmse=('rmse','mean'),rmse_sd=('rmse','std'),spearman=('spearman','mean'),ndcg=('ndcg','mean'),ap=('ap','mean'),ef=('ef','mean'));agg.to_csv(OUT/'crossstudy_transfer_summary.csv',index=False)
 status.update({'status':'completed','factor_recurrence':recurrence,'best_transfer_rmse_method':str(agg.loc[agg.rmse.idxmin(),'method']),'best_transfer_ndcg_method':str(agg.loc[agg.ndcg.idxmax(),'method'])});Path(OUT/'CROSSSTUDY_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':main()
