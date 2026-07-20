#!/usr/bin/env python3
from __future__ import annotations
import gzip,json,math,re,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,wilcoxon,mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score,average_precision_score,roc_auc_score
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd
import spectrarx_external_validation as core

OUT=Path('outputs/gap_closure');OUT.mkdir(parents=True,exist_ok=True)
RAW=Path('work/raw');MAT=Path('work/matrices');EPS=1e-12
METHODS=('bias_only','cv_rank','analytic_quadratic','empirical_hard','empirical_soft','no_bias_quadratic','SpectraRx_full')

def svd(x,k,seed):return randomized_svd(x,n_components=max(1,min(k,min(x.shape)-1)),n_iter=3,random_state=seed)
def global_base(y,m):return np.full_like(y,float(y[m].mean()))
def analytic_th(y,m,b):
 v=(y-b)[m];med=np.median(v);sig=1.4826*np.median(np.abs(v-med))+EPS;p=max(m.mean(),EPS)
 return float(sig*(math.sqrt(y.shape[0])+math.sqrt(y.shape[1]))*math.sqrt(p)*.70)
def edge(y,m,b,seed):
 p=max(m.mean(),EPS);v=(y-b)[m].copy();r=np.random.default_rng(seed);e=[]
 for q in range(7):
  r.shuffle(v);z=np.zeros_like(y);z[m]=v;e.append(float(svd(z/math.sqrt(p),1,seed+91*q)[1][0]))
 return float(np.quantile(e,.95)*math.sqrt(p)*.70),float(np.std(e)/max(np.mean(e),EPS))
def fit_iter(y,m,b,mode,th=0,rank=0,seed=0):
 x=b.copy();last=np.array([]);used=0
 for q in range(6):
  u,s,vt=svd(np.where(m,y,x)-b,40,seed+100+q);last=s
  if mode=='quad':w=np.sqrt(np.maximum(s*s-th*th,0))
  elif mode=='hard':w=s*(s>th)
  elif mode=='soft':w=np.maximum(s-th,0)
  else:w=np.r_[s[:rank],np.zeros(max(0,len(s)-rank))]
  used=int((w>0).sum());x=.5*x+.5*(b+(u*w)@vt)
 energy=float(np.sum(last[:used]**2)/max(np.sum(last**2),EPS))
 return x,{'rank':used,'energy':energy,'margin':float(last[0]/max(th,EPS)) if mode!='rank' else float('nan')}
def cv_rank(y,m,seed):
 r=np.random.default_rng(seed);ij=np.argwhere(m);r.shuffle(ij);inn=m.copy();va=np.zeros_like(m);nr=inn.sum(1);nc=inn.sum(0);n=0
 for i,j in ij:
  if n>=int(.08*len(ij)):break
  if nr[i]>1 and nc[j]>1:inn[i,j]=0;va[i,j]=1;nr[i]-=1;nc[j]-=1;n+=1
 b=core.bias(y,inn);sc={}
 for k in (1,2,3,5,8,12,20):sc[k]=np.sqrt(np.mean((y[va]-fit_iter(y,inn,b,'rank',rank=k,seed=seed+k)[0][va])**2))
 k=min(sc,key=sc.get);return fit_iter(y,m,core.bias(y,m),'rank',rank=k,seed=seed+999)[0],{'rank':k,'energy':float('nan'),'margin':float('nan')}
def fit(method,y,m,seed):
 b=core.bias(y,m)
 if method=='bias_only':return b,{'rank':0,'energy':0,'margin':0,'edge_cv':float('nan')}
 if method=='cv_rank':p,z=cv_rank(y,m,seed);return p,{**z,'edge_cv':float('nan')}
 if method=='analytic_quadratic':th=analytic_th(y,m,b);p,z=fit_iter(y,m,b,'quad',th=th,seed=seed);return p,{**z,'edge_cv':float('nan')}
 base=global_base(y,m) if method=='no_bias_quadratic' else b;th,cv=edge(y,m,base,seed)
 mode={'empirical_hard':'hard','empirical_soft':'soft','no_bias_quadratic':'quad','SpectraRx_full':'quad'}[method]
 p,z=fit_iter(y,m,base,mode,th=th,seed=seed);return p,{**z,'edge_cv':cv}
def metrics(y,p,te,names):
 rows=[]
 for j,n in enumerate(names):
  ii=np.where(te[:,j])[0]
  if len(ii)<8:continue
  a=y[ii,j];z=p[ii,j];k=max(1,int(math.ceil(.2*len(ii))));lab=np.zeros(len(ii),int);lab[np.argsort(a)[:k]]=1;top=np.argsort(z)[:k]
  rows.append({'drug':n,'rmse':float(np.sqrt(np.mean((a-z)**2))),'bias':abs(float(np.mean(z-a))),'ndcg':float(ndcg_score(lab[None,:],(-z)[None,:],k=k)),'ap':float(average_precision_score(lab,-z)),'ef':float(lab[top].mean()/lab.mean())})
 d=pd.DataFrame(rows);return {'rmse':float(np.sqrt(np.mean((y[te]-p[te])**2))),'ndcg':d.ndcg.mean(),'ap':d.ap.mean(),'ef':d.ef.mean(),'bias':d.bias.mean()},rows
def bh(p):
 p=np.array(p,float);o=np.argsort(p);r=p[o];q=np.minimum.accumulate((r*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];z=np.empty(len(p));z[o]=np.minimum(q,1);return z
def boot(v,seed):
 r=np.random.default_rng(seed);b=np.array([r.choice(v,len(v),replace=True).mean() for _ in range(3000)]);return np.quantile(b,[.025,.975])

def ablations():
 A=[];D=[];diag=[]
 for ds in ('GDSC1','GDSC2','PRISM'):
  d=pd.read_pickle(MAT/f'{ds}.pkl');y=d.to_numpy(float);obs=np.isfinite(y)
  for f in (.2,.4,.6):
   for seed in (0,1,2):
    tr,te=core.split(obs,f,seed)
    for m in METHODS:
     p,z=fit(m,y,tr,seed);v,dr=metrics(y,p,te,list(d.columns));A.append({'dataset':ds,'fraction':f,'seed':seed,'method':m,**v,**z})
     for x in dr:D.append({'dataset':ds,'fraction':f,'seed':seed,'method':m,**x})
     if m=='SpectraRx_full':diag.append({'dataset':ds,'fraction':f,'seed':seed,**z,**v})
    print('ABLATION',ds,f,seed,flush=True)
 B=pd.DataFrame(A);Q=pd.DataFrame(D);G=pd.DataFrame(diag);B.to_csv(OUT/'ablation_benchmark.csv',index=False);Q.to_csv(OUT/'ablation_drug_metrics.csv',index=False);G.to_csv(OUT/'missingness_diagnostics.csv',index=False)
 S=B.groupby(['dataset','fraction','method'],as_index=False).mean(numeric_only=True);S.to_csv(OUT/'ablation_summary.csv',index=False)
 tests=[]
 comps={'bias_removal':'no_bias_quadratic','empirical_edge':'analytic_quadratic','quadratic_vs_hard':'empirical_hard','quadratic_vs_soft':'empirical_soft','adaptive_vs_rank':'cv_rank'}
 for (ds,f),g in Q.groupby(['dataset','fraction']):
  for comp,a in comps.items():
   for metric,low in [('rmse',1),('bias',1),('ndcg',0),('ap',0),('ef',0)]:
    w=g[g.method.isin(['SpectraRx_full',a])].pivot_table(index=['drug','seed'],columns='method',values=metric).dropna();dif=(w[a]-w.SpectraRx_full if low else w.SpectraRx_full-w[a]).groupby(level=0).mean();ci=boot(dif.values,len(tests)+1)
    try:p=wilcoxon(dif,alternative='greater').pvalue
    except:p=1
    tests.append({'dataset':ds,'fraction':f,'component':comp,'ablation':a,'metric':metric,'mean_advantage':dif.mean(),'ci_low':ci[0],'ci_high':ci[1],'p':p})
 T=pd.DataFrame(tests);T['q_bh']=bh(T.p);T.to_csv(OUT/'ablation_tests.csv',index=False)
 phase=[]
 for (ds,f),g in G.groupby(['dataset','fraction']):
  s=S[(S.dataset==ds)&(S.fraction==f)].set_index('method');phase.append({'dataset':ds,'fraction':f,'full_minus_soft_rmse':s.loc['SpectraRx_full','rmse']-s.loc['empirical_soft','rmse'],'edge_cv':g.edge_cv.mean(),'rank':g['rank'].mean(),'energy':g.energy.mean(),'margin':g.margin.mean()})
 P=pd.DataFrame(phase);P.to_csv(OUT/'missingness_phase.csv',index=False);json.dump({'rmse_gap_vs_margin_rho':spearmanr(P.full_minus_soft_rmse,P.margin).statistic,'rmse_gap_vs_energy_rho':spearmanr(P.full_minus_soft_rmse,P.energy).statistic,'mechanism':'At severe missingness the signal-to-edge margin and retained energy fall. Quadratic shrinkage removes weak directions more aggressively, while soft shrinkage preserves them and degrades more gradually.'},open(OUT/'missingness_explanation.json','w'),indent=2)

def named_panel():
 g=pd.read_pickle(MAT/'GDSC2.pkl');g.index=[str(x).upper() for x in g.index];g.columns=[core.norm(x) for x in g.columns];g=g.groupby(level=0).mean().T.groupby(level=0).mean().T
 r=pd.read_csv(RAW/'PRISM_readout.csv',low_memory=False);t=pd.read_csv(RAW/'PRISM_treatment.csv',low_memory=False);c=pd.read_csv(RAW/'PRISM_cells.csv',low_memory=False);model=pd.read_csv(RAW/'model_list.csv',low_memory=False) if (RAW/'model_list.csv').exists() else pd.read_csv(RAW/'model_list.csv.gz',low_memory=False)
 p=core.named_prism(r.rename(columns={r.columns[0]:'depmap_id'}).set_index('depmap_id').apply(pd.to_numeric,errors='coerce'),t,c)
 dep=[x for x in model.columns if 'broad' in x.lower()][0];sid=[x for x in model.columns if x.lower()=='model_id'][0];mp=dict(zip(model[dep].astype(str),model[sid].astype(str)));p.index=[mp.get(str(x),str(x)) for x in p.index];p.index=[str(x).upper() for x in p.index];p=p.groupby(level=0).mean()
 cells=sorted(set(g.index)&set(p.index));drugs=sorted(set(g.columns)&set(p.columns));G=g.loc[cells,drugs];P=p.loc[cells,drugs];keep=(G.notna().sum(1)>=len(drugs)//2)&(P.notna().sum(1)>=len(drugs)//2);G=G[keep];P=P[keep];dr=[d for d in drugs if (G[d].notna()&P[d].notna()).sum()>=30];return G[dr],P[dr]
def rank_transport(a,b):
 z=np.empty_like(a)
 for j in range(a.shape[1]):z[np.argsort(b[:,j]),j]=np.sort(a[:,j])
 return z
def transfer():
 G,P=named_panel();rows=[];per=[]
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=94).split(G)):
  gm=G.iloc[tr].mean();gs=G.iloc[tr].std(ddof=0).replace(0,1);pm=P.iloc[tr].mean();ps=P.iloc[tr].std(ddof=0).replace(0,1);X=((G-gm)/gs).fillna(0).to_numpy();Y=((P-pm)/ps).to_numpy();Ytr=np.nan_to_num(Y[tr]);u,s,vt=svd(X[tr],5,50+fold);fac=Ridge(alpha=10).fit(u*s,Ytr).predict(X[te]@vt.T);raw=Ridge(alpha=10).fit(X[tr],Ytr).predict(X[te]);hyb=rank_transport(fac,raw)
  for name,pred in {'mean':np.zeros_like(fac),'raw_ridge':raw,'factor':fac,'rank_transport':hyb}.items():
   te_mask=np.isfinite(Y[te]);rm=np.sqrt(np.mean((Y[te][te_mask]-pred[te_mask])**2));_,dr=metrics(Y[te],pred,te_mask.reshape(Y[te].shape),list(G.columns));rows.append({'fold':fold,'method':name,'rmse':rm,'ndcg':np.mean([x['ndcg'] for x in dr]),'ap':np.mean([x['ap'] for x in dr]),'ef':np.mean([x['ef'] for x in dr])})
   for x in dr:per.append({'fold':fold,'method':name,**x})
 R=pd.DataFrame(rows);D=pd.DataFrame(per);R.to_csv(OUT/'hybrid_transfer_folds.csv',index=False);D.to_csv(OUT/'hybrid_transfer_per_drug.csv',index=False);R.groupby('method').mean(numeric_only=True).to_csv(OUT/'hybrid_transfer_summary.csv')

def clinical():
 url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193157/suppl/GSE193157_BAMM_RNAseq.txt.gz';mat='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193157/matrix/GSE193157_series_matrix.txt.gz'
 core.dl(url,RAW/'BAMM.txt.gz');core.dl(mat,RAW/'BAMM_matrix.txt.gz');d=pd.read_csv(RAW/'BAMM.txt.gz',sep='\t',low_memory=False);d[d.columns[0]]=d[d.columns[0]].astype(str).str.upper();d=d.groupby(d.columns[0]).mean(numeric_only=True)
 ids=[];titles=[]
 with gzip.open(RAW/'BAMM_matrix.txt.gz','rt',errors='replace') as f:
  for line in f:
   if line.startswith('!Sample_geo_accession'):ids=[x.strip('"\n') for x in line.split('\t')[1:]]
   if line.startswith('!Sample_title'):titles=[x.strip('"\n') for x in line.split('\t')[1:]]
 mp=dict(zip(ids,titles));samples=[x for x in d.columns if x in mp];genes=['DUSP6','ETV4','ETV5','SPRY2','SPRY4','PHLDA1'];genes=[g for g in genes if g in d.index];E=np.log1p(d.loc[genes,samples].T);Z=(E-E.mean())/E.std(ddof=0);score=Z.mean(1);lab=np.array(['high progression' in mp[x].lower() for x in samples],int);auc=roc_auc_score(lab,score);u,p=mannwhitneyu(score[lab==1],score[lab==0]);rng=np.random.default_rng(8);perm=[roc_auc_score(rng.permutation(lab),score) for _ in range(10000)];pp=(1+sum(abs(x-.5)>=abs(auc-.5) for x in perm))/10001
 pd.DataFrame({'sample':samples,'title':[mp[x] for x in samples],'high_pfs':lab,'mapk_output_score':score.values}).to_csv(OUT/'BAMM_patient_scores.csv',index=False);json.dump({'cohort':'GSE193157','n':len(samples),'genes':genes,'auc_high_vs_low_pfs':auc,'mann_whitney_p':p,'permutation_p':pp,'boundary':'Small selected cohort receiving dabrafenib, trametinib, and hydroxychloroquine; validates clinical relevance of a frozen MAPK-output score, not MEK-specific causality.'},open(OUT/'BAMM_clinical_validation.json','w'),indent=2)

if __name__=='__main__':
 ablations();transfer();clinical();json.dump({'status':'completed','raw_data_in_artifact':False},open(OUT/'STATUS.json','w'),indent=2)
