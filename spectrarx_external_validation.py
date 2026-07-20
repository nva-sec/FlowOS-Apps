#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[k]="1"
METHODS=("bias","soft_impute","SpectraRx"); FRACTIONS=(.2,.4,.6); SEEDS=(0,1,2); EPS=1e-12
URLS={
 "GDSC1":"https://raw.githubusercontent.com/CS-BIO/MolDr/main/GDSC1/data/GDSC1/merged_result.csv",
 "GDSC2":"https://raw.githubusercontent.com/CS-BIO/MolDr/main/GDSC2/data/GDSC2/merged_result.csv",
 "PRISM_readout":"https://ndownloader.figshare.com/files/20237709",
 "PRISM_treatment":"https://ndownloader.figshare.com/files/20237715",
 "PRISM_cells":"https://ndownloader.figshare.com/files/20237718"}

def dl(url,p):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.stat().st_size>1000:return
 subprocess.run(["curl","-L","--fail","--retry","8","--retry-delay","5","--connect-timeout","60","--max-time","3600","-o",str(p)+".part",url],check=True)
 q=Path(str(p)+".part"); assert q.stat().st_size>1000,(url,q.stat().st_size); q.replace(p)

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def norm(x):
 s=re.sub(r"[^A-Z0-9]","",str(x).upper()); return {"PALBOCICLIB":"PD0332991","NUTLIN3A":"NUTLIN3","DORAMAPIMOD":"BIRB0796"}.get(s,s)

def load_gdsc(p,name):
 d=pd.read_csv(p,low_memory=False); c="cell_names"; v="LN_IC50"; drug="drug_names" if "drug_names" in d else "drug_names.1"
 x=d[[c,drug,v]].dropna(); x[v]=pd.to_numeric(x[v],errors="coerce"); x=x.dropna(); m=x.pivot_table(index=c,columns=drug,values=v,aggfunc="mean")
 m=m.loc[m.notna().sum(1)>=5,m.notna().sum(0)>=20]
 return m,{"dataset":name,"input_rows":len(d),"n_cell_lines":m.shape[0],"n_drugs":m.shape[1],"n_observed":int(m.notna().sum().sum()),"sha256":sha(p),"source":URLS[name]}

def load_prism(rp,tp,cp):
 r=pd.read_csv(rp,low_memory=False); t=pd.read_csv(tp,low_memory=False); c=pd.read_csv(cp,low_memory=False)
 r=r.rename(columns={r.columns[0]:"depmap_id"}).set_index("depmap_id").apply(pd.to_numeric,errors="coerce")
 r=r.loc[r.notna().sum(1)>=50,r.notna().sum(0)>=50]
 return r,t,c,{"dataset":"PRISM","n_cell_lines":r.shape[0],"n_drugs":r.shape[1],"n_observed":int(r.notna().sum().sum()),"sha256":sha(rp),"source":URLS["PRISM_readout"]}

def svd(x,k,seed):return randomized_svd(x,n_components=max(1,min(k,min(x.shape)-1)),n_iter=3,random_state=seed)
def bias(y,m,it=16,lam=5):
 mu=float(y[m].mean()); a=np.zeros(y.shape[0]); b=np.zeros(y.shape[1])
 for _ in range(it):
  nr=m.sum(1); a=np.divide(np.where(m,y-mu-b[None,:],0).sum(1),nr+lam,out=np.zeros_like(a),where=nr+lam>0)
  nc=m.sum(0); b=np.divide(np.where(m,y-mu-a[:,None],0).sum(0),nc+lam,out=np.zeros_like(b),where=nc+lam>0)
 return mu+a[:,None]+b[None,:]
def soft(y,m,seed):
 base=bias(y,m); z=np.where(m,y-base,0); _,s,_=svd(z,1,seed); th=.15*s[0]; x=base.copy()
 for q in range(6):
  u,s,vt=svd(np.where(m,y,x)-base,30,seed+100+q); x=base+(u*np.maximum(s-th,0))@vt
 return x,{"rank":int((s>th).sum()),"threshold":float(th)}
def spectra(y,m,seed):
 base=bias(y,m); vals=(y-base)[m].copy(); rng=np.random.default_rng(seed); edges=[]; p=max(m.mean(),EPS)
 for q in range(5):
  rng.shuffle(vals); z=np.zeros_like(y); z[m]=vals; edges.append(float(svd(z/math.sqrt(p),1,seed+200+q)[1][0]))
 th=float(np.quantile(edges,.95)*math.sqrt(p)*.70); x=base.copy()
 for q in range(6):
  u,s,vt=svd(np.where(m,y,x)-base,30,seed+300+q); shr=np.sqrt(np.maximum(s*s-th*th,0)); x=.5*x+.5*(base+(u*shr)@vt)
 return x,{"rank":int((s>th).sum()),"threshold":th,"null_edge":float(np.quantile(edges,.95))}
def split(obs,f,seed):
 rng=np.random.default_rng(10000+seed); ij=np.argwhere(obs); pick=rng.choice(len(ij),int(round(f*len(ij))),replace=False); te=np.zeros_like(obs,bool); te[tuple(ij[pick].T)]=1; return obs&~te,te
def metrics(y,p,te,names):
 rows=[]
 for j,n in enumerate(names):
  ii=np.where(te[:,j])[0]
  if len(ii)<8:continue
  a=y[ii,j]; z=p[ii,j]; k=max(1,int(math.ceil(.2*len(ii)))); lab=np.zeros(len(ii),int);lab[np.argsort(a)[:k]]=1; score=-z; top=np.argsort(z)[:k]
  rows.append({"drug":str(n),"drug_rmse":float(np.sqrt(np.mean((a-z)**2))),"ndcg":float(ndcg_score(lab[None,:],score[None,:],k=k)),"ap":float(average_precision_score(lab,score)),"ef":float(lab[top].mean()/max(lab.mean(),EPS)),"recall":float(lab[top].sum()/lab.sum())})
 d=pd.DataFrame(rows); rho=spearmanr(y[te],p[te]).statistic
 out={"rmse":float(np.sqrt(np.mean((y[te]-p[te])**2))),"spearman":float(rho)}
 for c in ("ndcg","ap","ef","recall"):out[c]=float(d[c].mean())
 return out,d
def one_task(t):
 name,path,f,seed=t; d=pd.read_pickle(path); y=d.to_numpy(float); tr,te=split(np.isfinite(y),f,seed); A=[];D=[]
 for method in METHODS:
  st=time.time()
  if method=="bias":p,meta=bias(y,tr),{}
  elif method=="soft_impute":p,meta=soft(y,tr,seed)
  else:p,meta=spectra(y,tr,seed)
  o,q=metrics(y,p,te,list(d.columns)); A.append({"dataset":name,"fraction":f,"seed":seed,"method":method,"runtime":time.time()-st,**o,**{f"meta_{k}":v for k,v in meta.items()}})
  for r in q.to_dict("records"):D.append({"dataset":name,"fraction":f,"seed":seed,"method":method,**r})
 return A,D

def bh(p):
 p=np.asarray(p,float);o=np.argsort(p);r=p[o];q=np.minimum.accumulate((r*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];z=np.empty(len(p));z[o]=np.minimum(q,1);return z
def infer(B,D):
 out=[]
 for (ds,f),g in B.groupby(["dataset","fraction"]):
  comp=g[g.method!="SpectraRx"].groupby("method").rmse.mean().idxmin(); q=D[(D.dataset==ds)&(D.fraction==f)]
  for metric,lower in (("drug_rmse",1),("ndcg",0),("ap",0),("ef",0)):
   w=q.pivot_table(index=["drug","seed"],columns="method",values=metric).dropna();
   if comp not in w or "SpectraRx" not in w:continue
   diff=(w[comp]-w.SpectraRx if lower else w.SpectraRx-w[comp]).groupby(level=0).mean(); rng=np.random.default_rng(9); boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(3000)])
   try:pv=float(wilcoxon(diff,alternative="greater").pvalue)
   except:pv=1.0
   out.append({"dataset":ds,"fraction":f,"metric":metric,"comparator":comp,"n_drugs":len(diff),"mean_advantage":diff.mean(),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),"win_fraction":(diff>0).mean(),"p":pv})
 z=pd.DataFrame(out);z["q_bh"]=bh(z.p);return z

def col(d,terms):
 for x in d.columns:
  if any(t in x.lower() for t in terms):return x
 return None
def named_prism(r,t,c):
 cc=col(t,["column_name"]); nn=col(t,["name","compound_name"]); bb=col(t,["broad_id"]); mp={}
 if cc and nn:
  for _,x in t[[cc,nn]].dropna().iterrows():mp[str(x[cc])]=norm(x[nn])
 if bb and nn:
  bmap=dict(zip(t[bb].astype(str),t[nn].astype(str)))
  for x in r.columns:
   if str(x) not in mp:
    for b,n in bmap.items():
     if b in str(x):mp[str(x)]=norm(n);break
 groups={}
 for x in r.columns:groups.setdefault(mp.get(str(x),norm(str(x).split("::")[0])),[]).append(x)
 q=pd.DataFrame(index=r.index)
 for n,x in groups.items():q[n]=r[x].median(1)
 dep=col(c,["depmap_id"]); sid=col(c,["sanger_model_id","sanger_id"])
 if dep and sid:
  cm=dict(zip(c[dep].astype(str),c[sid].astype(str)));q.index=[cm.get(str(x),str(x)) for x in q.index]
 q.index=[norm(x) for x in q.index];return q.groupby(level=0).mean().T.groupby(level=0).mean().T

def cross(g,p,out):
 g=g.copy();g.index=[norm(x) for x in g.index];g.columns=[norm(x) for x in g.columns];g=g.groupby(level=0).mean().T.groupby(level=0).mean().T
 cells=sorted(set(g.index)&set(p.index));drugs=sorted(set(g.columns)&set(p.columns));status={"common_cells":len(cells),"common_drugs":len(drugs)}
 pd.Series(cells).to_csv(out/"common_cells.csv",index=False);pd.Series(drugs).to_csv(out/"common_drugs.csv",index=False)
 if len(cells)<30 or len(drugs)<3:status["status"]="insufficient_overlap";return status
 G=g.loc[cells,drugs];P=p.loc[cells,drugs];ok=(G.notna().sum(1)>=max(2,len(drugs)//2))&(P.notna().sum(1)>=max(2,len(drugs)//2));G=G[ok];P=P[ok]
 G=(G-G.mean())/G.std(ddof=0);P=(P-P.mean())/P.std(ddof=0);G=G.fillna(0);P=P.fillna(0);k=min(5,len(drugs)-1,len(G)-1)
 ug,sg,vg=svd(G.to_numpy(),k,91);up,sp,vp=svd(P.to_numpy(),k,92);C=np.corrcoef(vg,vp)[:k,k:];rr,cc=linear_sum_assignment(-abs(C));obs=float(abs(C[rr,cc]).mean());rng=np.random.default_rng(93);nul=[]
 for _ in range(2000):
  X=np.corrcoef(vg,vp[:,rng.permutation(vp.shape[1])])[:k,k:];a,b=linear_sum_assignment(-abs(X));nul.append(float(abs(X[a,b]).mean()))
 fact={"matched":abs(C[rr,cc]).tolist(),"mean_match":obs,"null_mean":float(np.mean(nul)),"permutation_p":float((1+np.sum(np.array(nul)>=obs))/2001)};Path(out/"factor_recurrence.json").write_text(json.dumps(fact,indent=2))
 Xraw=G.to_numpy();Xspec=ug*sg;Y=P.to_numpy();rows=[]
 for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=94).split(G)):
  for m,X in (("PRISM_mean",None),("raw_GDSC_ridge",Xraw),("SpectraRx_factor_transfer",Xspec)):
   pred=np.tile(Y[tr].mean(0),(len(te),1)) if X is None else Ridge(alpha=10).fit(X[tr],Y[tr]).predict(X[te]); rows.append({"fold":fold,"method":m,"rmse":float(np.sqrt(np.mean((Y[te]-pred)**2))),"spearman":float(spearmanr(Y[te].ravel(),pred.ravel()).statistic)})
 T=pd.DataFrame(rows);T.to_csv(out/"crossstudy_transfer.csv",index=False);status.update({"status":"completed","analysis_cells":len(G),"factor_mean_match":obs,"factor_p":fact["permutation_p"],"best_transfer_method":T.groupby("method").rmse.mean().idxmin()});return status

def main():
 raw=Path("work/raw");out=Path("outputs");matdir=Path("work/matrices");raw.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True);matdir.mkdir(parents=True,exist_ok=True)
 for n,u in URLS.items():dl(u,raw/f"{n}.csv")
 mats={};prov=[]
 for n in ("GDSC1","GDSC2"):
  m,x=load_gdsc(raw/f"{n}.csv",n);mats[n]=m;prov.append(x)
 r,t,c,x=load_prism(raw/"PRISM_readout.csv",raw/"PRISM_treatment.csv",raw/"PRISM_cells.csv");mats["PRISM"]=r;prov.append(x)
 for n,m in mats.items():m.to_pickle(matdir/f"{n}.pkl")
 tasks=[(n,str(matdir/f"{n}.pkl"),f,s) for n in mats for f in FRACTIONS for s in SEEDS];A=[];D=[]
 with ProcessPoolExecutor(max_workers=3) as ex:
  fs={ex.submit(one_task,t):t for t in tasks}
  for z in as_completed(fs):a,d=z.result();A+=a;D+=d;print("DONE",fs[z],flush=True)
 B=pd.DataFrame(A);Q=pd.DataFrame(D);B.to_csv(out/"benchmark.csv",index=False);Q.to_csv(out/"drug_metrics.csv",index=False)
 S=B.groupby(["dataset","fraction","method"],as_index=False).agg(rmse=("rmse","mean"),rmse_sd=("rmse","std"),ndcg=("ndcg","mean"),ap=("ap","mean"),ef=("ef","mean"),recall=("recall","mean"),spearman=("spearman","mean"));S["rmse_rank"]=S.groupby(["dataset","fraction"]).rmse.rank();S.to_csv(out/"summary.csv",index=False)
 I=infer(B,Q);I.to_csv(out/"independent_drug_tests.csv",index=False);P=named_prism(r,t,c);X=cross(mats["GDSC2"],P,out)
 status={"status":"completed","datasets":prov,"conditions":len(tasks),"methods":METHODS,"crossstudy":X,"raw_data_in_artifact":False};Path(out/"STATUS.json").write_text(json.dumps(status,indent=2));pd.DataFrame(prov).to_csv(out/"provenance.csv",index=False)
 print(json.dumps(status,indent=2))
if __name__=="__main__":main()
