"""Rigor check: can a CLUMP-AWARE synthesizer, fed even GT scalars (the ceiling), match the decoded
FIELD's clump + curvature distribution? If not, the 7 scalars are provably insufficient and the diffusion
FIELD is genuinely load-bearing (not a 'bad synthesizer' artifact).

Clump-aware synth: arc/curl/lift as before, PLUS per-strand azimuth drawn so the texel-group tip-resultant
= target clump, PLUS arc spread tuned to target tip_spread. This RECOVERS clump/tip_spread from scalars by
construction; the open question is curvature-distribution W1 (a scalar sets only the MEAN curl, not the
per-patch distribution SHAPE).
"""
import os,sys,json,argparse
import numpy as np,torch
sys.path.insert(0,os.path.join(os.getcwd(),"patchlocks","v14"));sys.path.insert(0,os.getcwd())
from v14_common import build_env
from models.strand_codec import StrandCodec
ap=argparse.ArgumentParser()
ap.add_argument("--renders",default="patchlocks/experiments/rebuttal_2027/testB_renders")
ap.add_argument("--e1_dir",default="patchlocks/datasets/v16/e2a_cw15")
ap.add_argument("--codec_path",default="patchlocks/models/fur_strand_codec_v6upright.pt")
ap.add_argument("--ckpt_cond_stats",action="store_true");ap.add_argument("--lk",type=int,default=32)
ap.add_argument("--out",required=True);a=ap.parse_args();dev="cuda"
ATTRS=["arc","curl","clump","straight","lift_arc","tip_spread"]
cond=None
if a.ckpt_cond_stats:
    ck=torch.load(f"{a.e1_dir}/v14_best.pt",map_location="cpu",weights_only=False);cond={k:ck[k] for k in ["cond_mean","cond_std","glob_mean","glob_std"]}
env=build_env(renders=a.renders,e1_dir=a.e1_dir,cond_stats=cond,codec_path=a.codec_path,len_k=a.lk,verbose=True)
H=env.H;fetch=env.fetch
_ck=torch.load(a.codec_path,weights_only=False);_cn={k:torch.tensor(v,device=dev,dtype=torch.float32) for k,v in _ck["fur_norm"].items()}
codec=StrandCodec(do_vae=False,decode_type="dir",scale_init=30.0,nr_verts_per_strand=256,nr_values_to_decode=255,dim_per_value_decoded=3).to(dev).eval()
codec.load_state_dict(_ck["state_dict"])
@torch.no_grad()
def decode(z,ib,ihw):
    zz=(z[:,:64]*env.xs+env.xm).permute(0,2,3,1).reshape(len(z),H*H,64)[ib,ihw];rec=codec.decoder(zz,None,_cn)["strand_positions"];return (rec-rec[:,:1]).cpu().numpy()
def measure(rec,lk=32):
    rec=np.asarray(rec,np.float32);seg=np.linalg.norm(rec[:,1:]-rec[:,:-1],axis=2);arc=seg.sum(1).clip(1e-6)
    tip=rec[:,-1,:2];ch2=np.linalg.norm(tip,axis=1);ang=np.arctan2(tip[:,1],tip[:,0])
    yc=np.sin(-ang)[:,None]*rec[...,0]+np.cos(-ang)[:,None]*rec[...,1];curl=np.abs(yc).mean(1)/arc
    nv=len(rec)//lk;R=lambda v:v.reshape(nv,lk);cm=R(np.cos(ang)).mean(1);sm=R(np.sin(ang)).mean(1);med=lambda v:np.median(R(v),1)
    return dict(arc=med(arc),curl=med(curl),clump=np.sqrt(cm**2+sm**2),straight=med(np.linalg.norm(rec[:,-1],axis=1)/arc),
                lift_arc=med(np.abs(rec[...,2]).max(1)/arc),tip_spread=R(ch2).std(1)/med(arc).clip(1e-6))
def curv(rec):
    rec=np.asarray(rec,np.float32);seg=rec[:,1:]-rec[:,:-1];sn=np.linalg.norm(seg,axis=2,keepdims=True).clip(1e-9);u=seg/sn
    return np.arccos(np.clip((u[:,1:]*u[:,:-1]).sum(-1),-1,1)).mean(1)
def clumpaware(lk,P,arc,cu,li,clump,tsp,rng):
    # base strokes
    t=np.linspace(0,1,P,dtype=np.float32)[None,:,None];arc=max(arc,1e-6);cu=max(cu,0.);li=max(li,0.)
    # tip_spread -> arc coefficient of variation
    cvv=float(np.clip(tsp,0.02,0.6))
    ai=arc*np.exp(rng.normal(0,cvv,lk)).astype(np.float32)
    ci=cu*np.exp(rng.normal(0,0.18,lk)).astype(np.float32);ll=li*np.exp(rng.normal(0,0.16,lk)).astype(np.float32)
    ya=(ci*ai/(2/np.pi+1e-6)).astype(np.float32);za=(ll*ai).astype(np.float32)
    sg=rng.choice(np.array([-1,1],np.float32),lk);ph=rng.uniform(-.18,.18,lk).astype(np.float32);x=(ai*.985).astype(np.float32);off=np.zeros((lk,P,3),np.float32)
    for _ in range(3):
        tt=t[:,:,0];off[:,:,0]=x[:,None]*tt;off[:,:,1]=(sg*ya)[:,None]*np.sin(np.pi*np.clip(tt+ph[:,None]*tt*(1-tt),0,1));off[:,:,2]=za[:,None]*(1-np.cos(np.pi*tt))*.5
        ca=np.linalg.norm(np.diff(off,axis=1),axis=2).sum(1);x*=np.clip(ai/np.maximum(ca,1e-9),.85,1.08).astype(np.float32)
    # CLUMP-AWARE: rotate each strand in-plane by azimuth drawn to hit target resultant length = clump
    R=float(np.clip(clump,0.001,0.999))
    # von Mises kappa s.t. A1(kappa)=R  (approx inverse)
    if R<0.53: kap=2*R+R**3+5*R**5/6
    elif R<0.85: kap=-0.4+1.39*R+0.43/(1-R)
    else: kap=1/(R**3-4*R**2+3*R)
    az=rng.vonmises(0.0,max(kap,1e-3),lk).astype(np.float32)
    ca_,sa=np.cos(az),np.sin(az)
    xr=ca_[:,None]*off[:,:,0]-sa[:,None]*off[:,:,1];yr=sa[:,None]*off[:,:,0]+ca_[:,None]*off[:,:,1]
    off[:,:,0]=xr;off[:,:,1]=yr
    return off
idx=list(range(len(env.dirs)))
res={r:{k:[] for k in ATTRS} for r in ["field","ca_gt"]};w1={r:[] for r in ["field","ca_gt"]};GT={k:[] for k in ATTRS}
with torch.no_grad():
    for i,g in enumerate(idx):
        j=torch.tensor([g],device=env.idx_dev);mm=fetch(env.MASK,j);ib,ihw,kept=env.sample_texels(mm,a.lk,seed=1000000+g)
        if ib is None or not bool(kept[0]):continue
        rec_gt=decode(fetch(env.Xn,j).float(),ib,ihw)
        gc,gl=fetch(env.G_COND,j),fetch(env.GLOB,j);rc,rp,ry=fetch(env.RHOC,j).float(),fetch(env.RGBPROJ,j).float(),fetch(env.RAYC,j).float()
        rec_f=decode(env.sample_z(gc,gl,rc,rp,ry,cfg=3.0,seed=g).float(),ib,ihw)
        pg=measure(rec_gt);pf=measure(rec_f)
        rng=np.random.RandomState(60000+g)
        # clump-aware from GT scalars (the ceiling)
        ca=clumpaware(a.lk,256,float(pg['arc'][0]),float(pg['curl'][0]),float(pg['lift_arc'][0]),float(pg['clump'][0]),float(pg['tip_spread'][0]),rng)
        pca=measure(ca)
        for k in ATTRS:
            GT[k].append(float(pg[k][0]));res["field"][k].append(float(pf[k][0]));res["ca_gt"][k].append(float(pca[k][0]))
        cg=curv(rec_gt);w1["field"].append(float(np.abs(np.sort(curv(rec_f))-np.sort(cg)).mean()));w1["ca_gt"].append(float(np.abs(np.sort(curv(ca))-np.sort(cg)).mean()))
        if (i+1)%40==0:print(f"  {i+1}/{len(idx)}",flush=True)
def nm(p,gg):p,gg=np.asarray(p),np.asarray(gg);return float(np.abs(p-gg).mean()/(gg.std()+1e-12))
print(f"\n{'arm':28} {'mNMAE':>6} {'clumpVar':>8} {'W1curv':>7}")
out={}
for r in ["field","ca_gt"]:
    mn=np.mean([nm(res[r][k],GT[k]) for k in ATTRS]);cv=np.std(res[r]['clump'])/(np.std(GT['clump'])+1e-12);ww=np.mean(w1[r])*1e3
    out[r]=dict(mnmae=mn,clumpvar=float(cv),w1=float(ww));print(f"{('Diffusion FIELD' if r=='field' else 'CLUMP-AWARE synth (GT scalars=ceiling)'):28} {mn:6.3f} {cv:8.3f} {ww:7.2f}")
json.dump(out,open(a.out,"w"),indent=1);print("saved",a.out)
