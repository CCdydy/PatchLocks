"""Identifiability of the splatting fit — CORRECT minimal model with LOCAL loss windows.
A patch's fur = oriented gaussian strokes from a root, half-length L forward along dir phi. The fit loss is
evaluated ONLY in a local window around the root (as in the real per-patch graph). Two regimes:
  BOUNDARY: local target = the patch's own fur fringe on an otherwise-empty local window -> L & phi VISIBLE.
  INTERIOR: local window is pre-SATURATED (dense neighbor coverage, mask=1 everywhere) -> L & phi INVISIBLE.
Loss = local coverage BCE + soft-IoU (+ optional UNDIRECTED orientation cos2). Separated tests:
  A) DIRECTION: fix L=GT, init phi FLIPPED 180, fit phi. Run WITH orientation and SILHOUETTE-ONLY.
  B) LENGTH:    fix phi=GT, init L wrong, fit L.  -> r_flip, e_signed, length err%, per regime.
"""
import numpy as np, torch, torch.nn.functional as F
dev="cuda" if torch.cuda.is_available() else "cpu"; W=64; N_STR=12; SIG=2.0; LEN_PX=20.0; LMIN=2.0
PX=torch.stack(torch.meshgrid(torch.arange(W,device=dev).float(),torch.arange(W,device=dev).float(),indexing="xy"),-1)
root=torch.tensor([W*0.5,W*0.85],device=dev)   # root near bottom of the LOCAL window; fur points UP into it
np.random.seed(0); torch.manual_seed(0)

def fur(phi,llen):
    L=F.softplus(llen)*LEN_PX+LMIN
    fl=torch.stack([torch.cos(phi),torch.sin(phi)]); pe=torch.stack([-torch.sin(phi),torch.cos(phi)])
    off=torch.linspace(-5,5,N_STR,device=dev); anc=root[None]+off[:,None]*pe[None]
    rel=PX[None]-anc[:,None,None]; al=(rel*fl).sum(-1); pp=(rel*pe).sum(-1)
    return (torch.exp(-torch.relu(al)**2/(2*L**2))*torch.exp(-pp**2/(2*SIG**2))).sum(0).clamp(0,1),L

def target_for(regime,phi_gt,llen_gt):
    with torch.no_grad(): cov,_=fur(phi_gt,llen_gt)
    if regime=="interior":
        return torch.ones(W,W,device=dev)          # saturated: neighbors fill the whole local window
    return (cov>0.3).float()                        # boundary: only the patch's own fringe is 'fur'

def fit_loss(pred,tgt,phi,phi_star,use_orient):
    pc=pred.clamp(1e-4,1-1e-4); bce=F.binary_cross_entropy(pc,tgt)
    inter=(pred*tgt).sum(); iou=1-(inter+1)/(pred.sum()+tgt.sum()-inter+1)
    l=1.0*bce+1.5*iou
    if use_orient: l=l+0.5*(1-torch.cos(2*(phi-phi_star)))
    return l

def testA(regime,use_orient,n=40):
    fl=[];es=[]
    for _ in range(n):
        phi_gt=torch.tensor(float(np.random.uniform(0,2*np.pi)),device=dev)
        llen_gt=torch.tensor(float(np.random.uniform(0.2,1.2)),device=dev); tgt=target_for(regime,phi_gt,llen_gt); ps=phi_gt%np.pi
        phi=torch.nn.Parameter(phi_gt+np.pi+torch.randn(1,device=dev)[0]*0.1); opt=torch.optim.Adam([phi],lr=0.08)
        for _ in range(400):
            opt.zero_grad(); pred,_=fur(phi,llen_gt.detach()); fit_loss(pred,tgt,phi,ps,use_orient).backward(); opt.step()
        with torch.no_grad():
            dot=float(torch.cos(phi)*torch.cos(phi_gt)+torch.sin(phi)*torch.sin(phi_gt))
            fl.append(1.0 if dot<0 else 0.0); es.append(float(np.degrees(np.arccos(np.clip(dot,-1,1)))))
    return float(np.mean(fl)),float(np.median(es))

def testB(regime,n=40):
    err=[]
    for _ in range(n):
        phi_gt=torch.tensor(float(np.random.uniform(0,2*np.pi)),device=dev)
        llen_gt=torch.tensor(float(np.random.uniform(0.2,1.2)),device=dev)
        with torch.no_grad():
            cov_gt,Lg=fur(phi_gt,llen_gt)
        tgt = torch.ones(W,W,device=dev) if regime=="interior" else cov_gt   # SOFT target
        llen=torch.nn.Parameter(torch.tensor(float(llen_gt)-1.0,device=dev)); opt=torch.optim.Adam([llen],lr=0.1)
        for _ in range(500):
            opt.zero_grad(); pred,_=fur(phi_gt.detach(),llen); F.mse_loss(pred,tgt).backward(); opt.step()
        with torch.no_grad(): Lf=fur(phi_gt,llen)[1].item(); err.append(abs(Lf-Lg.item())/Lg.item())
    return float(np.median(err))

print("== A) DIRECTION SIGN (fix L=GT, init phi FLIPPED 180) ==")
out={}
for reg in ["boundary","interior"]:
    rf_o,es_o=testA(reg,True); rf_s,es_s=testA(reg,False)
    out[f"dir_{reg}"]={"with_orient":{"r_flip":rf_o,"e":es_o},"silhouette_only":{"r_flip":rf_s,"e":es_s}}
    print(f"  [{reg:8}] with-orient r_flip={rf_o:.2f} e={es_o:5.1f} | silhouette-only r_flip={rf_s:.2f} e={es_s:5.1f}")
print("== B) LENGTH (fix phi=GT, init L wrong) ==")
for reg in ["boundary","interior"]:
    le=testB(reg); out[f"len_{reg}"]={"lenerr_med":le}; print(f"  [{reg:8}] length err% med={le*100:5.1f}")
import json; json.dump(out,open("patchlocks/experiments/rebuttal_2027/identifiability.json","w"),indent=1)
print("\nEXPECT: boundary length err LOW, interior HIGH. Direction: silhouette-only may recover at boundary;")
print("if r_flip high at boundary even silhouette-only -> in-plane sign truly unobservable from mask.")
