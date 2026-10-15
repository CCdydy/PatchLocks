"""Standalone Blender renderer for a predicted patch.npz as a COLORED fur swatch.

Self-contained QA tool — does NOT touch the deployment pipeline. Auto-frames the
fur, uses a tuned Hair BSDF (low reflection + darkened albedo so color survives,
no white wash) under moderate lighting, viewed from a slight oblique-top angle.

  blender -b --factory-startup --python patchlocks/qa/render_pred_swatch.py -- \
    --patch <patch.npz> --out <rgb.png> --val 0.6 --spec 0.1

patch.npz must contain: strands (N,L,3), color (N,3), [color_tip], radius, [size].
"""
import argparse
import os
import sys

import numpy as np

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--patch", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=640)
ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--maxN", type=int, default=9000, help="subsample strands to avoid overlap saturation")
ap.add_argument("--val", type=float, default=0.6, help="darken albedo so Hair BSDF shows color (light fur washes white near 1)")
ap.add_argument("--sat", type=float, default=1.3, help="pre-boost saturation to survive render desaturation")
ap.add_argument("--spec", type=float, default=0.10, help="Hair BSDF Reflection; low = no white sheen")
ap.add_argument("--hair_ior", type=float, default=1.12)
ap.add_argument("--rough", type=float, default=0.42)
ap.add_argument("--light", type=float, default=2.0, help="sun energy")
ap.add_argument("--bg", type=float, default=0.52, help="neutral background gray")
ap.add_argument("--el", type=float, default=64.0, help="camera elevation deg above the surface plane (90=straight top-down, matches face-on 2D crop)")
ap.add_argument("--az", type=float, default=-55.0, help="camera azimuth deg")
ap.add_argument("--margin", type=float, default=1.06, help="frame tightly so fur FILLS the frame 1:1 with the 2D crop")
ap.add_argument("--radius_mul", type=float, default=1.0,
                help="display-only radius multiplier; useful for readable top views")
ap.add_argument("--diffuse", action="store_true", help="use diffuse Principled (Base Color = exact color) instead of Hair BSDF")
ap.add_argument("--param", default="color", choices=["color", "melanin"],
                help="Hair BSDF parametrization; melanin matches the paper whole-animal renderer")
ap.add_argument("--melanin", type=float, default=0.24)
ap.add_argument("--melanin_red", type=float, default=0.60)
ap.add_argument("--spread", type=float, default=1.0, help="re-spread roots over the in-plane drift extent so fur is a UNIFORM field filling the frame (1:1 with the 2D crop), not a central tuft; 0 disables")
a = ap.parse_args(argv)

d = np.load(a.patch, allow_pickle=True)
S = d["strands"].astype(np.float32)
N, L, _ = S.shape
COL = d["color"].astype(np.float32)
COLT = d["color_tip"].astype(np.float32) if "color_tip" in d.files else COL
RAD = d["radius"].astype(np.float32).reshape(-1) * float(a.radius_mul)
if len(RAD) == 1:
    RAD = np.full(N, float(RAD[0]), np.float32)
RUV = d["root_uv"].astype(np.float32) if "root_uv" in d.files else None
if N > a.maxN:
    sel = np.random.default_rng(0).choice(N, a.maxN, replace=False)
    S, COL, COLT, RAD, N = S[sel], COL[sel], COLT[sel], RAD[sel], a.maxN
    if RUV is not None:
        RUV = RUV[sel]

# Re-spread roots over the in-plane DRIFT extent so the fur reads as a UNIFORM field
# that fills the frame 1:1 with the 2D crop (infer clusters roots on a tiny footprint
# while strands drift far -> a central tuft with sparse radiating edges otherwise).
if a.spread > 0 and RUV is not None:
    off = S - S[:, :1]
    oh = float(np.abs(off[:, :, :2]).max())
    roots = np.zeros((N, 3), np.float32)
    roots[:, 0] = (RUV[:, 0] - 0.5) * (2 * oh * a.spread)
    roots[:, 1] = (RUV[:, 1] - 0.5) * (2 * oh * a.spread)
    S = roots[:, None, :] + off

# saturation boost + darken so color survives the Hair BSDF
def adjust(c):
    y = (0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2])[:, None]
    c = np.clip(y + (c - y) * a.sat, 0, 1) * a.val
    return c.astype(np.float32)
COL, COLT = adjust(COL), adjust(COLT)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

cv = bpy.data.hair_curves.new("fur")
cv.add_curves([L] * N)
cv.attributes["position"].data.foreach_set("vector", S.reshape(-1).astype(np.float32))
t = np.linspace(0, 1, L, np.float32)[None].repeat(N, 0)
cv.attributes.new("radius", "FLOAT", "POINT").data.foreach_set("value", (RAD[:, None] * (1 - 0.82 * t)).reshape(-1).astype(np.float32))
tg = np.linspace(0, 1, L, np.float32)[None, :, None]
grad = COL[:, None, :] * (1 - tg) + COLT[:, None, :] * tg
cv.attributes.new("strand_col", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.concatenate([grad, np.ones((N, L, 1), np.float32)], -1).reshape(-1).astype(np.float32))
ob = bpy.data.objects.new("fur", cv); sc.collection.objects.link(ob)

m = bpy.data.materials.new("fur"); m.use_nodes = True; nt = m.node_tree; out = nt.nodes["Material Output"]
at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "strand_col"; at.attribute_type = "GEOMETRY"
if a.diffuse:
    pb = nt.nodes["Principled BSDF"]; pb.inputs["Roughness"].default_value = 0.85
    if "Specular IOR Level" in pb.inputs:
        pb.inputs["Specular IOR Level"].default_value = 0.1
    nt.links.new(at.outputs["Color"], pb.inputs["Base Color"])
else:
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    def setin(n, v):
        if n in h.inputs:
            h.inputs[n].default_value = v
    h.parametrization = "MELANIN" if a.param == "melanin" else "COLOR"
    setin("Roughness", a.rough); setin("Radial Roughness", 0.5); setin("Coat", 0.0)
    setin("IOR", a.hair_ior); setin("Reflection", a.spec); setin("Random Color", 0.04)
    if a.param == "melanin":
        setin("Melanin", a.melanin); setin("Melanin Redness", a.melanin_red)
    else:
        nt.links.new(at.outputs["Color"], h.inputs["Color"])
    nt.links.new(h.outputs["BSDF"], out.inputs["Surface"]); nt.nodes.remove(nt.nodes["Principled BSDF"])
ob.data.materials.append(m)

w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (a.bg, a.bg, a.bg, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
bpy.ops.object.light_add(type="SUN", location=(2, -3, 5)); bpy.context.object.data.energy = a.light
bpy.ops.object.light_add(type="AREA", location=(-2, 2, 3)); kl = bpy.context.object.data; kl.energy = 60 * a.light; kl.size = 6

sc.render.engine = "CYCLES"; sc.cycles.device = "GPU"; sc.cycles.samples = a.samples
sc.cycles_curves.shape = "THICK"; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = a.res
sc.view_settings.view_transform = "AgX" if a.param == "melanin" else "Standard"

# Frame the ROOT-covered area (the dense field), NOT the full drifting-strand bbox,
# so the fur FILLS the frame edge-to-edge (drifting tips spill past the frame, like the
# synthetic gate where roots cover the patch and fur fills it).
RT = S[:, 0, :]
ctr = Vector(((RT.min(0) + RT.max(0)) * 0.5).tolist())
half = float(max((RT.max(0) - RT.min(0))[:2]) * 0.5)
dist = half / 0.36 * a.margin
el, az = np.radians(a.el), np.radians(a.az)
dirv = Vector((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
loc = ctr + dist * dirv
fwd = (ctr - loc).normalized()
right = fwd.cross(Vector((0, 0, 1))).normalized()
up = right.cross(fwd).normalized()
cd = bpy.data.cameras.new("c"); cd.lens = 50; cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam)
cam.matrix_world = Matrix.Translation(loc) @ Matrix((right, up, -fwd)).transposed().to_4x4()
sc.camera = cam
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
sc.render.filepath = a.out
bpy.ops.render.render(write_still=True)
print(f"PRED-SWATCH-DONE {a.out}", flush=True)
