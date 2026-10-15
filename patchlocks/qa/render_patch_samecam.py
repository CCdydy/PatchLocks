"""Render a patch.npz through an existing cameras.json.

This is for v10 visual gating: render GT and predicted patches with the same
camera and deterministic lighting so color/groom differences are visible.

Run:
  blender -b --factory-startup --python patchlocks/qa/render_patch_samecam.py -- \
    --patch <patch.npz> --cameras <cameras.json> --out <dir> --samples 48
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--patch", required=True)
ap.add_argument("--cameras", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", default="", help="comma-separated view indices; empty renders all")
ap.add_argument("--res", type=int, default=0, help="0 uses cameras.json res")
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--hair_ior", type=float, default=1.35)
ap.add_argument("--light_scale", type=float, default=1.0)
ap.add_argument("--view_transform", default="Standard", help="Blender view transform; AgX rolls off highlights (fixes blown-out bright fur)")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

d = np.load(a.patch, allow_pickle=True)
S = d["strands"].astype(np.float32)
root_uv = d["root_uv"].astype(np.float32)
color = d["color"].astype(np.float32)
color_tip = d["color_tip"].astype(np.float32) if "color_tip" in d.files else color
radius = d["radius"].astype(np.float32).reshape(-1)
size = float(d["size"])
N, L, _ = S.shape
if len(radius) == 1:
    radius = np.full(N, float(radius[0]), np.float32)
meta = json.load(open(a.cameras))
view_ids = list(range(len(meta["views"]))) if not a.views else [int(x) for x in a.views.split(",") if x.strip()]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

cv = bpy.data.hair_curves.new("fur")
cv.add_curves([L] * N)
cv.attributes["position"].data.foreach_set("vector", S.reshape(-1).astype(np.float32))
t = np.linspace(0, 1, L, dtype=np.float32)[None].repeat(N, 0)
cv.attributes.new("radius", "FLOAT", "POINT").data.foreach_set("value", (radius[:, None] * (1.0 - 0.82 * t)).reshape(-1).astype(np.float32))
tg = np.linspace(0, 1, L, dtype=np.float32)[None, :, None]
col_grad = color[:, None, :] * (1 - tg) + color_tip[:, None, :] * tg
cv.attributes.new("strand_col", "FLOAT_COLOR", "POINT").data.foreach_set("color", np.concatenate([col_grad, np.ones((N, L, 1), np.float32)], -1).reshape(-1).astype(np.float32))
obj = bpy.data.objects.new("fur", cv)
sc.collection.objects.link(obj)

bpy.ops.mesh.primitive_plane_add(size=size * 2.4, location=(0, 0, 0))
plane = bpy.context.object
pmat = bpy.data.materials.new("skin")
pmat.use_nodes = True
pbsdf = pmat.node_tree.nodes["Principled BSDF"]
pbsdf.inputs["Base Color"].default_value = (0.18, 0.145, 0.115, 1)
pbsdf.inputs["Roughness"].default_value = 0.9
plane.data.materials.append(pmat)

mat = bpy.data.materials.new("furRGB")
mat.use_nodes = True
nt = mat.node_tree
out = nt.nodes["Material Output"]
for node in list(nt.nodes):
    if node.type == "BSDF_PRINCIPLED":
        nt.nodes.remove(node)
attr = nt.nodes.new("ShaderNodeAttribute")
attr.attribute_name = "strand_col"
hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
try:
    hair.parametrization = "COLOR"
except Exception:
    pass
for name, val in [("Roughness", 0.58), ("Radial Roughness", 0.45), ("Coat", 0.04), ("IOR", a.hair_ior)]:
    if name in hair.inputs:
        hair.inputs[name].default_value = val
cin = "Color" if "Color" in hair.inputs else hair.inputs[0].name
nt.links.new(attr.outputs["Color"], hair.inputs[cin])
nt.links.new(hair.outputs[0], out.inputs["Surface"])
obj.data.materials.append(mat)

world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.55, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45 * a.light_scale
bpy.ops.object.light_add(type="AREA", location=(1.2 * size, -1.4 * size, 1.8 * size))
key = bpy.context.object
key.data.energy = 280 * a.light_scale
key.data.size = 2.0 * size
bpy.ops.object.light_add(type="AREA", location=(-1.8 * size, 1.0 * size, 1.2 * size))
fill = bpy.context.object
fill.data.energy = 70 * a.light_scale
fill.data.size = 2.8 * size

sc.render.engine = "CYCLES"
sc.cycles_curves.shape = "THICK"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.refresh_devices()
for dv in prefs.devices:
    dv.use = dv.type == "OPTIX"
sc.cycles.device = "GPU"
sc.cycles.samples = a.samples
sc.cycles.use_denoising = True
try:
    sc.cycles.denoiser = "OPTIX"
except Exception:
    pass
sc.view_settings.view_transform = a.view_transform
sc.view_settings.exposure = 0
sc.view_settings.gamma = 1
sc.render.film_transparent = False
sc.render.filter_size = 1.2
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"

for vid in view_ids:
    vc = meta["views"][vid]
    res = int(a.res or vc.get("res", 256))
    sc.render.resolution_x = res
    sc.render.resolution_y = res
    cd = bpy.data.cameras.new(f"cam{vid}")
    cd.lens = float(vc["lens"])
    cd.sensor_width = float(vc.get("sensor_width", 36.0))
    cam = bpy.data.objects.new(f"cam{vid}", cd)
    sc.collection.objects.link(cam)
    cam.matrix_world = Matrix(vc["matrix_world"])
    sc.camera = cam
    sc.render.filepath = os.path.join(a.out, f"rgb_{vid}.png")
    bpy.ops.render.render(write_still=True)

json.dump(meta, open(os.path.join(a.out, "cameras.json"), "w"), indent=2)
print(f"rendered {len(view_ids)} view(s): {a.patch} -> {a.out}", flush=True)
