"""PHOTOREAL render for the v8 realdog GEOMETRY-target dataset. Same training-compatible outputs as
photoreal_render_loop.py (cameras.json + rgb_k.png + patch.npz copy), HDRI IBL + AgX + DR, but the FRAMING
is tightened for SHORT-DENSE fur: high elevation (look down onto the coat, no HDRI horizon) + zoomed so the
fur fills the frame (matches the real-crop FILL~0.7 of extract.py) + lighter DoF (heavy DoF made dense fine
fur go sub-pixel/smooth). Geometry differs from v6 => latents MUST be re-baked (not reused like v7).
Run: blender -b --factory-startup --python photoreal_render_loop_v8.py -- --patches <v8/patches> --out <dir> --views 4 --res 256 --samples 64 --shard i/n
"""
import bpy, sys, os, glob, json, math
import numpy as np
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--patches", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=4); ap.add_argument("--res", type=int, default=256)
ap.add_argument("--samples", type=int, default=64); ap.add_argument("--limit", type=int, default=100000)
ap.add_argument("--hair_ior", type=float, default=1.35); ap.add_argument("--seed0", type=int, default=7000)
ap.add_argument("--shard", default="0/1")
a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
WORLD = sorted(glob.glob(os.path.join(os.path.dirname(bpy.app.binary_path), "4.5/datafiles/studiolights/world/*.exr")))
assert WORLD, "no studiolight world HDRIs found"
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'; prefs.refresh_devices()
for dv in prefs.devices: dv.use = (dv.type == 'OPTIX')


def build_and_render(patch_path, out_dir, jseed):
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(patch_path, allow_pickle=True)
    S = d["strands"].astype(np.float32); col = d["color"].astype(np.float32)
    ctip = d["color_tip"].astype(np.float32) if "color_tip" in d.files else col
    rad = d["radius"].astype(np.float32).reshape(-1); size = float(d["size"]); N, L, _ = S.shape
    if len(rad) == 1: rad = np.full(N, float(rad[0]), np.float32)
    mt = {}
    if "params" in d.files:
        try:
            pv = d["params"]; pv = pv.item() if hasattr(pv, "item") else pv
            if isinstance(pv, bytes): pv = pv.decode("utf-8")
            mt = json.loads(pv).get("material", {})
        except Exception: mt = {}
    rough = float(mt.get("roughness", 0.5)); rrough = rough * 0.8; coat = float(mt.get("coat", 0.05))
    J = np.random.default_rng(jseed); jit = lambda s: float(J.uniform(-s, s))
    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    for coll in (bpy.data.hair_curves, bpy.data.curves, bpy.data.meshes, bpy.data.materials,
                 bpy.data.lights, bpy.data.cameras, bpy.data.worlds, bpy.data.images, bpy.data.objects, bpy.data.node_groups):
        for b in list(coll):
            if b.users == 0:
                try: coll.remove(b)
                except Exception: pass
    cv = bpy.data.hair_curves.new("fur"); cv.add_curves([L] * N)
    cv.attributes['position'].data.foreach_set('vector', S.reshape(-1))
    t = np.linspace(0, 1, L)[None].repeat(N, 0)
    cv.attributes.new('radius', 'FLOAT', 'POINT').data.foreach_set('value', (rad[:, None] * (1.0 - 0.80 * t)).reshape(-1).astype(np.float32))
    tg = np.linspace(0, 1, L)[None, :, None]; col_grad = col[:, None, :] * (1 - tg) + ctip[:, None, :] * tg
    cv.attributes.new('strand_col', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', np.concatenate([col_grad, np.ones((N, L, 1), np.float32)], -1).reshape(-1).astype(np.float32))
    ob = bpy.data.objects.new("fur", cv); sc.collection.objects.link(ob)
    # skin plane: warm dog skin shows through inter-strand gaps
    bpy.ops.mesh.primitive_plane_add(size=size * 3.0); pm = bpy.data.materials.new("skin"); pm.use_nodes = True
    sl = float(J.uniform(0.16, 0.34)); pm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (sl * 1.12, sl * 0.95, sl * 0.82, 1)
    pm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9; bpy.context.object.data.materials.append(pm)
    # Principled Hair
    m = bpy.data.materials.new("furRGB"); m.use_nodes = True; nt = m.node_tree; out = nt.nodes["Material Output"]
    for n in list(nt.nodes):
        if n.type == 'BSDF_PRINCIPLED': nt.nodes.remove(n)
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "strand_col"; hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    try: hair.parametrization = 'COLOR'
    except Exception: pass
    for inp, val in [("Roughness", rough), ("Radial Roughness", rrough), ("Coat", coat), ("IOR", a.hair_ior)]:
        if inp in hair.inputs: hair.inputs[inp].default_value = val
    cin = "Color" if "Color" in hair.inputs else hair.inputs[0].name
    nt.links.new(at.outputs["Color"], hair.inputs[cin]); nt.links.new(hair.outputs[0], out.inputs["Surface"]); ob.data.materials.append(m)
    # world = random HDRI IBL (real env light), random rotation + strength
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True; wnt = w.node_tree
    env = wnt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(WORLD[J.integers(len(WORLD))])
    mp = wnt.nodes.new("ShaderNodeMapping"); tc = wnt.nodes.new("ShaderNodeTexCoord"); mp.inputs["Rotation"].default_value[2] = float(J.uniform(0, 6.283))
    wnt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); wnt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    bgn = wnt.nodes["Background"]; bgn.inputs["Strength"].default_value = float(J.uniform(0.6, 1.1)); wnt.links.new(env.outputs["Color"], bgn.inputs["Color"])
    # soft fill
    bpy.ops.object.light_add(type='AREA', location=(2 + jit(2), -2 + jit(2), 4)); bpy.context.object.data.energy = float(J.uniform(40, 110)); bpy.context.object.data.size = 3
    sc.render.engine = 'CYCLES'; sc.cycles_curves.shape = 'THICK'; sc.cycles.device = 'GPU'
    sc.cycles.use_denoising = True
    try: sc.cycles.denoiser = 'OPTIX'
    except Exception: pass
    sc.cycles.samples = a.samples; sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.film_transparent = False; sc.render.filter_size = 1.5
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.exposure = float(J.uniform(-0.4, 0.1))
    # ---- TIGHT framing: HIGH elevation (no horizon) + zoom so fur fills frame (FILL~0.7 like real crop) ----
    elevs = [58, 66, 52, 70, 61, 55, 68, 63]; az0 = jit(0.6); cams = []
    for k in range(a.views):
        az = 2 * math.pi * k / a.views + 0.3 + az0; elev = math.radians(elevs[k % len(elevs)] + jit(5))
        r = (0.66 + jit(0.06)) * size          # close -> far patch edge falls outside frame, fur fills
        loc = Vector((r * math.cos(elev) * math.cos(az), r * math.cos(elev) * math.sin(az), r * math.sin(elev)))
        cd = bpy.data.cameras.new(f"c{k}"); cd.lens = 55 * (1 + jit(0.10))
        cd.dof.use_dof = True; cd.dof.focus_distance = float((loc - Vector((0, 0, size * 0.05))).length); cd.dof.aperture_fstop = float(J.uniform(4.0, 9.0))  # light DoF
        cam = bpy.data.objects.new(f"c{k}", cd); sc.collection.objects.link(cam)
        cam.location = loc; cam.rotation_euler = (loc - Vector((0, 0, size * 0.04))).to_track_quat('Z', 'Y').to_euler(); cams.append(cam)
    for k, cam in enumerate(cams):
        sc.camera = cam; sc.render.filepath = os.path.join(out_dir, f"rgb_{k}.png"); bpy.ops.render.render(write_still=True)
    json.dump({"views": [dict(name=c.name, location=list(c.location), matrix_world=[list(r) for r in c.matrix_world], lens=c.data.lens, sensor_width=c.data.sensor_width, res=a.res) for c in cams],
               "size": size, "n_strands": N, "L": L}, open(os.path.join(out_dir, "cameras.json"), "w"))
    np.savez(os.path.join(out_dir, "patch.npz"), **{k: d[k] for k in d.files})


_si, _sn = map(int, a.shard.split("/"))
patches = [p for _k, p in enumerate(sorted(glob.glob(os.path.join(a.patches, "*.npz")))) if _k % _sn == _si][:a.limit]
print(f"V8 PHOTOREAL LOOP {len(patches)} patches -> {a.out} (samples={a.samples})", flush=True)
for i, pp in enumerate(patches):
    name = os.path.splitext(os.path.basename(pp))[0]; od = os.path.join(a.out, name)
    if os.path.exists(os.path.join(od, "cameras.json")) and os.path.exists(os.path.join(od, f"rgb_{a.views-1}.png")) and os.path.exists(os.path.join(od, "patch.npz")): continue
    build_and_render(pp, od, jseed=a.seed0 + i)
    if (i + 1) % 25 == 0: print(f"  {i+1}/{len(patches)}", flush=True)
print("V8-PHOTOREAL-LOOP-DONE", flush=True)
