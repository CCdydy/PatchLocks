"""Single-process Blender loop renderer for v5 REALISTIC pet fur (RGB only; analytic rootUV is canonical).
Loads Blender ONCE and renders every patch_*.npz under <patches_dir> to <out_dir>/<name>/ with K views.
Realism = Principled Hair BSDF + per-strand radius taper (undercoat thin / guard thick) + root->tip color
gradient + per-patch material (finish -> roughness/radial-roughness/coat from params.material) + light domain
randomization (white balance, light level, world background, exposure/gamma) keyed per patch + OPTIX denoiser
(so low --samples stays clean). Mirrors render_patch.py's look but in one fast process.
Run:  blender -b --factory-startup --python render_loop.py -- --patches <dir> --out <dir> --views 4 --res 256 --samples 24 --dr
"""
import bpy, sys, os, glob, json, math
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--patches", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=4); ap.add_argument("--res", type=int, default=256)
ap.add_argument("--samples", type=int, default=48); ap.add_argument("--limit", type=int, default=100000)
ap.add_argument("--dr", action="store_true", help="light domain randomization (WB/level/background/exposure)")
ap.add_argument("--denoise", action="store_true", help="OPTIX denoiser (OFF by default: hallucinates rainbow chroma on dark thin hair)")
ap.add_argument("--light_scale", type=float, default=0.5, help="dim lights to kill colored specular sheen on dark fur")
ap.add_argument("--hair_ior", type=float, default=1.3, help="Principled Hair IOR (lower=less plastic specular sheen)")
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--shard", default="0/1", help="i/n: render only patches where index%n==i (parallel sharding across processes)")
a = ap.parse_args(argv)
os.makedirs(a.out, exist_ok=True)

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'; prefs.refresh_devices()
for dv in prefs.devices: dv.use = (dv.type == 'OPTIX')


def build_and_render(patch_path, out_dir, jseed):
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(patch_path, allow_pickle=True)
    S = d["strands"].astype(np.float32); ruv = d["root_uv"].astype(np.float32)
    col = d["color"].astype(np.float32)
    ctip = d["color_tip"].astype(np.float32) if "color_tip" in d.files else col
    rad = d["radius"].astype(np.float32).reshape(-1); size = float(d["size"])
    N, L, _ = S.shape
    if len(rad) == 1: rad = np.full(N, float(rad[0]), np.float32)
    mt = {}                                                                          # robust to numpy unicode/bytes/0d-array params
    if "params" in d.files:
        try:
            pv = d["params"]; pv = pv.item() if hasattr(pv, "item") else pv
            if isinstance(pv, bytes): pv = pv.decode("utf-8")
            mt = json.loads(pv).get("material", {})
        except Exception:
            mt = {}
    rough = float(mt.get("roughness", 0.45)); rrough = rough * 0.8; coat = float(mt.get("coat", 0.1))
    J = np.random.default_rng(jseed); jit = lambda s: float(J.uniform(-s, s))

    bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene
    for coll in (bpy.data.hair_curves, bpy.data.curves, bpy.data.meshes, bpy.data.materials,   # purge orphan data so 5000-iter loop doesn't leak/slow/crash
                 bpy.data.lights, bpy.data.cameras, bpy.data.worlds, bpy.data.images, bpy.data.objects, bpy.data.node_groups):
        for b in list(coll):
            if b.users == 0:
                try: coll.remove(b)
                except Exception: pass
    cv = bpy.data.hair_curves.new("fur"); cv.add_curves([L] * N)
    cv.attributes['position'].data.foreach_set('vector', S.reshape(-1))
    t = np.linspace(0, 1, L)[None].repeat(N, 0)                                     # per-point taper root->tip
    radii = (rad[:, None] * (1.0 - 0.85 * t)).reshape(-1).astype(np.float32)
    cv.attributes.new('radius', 'FLOAT', 'POINT').data.foreach_set('value', radii)
    tg = np.linspace(0, 1, L)[None, :, None]                                        # root->tip color gradient
    col_grad = col[:, None, :] * (1 - tg) + ctip[:, None, :] * tg
    cpt = np.concatenate([col_grad, np.ones((N, L, 1), np.float32)], -1).reshape(-1).astype(np.float32)
    cv.attributes.new('strand_col', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cpt)
    ob = bpy.data.objects.new("fur", cv); sc.collection.objects.link(ob)
    bpy.ops.mesh.primitive_plane_add(size=size); pm = bpy.data.materials.new("skin"); pm.use_nodes = True
    skin_l = float(J.uniform(0.20, 0.45)) if a.dr else 0.13                            # neutral-warm dog-skin tone so inter-strand gap pixels read as skin, not void (kept near-neutral to not over-warm fur, real R/B~0.96)
    pm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (skin_l * 1.05, skin_l * 0.98, skin_l * 0.90, 1)
    bpy.context.object.data.materials.append(pm)
    if a.dr:                                                                            # booth platform/backdrop -> empty pixels read as a bright studio (DINOv2 context match), NOT black void
        bpy.ops.mesh.primitive_plane_add(size=size * 8, location=(0, 0, -0.02 * size)); bm = bpy.data.materials.new("booth"); bm.use_nodes = True
        bl = float(J.uniform(0.48, 0.68))                                               # bright + near-neutral (sat<0.04) so it does NOT bleed colored chroma through dark-fur gaps (rainbow guard)
        bm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (bl, bl * 0.99, bl * 0.97, 1)
        bm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.92
        bpy.context.object.data.materials.append(bm)

    # Principled Hair BSDF driven by per-strand color + per-patch material
    m = bpy.data.materials.new("furRGB"); m.use_nodes = True; nt = m.node_tree; out = nt.nodes["Material Output"]
    for n in list(nt.nodes):
        if n.type == 'BSDF_PRINCIPLED': nt.nodes.remove(n)
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "strand_col"
    hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    try: hair.parametrization = 'COLOR'
    except Exception: pass
    hv = [("Roughness", rough), ("Radial Roughness", rrough), ("Coat", coat)]
    if a.hair_ior > 0: hv.append(("IOR", a.hair_ior))
    for inp, val in hv:
        if inp in hair.inputs: hair.inputs[inp].default_value = val
    cin = "Color" if "Color" in hair.inputs else hair.inputs[0].name
    nt.links.new(at.outputs["Color"], hair.inputs[cin]); nt.links.new(hair.outputs[0], out.inputs["Surface"])
    ob.data.materials.append(m)

    # lights + world (with light DR)
    def wb():
        if not a.dr: return (1.0, 1.0, 1.0)
        tt = J.uniform(-1, 1); return (1.0 + 0.10 * tt - 0.02, 1.0, 1.0 - 0.10 * tt + 0.02)   # mild cool bias (booth is fluorescent-cool) but small so fur keeps saturation (~40)
    glv = (float(np.exp(J.uniform(-0.45, 0.45))) if a.dr else 1.0) * a.light_scale * (0.95 if a.dr else 1.0)  # booth light tuned to REAL fur-region luminance (~111)
    bpy.ops.object.light_add(type='SUN', location=(2 + jit(2), -2 + jit(2), 5)); _s = bpy.context.object
    _s.data.energy = 3 * 0.5 * (1 + jit(0.4)) * glv; _s.data.color = wb()
    bpy.ops.object.light_add(type='AREA', location=(-2 + jit(2), 1, 3)); bpy.context.object.data.energy = 200 * 0.5 * (1 + jit(0.4)) * glv
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    if a.dr:                                                                        # BRIGHT NEUTRAL bg to match real booth luminance (145-180); near-neutral so it does NOT bleed colored chroma through dark-fur gaps (rainbow guard)
        bg = float(J.uniform(0.50, 0.78))
        w.node_tree.nodes["Background"].inputs["Color"].default_value = (bg * 1.02, bg, bg * 0.97, 1)
    else:
        w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)

    sc.render.engine = 'CYCLES'; sc.cycles_curves.shape = 'THICK'; sc.cycles.device = 'GPU'
    if a.denoise:
        sc.cycles.use_denoising = True
        try: sc.cycles.denoiser = 'OPTIX'
        except Exception: pass
    sc.cycles.samples = a.samples; sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.film_transparent = (not a.dr); sc.render.filter_size = 1.5; sc.view_settings.view_transform = 'Standard'  # --dr: opaque bright booth bg (no black void for DINOv2)
    if a.dr:
        sc.view_settings.exposure = float(J.uniform(-0.6, 0.2)); sc.view_settings.gamma = float(J.uniform(0.9, 1.1))   # centered (real fur is mid-toned ~111, not bright)
    fur_reach = float(np.percentile(np.linalg.norm(S - S[:, :1], axis=-1), 99))      # actual strand extent -> frame so LONG coats are not cut off; density reads correctly
    frame = max(size, 2.2 * fur_reach); zoomf = 1.9 if a.dr else 1.55                # dr zooms out a touch (room for booth context); frame grows for long fur
    elevs = [42, 55, 62, 48, 70, 50, 58, 45]; az0 = jit(1.0); cams = []
    for k in range(a.views):
        az = 2 * math.pi * k / a.views + 0.3 + az0; elev = math.radians(elevs[k % len(elevs)] + jit(8))
        r = (zoomf + jit(0.2)) * frame + 0.45
        loc = Vector((r * math.cos(elev) * math.cos(az), r * math.cos(elev) * math.sin(az), r * math.sin(elev)))
        cd = bpy.data.cameras.new(f"c{k}"); cd.lens = 75 * (1 + jit(0.15))
        cam = bpy.data.objects.new(f"c{k}", cd); sc.collection.objects.link(cam)
        cam.location = loc; cam.rotation_euler = (loc - Vector((0, 0, size * 0.15))).to_track_quat('Z', 'Y').to_euler()
        cams.append(cam)
    for k, cam in enumerate(cams):
        sc.camera = cam; sc.render.filepath = os.path.join(out_dir, f"rgb_{k}.png"); bpy.ops.render.render(write_still=True)
    json.dump({"views": [dict(name=c.name, location=list(c.location),
                              matrix_world=[list(r) for r in c.matrix_world], lens=c.data.lens,
                              sensor_width=c.data.sensor_width, res=a.res) for c in cams],
               "size": size, "n_strands": N, "L": L}, open(os.path.join(out_dir, "cameras.json"), "w"))
    np.savez(os.path.join(out_dir, "patch.npz"), **{k: d[k] for k in d.files})


_si, _sn = map(int, a.shard.split("/"))
patches = [p for _k, p in enumerate(sorted(glob.glob(os.path.join(a.patches, "*.npz")))) if _k % _sn == _si][:a.limit]
print(f"LOOP rendering {len(patches)} patches -> {a.out} (dr={a.dr} samples={a.samples})", flush=True)
for i, pp in enumerate(patches):
    name = os.path.splitext(os.path.basename(pp))[0]
    od = os.path.join(a.out, name)
    if os.path.exists(os.path.join(od, "cameras.json")) and os.path.exists(os.path.join(od, f"rgb_{a.views-1}.png")) and os.path.exists(os.path.join(od, "patch.npz")):
        continue
    build_and_render(pp, od, jseed=a.seed0 + i)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(patches)}", flush=True)
print("LOOP-DONE", flush=True)
