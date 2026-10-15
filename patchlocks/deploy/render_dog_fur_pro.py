"""Knife 1 — production-quality STATIC render of the real-dog fur: Principled Hair BSDF (realistic
anisotropic sheen) + HDRI environment + AgX + subsurface skin. Renders our guide strands directly
(already ~1.4M, dense). No surface binding yet (that's knife 2 / drive).
Run: blender -b --factory-startup --python render_dog_fur_pro.py -- --fur <npz> --mesh <obj> --out <dir>
"""
import bpy, sys, os, math, glob
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--fur", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=4); ap.add_argument("--only_view", type=int, default=-1, help="render only this turntable view index; default renders all views")
ap.add_argument("--res", type=int, default=1024); ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--device", choices=["GPU", "CPU"], default="GPU", help="Cycles compute device; CPU is useful for low-resolution previews while the GPU is occupied")
ap.add_argument("--radius", type=float, default=0.0); ap.add_argument("--camdist", type=float, default=1.6)
ap.add_argument("--radius_mul", type=float, default=1.0, help="scale per-strand radius (visual fur volume/fullness); applies to the loaded RAD when --radius is not forced")
ap.add_argument("--length_mul", type=float, default=1.0, help="render-only strand length multiplier about each root; preserves roots and directions")
ap.add_argument("--param", default="color", choices=["color", "melanin"]); ap.add_argument("--melanin", type=float, default=0.55); ap.add_argument("--melanin_red", type=float, default=0.7)
ap.add_argument("--roughness", type=float, default=0.32); ap.add_argument("--radial_rough", type=float, default=0.35); ap.add_argument("--coat", type=float, default=0.08)
ap.add_argument("--rand_color", type=float, default=0.05); ap.add_argument("--rand_rough", type=float, default=0.12)
ap.add_argument("--view", default="agx", choices=["agx", "standard", "filmic"], help="view transform; AgX desaturates highlights -> use 'standard' to keep warm color")
ap.add_argument("--spec", type=float, default=1.0, help="scales Hair BSDF Reflection/Transmission/Secondary (white sheen that washes color); lower = keep base color")
ap.add_argument("--hair_ior", type=float, default=1.55, help="Hair BSDF IOR; lower (1.1-1.3) = weaker grazing-angle Fresnel reflection -> less white rim-wash on light fur")
ap.add_argument("--sat", type=float, default=1.0, help="pre-boost per-strand color saturation (counteracts render desaturation)")
ap.add_argument("--val", type=float, default=1.0, help="darken albedo (Hair BSDF only shows base color when DARK enough; light hair renders white)")
ap.add_argument("--flat", default="", help="debug: force a constant hair Color 'r,g,b' (bypass strand_col attribute)")
ap.add_argument("--diffuse", action="store_true", help="use a diffuse Principled BSDF (Base Color = strand color) instead of Hair BSDF -> color shows reliably")
ap.add_argument("--key_warm", type=float, default=0.0, help="0=white key, 1=warm key; lifts R lowers B of the soft key light")
ap.add_argument("--hdri", default=""); ap.add_argument("--hdri_strength", type=float, default=1.0); ap.add_argument("--exposure", type=float, default=0.0); ap.add_argument("--skin", default="0.22,0.16,0.12")
ap.add_argument("--look", default="None", help="color-management look, e.g. 'AgX - Medium High Contrast'")
ap.add_argument("--no_denoise", action="store_true", help="disable Cycles denoising so sub-pixel fibers are not smeared")
ap.add_argument("--mesh_vert_rgb", default="", help="optional (V,3) npy vertex color for the bare mesh; fixes gray mouth/nose/skin gaps")
ap.add_argument("--fur_mesh_albedo", default="", help="optional (V,3) mesh albedo; barycentrically color fur roots via root_face/root_bary")
ap.add_argument("--fur_root_rgb", default="", help="optional (N,3) display color array that overrides the NPZ strand root colors")
ap.add_argument("--mesh_val", type=float, default=1.0, help="brightness multiplier for --mesh_vert_rgb")
ap.add_argument("--mesh_sat", type=float, default=1.0, help="saturation multiplier for --mesh_vert_rgb")
ap.add_argument("--simple_light", action="store_true", help="drop the cool studio HDRI -> warm white sun+area + neutral bg (matches render_colmap_views warm path)")
ap.add_argument("--paper_light", action="store_true", help="white-paper studio rig: warm key, restrained fill and neutral rim for readable fibers")
ap.add_argument("--camera_relative_light", action="store_true", help="place the paper-light rig in camera coordinates and aim each area light at the groom center")
ap.add_argument("--camera_relative_soft", action="store_true", help="use a broader, less grazing camera-relative key with stronger fill and a softer rim")
ap.add_argument("--light_scale", type=float, default=1.0, help="scale simple_light energies (lower = dimmer, less grazing white rim on light fur)")
ap.add_argument("--world_color", default="0.62,0.64,0.68", help="simple_light world AMBIENT color 'r,g,b'. Default is slightly BLUE (B>R) -> tints DARK fur purple (dark base reflects the blue ambient). Use neutral/warm like '0.66,0.64,0.60' to keep dark markings neutral-black.")
ap.add_argument("--up", default="-y", help="vertical = turntable spin axis: x/y/z/-x/-y/-z (standing dog up = MIDDLE bbox extent; default -y is legacy)")
ap.add_argument("--curve_chunk", type=int, default=1500000,
                help="max strands per Blender hair_curves object. Large coats must be split to avoid foreach_set internal array-length overflow.")
ap.add_argument("--save_blend", default="", help="optional .blend path to save the constructed scene after rendering")
ap.add_argument("--transparent", action="store_true", help="render with transparent film (RGBA alpha) so the subject can be cleanly composited onto any background")
ap.add_argument("--az_deg", type=float, default=-999.0, help="explicit camera azimuth in degrees (overrides the turntable az for the rendered view; -999=off)")
ap.add_argument("--el_deg", type=float, default=-999.0, help="explicit camera elevation in degrees (overrides the turntable el; -999=off)")
ap.add_argument("--out_name", default="", help="output filename (without dir) when using az_deg/el_deg; default furview_<k>.png")
ap.add_argument("--ground_shadow", action="store_true", help="add a shadow-catcher ground plane at feet level; with --transparent the soft contact shadow lands in the alpha channel while fur shading is untouched")
a = ap.parse_args(argv); os.makedirs(a.out, exist_ok=True)
BL_DATA = os.path.join(os.path.dirname(bpy.app.binary_path), bpy.app.version_string.split()[0][:3], "datafiles")
hdri = a.hdri or os.path.join(BL_DATA, "studiolights", "world", "studio.exr")

d = np.load(a.fur); S = d["strands"].astype(np.float32); N, L, _ = S.shape
COL = d["color"].astype(np.float32) if "color" in d.files else np.tile([0.5, 0.4, 0.3], (N, 1)).astype(np.float32)
if a.fur_root_rgb:
    COL = np.load(a.fur_root_rgb).astype(np.float32)
    if COL.shape != (N, 3):
        raise ValueError(f"fur_root_rgb must have shape {(N, 3)}, got {COL.shape}")
    print(f"fur root display color: {a.fur_root_rgb}", flush=True)
RAD = d["radius"].astype(np.float32) if "radius" in d.files else None
ctr = S[:, 0].mean(0); size = float(np.linalg.norm(S.reshape(-1, 3).max(0)-S.reshape(-1, 3).min(0)))
if a.length_mul <= 0:
    raise ValueError("--length_mul must be positive")
if a.length_mul != 1.0:
    S = S[:, :1] + (S - S[:, :1]) * float(a.length_mul)
    print(f"render-only strand length multiplier: {a.length_mul:.4f}", flush=True)
rad = a.radius if a.radius > 0 else (float(np.median(RAD)) if RAD is not None else size*0.0008)
print(f"PRO fur {N}x{L} | size {size:.3f} radius {rad:.6f} | param {a.param} | hdri {os.path.basename(hdri)}", flush=True)

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'; prefs.refresh_devices()
for dv in prefs.devices: dv.use = (dv.type == 'OPTIX')
bpy.ops.wm.read_factory_settings(use_empty=True); sc = bpy.context.scene

# skin (subsurface so coat gaps read as skin, not black)
verts = []; faces = []
for ln in open(a.mesh):
    if ln.startswith('v '): verts.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith('f '): faces.append([int(p.split('/')[0])-1 for p in ln.split()[1:4]])
me = bpy.data.meshes.new("dog"); me.from_pydata(verts, [], faces); me.update()
if a.fur_mesh_albedo:
    alb = np.load(a.fur_mesh_albedo).astype(np.float32)
    if len(alb) != len(verts):
        raise ValueError(f"fur_mesh_albedo has {len(alb)} colors for {len(verts)} mesh vertices")
    if "root_face" not in d.files or "root_bary" not in d.files:
        raise ValueError("--fur_mesh_albedo requires root_face/root_bary in the fur NPZ")
    rf = d["root_face"].astype(np.int64)
    rb = d["root_bary"].astype(np.float32)
    tri = np.asarray(faces, np.int64)[rf]
    COL = (alb[tri] * rb[:, :, None]).sum(1).astype(np.float32)
    print(f"fur color inherited from mesh albedo: {a.fur_mesh_albedo}", flush=True)
if a.val != 1.0: COL = np.clip(COL*a.val, 0, 1).astype(np.float32)                 # darken so Hair BSDF absorption shows the color
if a.sat != 1.0:                                                                   # pre-boost saturation to survive render desaturation
    Yc = (0.299*COL[:, 0]+0.587*COL[:, 1]+0.114*COL[:, 2])[:, None]; COL = np.clip(Yc+(COL-Yc)*a.sat, 0, 1).astype(np.float32)
mesh_col = None
if a.mesh_vert_rgb:
    mesh_col = np.load(a.mesh_vert_rgb).astype(np.float32)
    if len(mesh_col) != len(verts):
        print(f"WARNING: mesh_vert_rgb has {len(mesh_col)} colors for {len(verts)} verts; ignoring", flush=True)
        mesh_col = None
    else:
        mesh_col = np.clip(mesh_col*a.mesh_val, 0, 1)
        if a.mesh_sat != 1.0:
            y = (0.299*mesh_col[:, 0]+0.587*mesh_col[:, 1]+0.114*mesh_col[:, 2])[:, None]
            mesh_col = np.clip(y+(mesh_col-y)*a.mesh_sat, 0, 1).astype(np.float32)
        ca = me.color_attributes.new(name="mesh_col", type="FLOAT_COLOR", domain="CORNER")
        loop_col = []
        for poly in me.polygons:
            for vi in poly.vertices:
                loop_col.extend([float(mesh_col[vi, 0]), float(mesh_col[vi, 1]), float(mesh_col[vi, 2]), 1.0])
        ca.data.foreach_set("color", np.asarray(loop_col, np.float32))
        print(f"mesh vertex color on: {a.mesh_vert_rgb} | val {a.mesh_val} sat {a.mesh_sat}", flush=True)
dog = bpy.data.objects.new("dog", me); sc.collection.objects.link(dog)
sk = bpy.data.materials.new("skin"); sk.use_nodes = True; b = sk.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = tuple([float(x) for x in a.skin.split(",")])+(1,)
if mesh_col is not None:
    at = sk.node_tree.nodes.new("ShaderNodeAttribute"); at.attribute_name = "mesh_col"; at.attribute_type = 'GEOMETRY'
    sk.node_tree.links.new(at.outputs["Color"], b.inputs["Base Color"])
for nm, val in [("Subsurface Weight", 0.12), ("Roughness", 0.6)]:
    if nm in b.inputs: b.inputs[nm].default_value = val
dog.data.materials.append(sk)

# coat curves.  Blender's foreach_set uses internal signed lengths in a few
# paths, so a 6M-strand dog can overflow if all curves live in one object.
# Split into identical-material curve chunks; geometry and render are unchanged.
fur_obs = []
chunk = max(int(a.curve_chunk), 1)
t_base = np.linspace(0, 1, L, dtype=np.float32)[None, :]
for s0 in range(0, N, chunk):
    s1 = min(s0 + chunk, N)
    Sc = S[s0:s1]
    Cc = COL[s0:s1]
    Rc = RAD[s0:s1] if RAD is not None else None
    Nc = len(Sc)
    cv = bpy.data.hair_curves.new(f"fur_{s0//chunk:03d}"); cv.add_curves([L] * Nc)
    cv.attributes['position'].data.foreach_set('vector', Sc.reshape(-1))
    rr = (Rc[:, None] if (Rc is not None and a.radius <= 0) else np.full((Nc, 1), rad, np.float32))
    if a.radius_mul != 1.0: rr = (rr * a.radius_mul).astype(np.float32)
    t = np.repeat(t_base, Nc, 0)
    cv.attributes.new('radius', 'FLOAT', 'POINT').data.foreach_set('value', (rr * (1 - 0.72 * t)).reshape(-1).astype(np.float32))
    cpt = np.repeat(np.concatenate([Cc, np.ones((Nc, 1), np.float32)], 1)[:, None], L, 1).reshape(-1).astype(np.float32)
    cv.attributes.new('strand_col', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', cpt)
    ob = bpy.data.objects.new(f"fur_{s0//chunk:03d}", cv); sc.collection.objects.link(ob)
    fur_obs.append(ob)
print(f"fur curve chunks: {len(fur_obs)} x <= {chunk}", flush=True)

# fur material
m = bpy.data.materials.new("furM"); m.use_nodes = True; nt = m.node_tree; out = nt.nodes["Material Output"]
def colsrc():
    if a.flat:
        n = nt.nodes.new("ShaderNodeRGB"); n.outputs[0].default_value = tuple(float(x) for x in a.flat.split(","))+(1,); print("FLAT debug color", a.flat, flush=True); return n.outputs[0]
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "strand_col"; at.attribute_type = 'GEOMETRY'; return at.outputs["Color"]
if a.diffuse:                                                                      # diffuse fur: Base Color shows the real color reliably (Hair BSDF washes light colors)
    pb = nt.nodes["Principled BSDF"]; pb.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in pb.inputs: pb.inputs["Specular IOR Level"].default_value = 0.15*a.spec
    nt.links.new(colsrc(), pb.inputs["Base Color"])
else:
    hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    def setin(n, v):
        if n in hair.inputs: hair.inputs[n].default_value = v
    setin("Roughness", a.roughness); setin("Radial Roughness", a.radial_rough); setin("Coat", a.coat); setin("IOR", a.hair_ior)
    setin("Random Color", a.rand_color); setin("Random Roughness", a.rand_rough); setin("Reflection", a.spec)
    if a.param == "melanin":
        hair.parametrization = 'MELANIN'; setin("Melanin", a.melanin); setin("Melanin Redness", a.melanin_red)
    else:
        hair.parametrization = 'COLOR'; nt.links.new(colsrc(), hair.inputs["Color"])
    nt.links.new(hair.outputs["BSDF"], out.inputs["Surface"]); nt.nodes.remove(nt.nodes["Principled BSDF"])
for ob in fur_obs:
    ob.data.materials.append(m)

if a.ground_shadow:
    _ax = {"x": 0, "y": 1, "z": 2}[a.up.lstrip("-")]
    _neg = a.up.startswith("-")
    _lo = float(S[:, :, _ax].max() if _neg else S[:, :, _ax].min())
    _g = size * 6.0
    _c = [float(ctr[0]), float(ctr[1]), float(ctr[2])]; _c[_ax] = _lo
    _u = [i for i in range(3) if i != _ax]
    _gv = []
    for du, dv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        v = list(_c); v[_u[0]] += du * _g; v[_u[1]] += dv * _g; _gv.append(tuple(v))
    _gme = bpy.data.meshes.new("ground"); _gme.from_pydata(_gv, [], [[0, 1, 2, 3]]); _gme.update()
    _gob = bpy.data.objects.new("ground", _gme); sc.collection.objects.link(_gob)
    _gob.is_shadow_catcher = True
    print("ground shadow catcher at", a.up, "=", _lo, flush=True)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True; wnt = w.node_tree
if a.paper_light:
    # Print-oriented three-point rig. Keep ambient low so the roots remain
    # darker than the tips, then use a broad key and a rear rim to reveal the
    # anisotropic strand structure against a white compositing background.
    bg = wnt.nodes["Background"]
    bg.inputs["Color"].default_value = (0.48, 0.47, 0.45, 1)
    bg.inputs["Strength"].default_value = 0.40
    if a.camera_relative_light:
        # Derive the same camera frame used below, then express the complete
        # three-point rig in screen coordinates.  This preserves the selected
        # camera while keeping key/fill/rim placement consistent across animals.
        _up = Vector({"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0),
                      "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}[a.up])
        _C = Vector(ctr.tolist())
        _e1 = _up.cross(Vector((1, 0, 0)))
        _e1 = (_up.cross(Vector((0, 0, 1))) if _e1.length < 1e-3 else _e1).normalized()
        _e2 = _up.cross(_e1).normalized()
        _k = a.only_view if 0 <= a.only_view < a.views else 0
        _az = 2 * math.pi * _k / a.views + 0.4
        _el = math.radians(8 + 12 * (_k % 2))
        if a.az_deg > -900.0: _az = math.radians(a.az_deg)
        if a.el_deg > -900.0: _el = math.radians(a.el_deg)
        _horiz = math.cos(_az) * _e1 + math.sin(_az) * _e2
        _cam = _C + a.camdist * size * (math.cos(_el) * _horiz + math.sin(_el) * _up)
        _fwd = (_C - _cam).normalized()
        _right = _fwd.cross(_up).normalized()
        _screen_up = _right.cross(_fwd).normalized()
        _toward_camera = -_fwd

        def _aimed_area(name, offset, energy, area_size, color):
            loc = _C + size * (offset[0] * _right + offset[1] * _screen_up + offset[2] * _toward_camera)
            bpy.ops.object.light_add(type='AREA', location=loc)
            ob = bpy.context.object; ob.name = name
            ob.rotation_euler = (_C - ob.location).to_track_quat('-Z', 'Y').to_euler()
            ob.data.energy = energy * a.light_scale
            ob.data.size = area_size
            ob.data.color = color
            print(f"camera-relative light {name}: loc={tuple(round(v, 4) for v in ob.location)} "
                  f"energy={ob.data.energy:.1f} size={ob.data.size:.2f}", flush=True)

        # Offsets are (screen-right, screen-up, toward-camera), in groom-size units.
        # The key is deliberately lateral: a frontal aimed key washes the Hair
        # BSDF and hides strand relief.  Fill and ambient are restrained, while
        # the rear rim separates the silhouette without flattening the torso.
        if a.camera_relative_soft:
            # Softer paper preset: broaden and move the key slightly toward the
            # camera, lift the shadow side, and keep the rim mostly on the
            # silhouette. Geometry, material, exposure and camera are unchanged.
            bg.inputs["Strength"].default_value = 0.34
            _aimed_area("paper_key",  (-1.55, 1.10,  1.45), 300, 4.8, (1.0, 0.97, 0.92))
            _aimed_area("paper_fill", ( 1.25, 0.20,  1.45), 110, 8.0, (0.93, 0.96, 1.0))
            _aimed_area("paper_rim",  ( 0.90, 1.00, -1.90), 190, 3.6, (1.0, 0.99, 0.97))
        else:
            bg.inputs["Strength"].default_value = 0.32
            _aimed_area("paper_key",  (-1.80, 1.10,  1.15), 330, 3.8, (1.0, 0.97, 0.92))
            _aimed_area("paper_fill", ( 1.40, 0.15,  1.25),  80, 7.0, (0.93, 0.96, 1.0))
            _aimed_area("paper_rim",  ( 0.90, 1.00, -1.90), 240, 2.8, (1.0, 0.99, 0.97))
    else:
        # Legacy world-space rig retained for exact Figure 3 reproducibility.
        bpy.ops.object.light_add(type='AREA', location=(-3.2, 2.4, 4.8))
        kl = bpy.context.object.data; kl.energy = 480*a.light_scale; kl.size = 4.5
        kl.color = (1.0, 0.97, 0.92)
        bpy.ops.object.light_add(type='AREA', location=(0.0, -4.2, 1.6))
        fl = bpy.context.object.data; fl.energy = 135*a.light_scale; fl.size = 7.0
        fl.color = (0.93, 0.96, 1.0)
        bpy.ops.object.light_add(type='AREA', location=(3.4, 1.8, 4.2))
        rl = bpy.context.object.data; rl.energy = 280*a.light_scale; rl.size = 3.0
        rl.color = (1.0, 0.99, 0.97)
elif a.simple_light:
    # warm white path (matches render_colmap_views): neutral solid bg, white sun + soft area; no cool HDRI tint
    wnt.nodes["Background"].inputs["Color"].default_value = tuple(float(x) for x in a.world_color.split(",")) + (1,)
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 6)); bpy.context.object.data.energy = 4.0*a.light_scale
    bpy.ops.object.light_add(type='AREA', location=(-3, 2, 4)); kl = bpy.context.object.data; kl.energy = 400*a.light_scale; kl.size = 5
    kl.color = (1.0, 1.0-0.18*a.key_warm, 1.0-0.40*a.key_warm)
    bpy.ops.object.light_add(type='AREA', location=(0, -4, 1)); fl = bpy.context.object.data; fl.energy = 180*a.light_scale; fl.size = 8   # frontal fill -> even out the grazing rim that washes light fur
else:
    # HDRI lights the dog, but the CAMERA sees a clean neutral background (not the dark studio.exr)
    bg = wnt.nodes["Background"]; env = wnt.nodes.new("ShaderNodeTexEnvironment")
    try: env.image = bpy.data.images.load(hdri); wnt.links.new(env.outputs["Color"], bg.inputs["Color"])
    except Exception as e: print("HDRI load fail ->", e); bg.inputs["Color"].default_value = (0.6, 0.62, 0.66, 1)
    bg.inputs["Strength"].default_value = a.hdri_strength
    bg2 = wnt.nodes.new("ShaderNodeBackground"); bg2.inputs["Color"].default_value = (0.58, 0.60, 0.63, 1); bg2.inputs["Strength"].default_value = 1.0
    lp = wnt.nodes.new("ShaderNodeLightPath"); mx = wnt.nodes.new("ShaderNodeMixShader")
    wnt.links.new(lp.outputs["Is Camera Ray"], mx.inputs["Fac"]); wnt.links.new(bg.outputs["Background"], mx.inputs[1]); wnt.links.new(bg2.outputs["Background"], mx.inputs[2])
    wnt.links.new(mx.outputs["Shader"], wnt.nodes["World Output"].inputs["Surface"])
    bpy.ops.object.light_add(type='AREA', location=(-2.5, 2.5, 4)); kl = bpy.context.object.data; kl.energy = 250; kl.size = 6  # soft key
    kl.color = (1.0, 1.0-0.18*a.key_warm, 1.0-0.40*a.key_warm)  # warm the key to restore lost warmth

sc.render.engine = 'CYCLES'; sc.cycles_curves.shape = 'THICK'; sc.cycles.device = a.device; sc.cycles.samples = a.samples
sc.cycles.use_denoising = not a.no_denoise; sc.render.resolution_x = sc.render.resolution_y = a.res; sc.render.image_settings.file_format = 'PNG'
if a.transparent: sc.render.film_transparent = True; sc.render.image_settings.color_mode = 'RGBA'
try: sc.view_settings.view_transform = {"agx": "AgX", "standard": "Standard", "filmic": "Filmic"}[a.view]
except Exception: sc.view_settings.view_transform = 'Standard'
if a.look != "None":
    try: sc.view_settings.look = a.look
    except Exception as e: print(f"WARNING: unsupported color look {a.look!r}: {e}", flush=True)
sc.view_settings.exposure = a.exposure
_UP = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}[a.up]
C = Vector(ctr.tolist()); up = Vector(_UP)
e1 = up.cross(Vector((1, 0, 0))); e1 = (up.cross(Vector((0, 0, 1))) if e1.length < 1e-3 else e1).normalized(); e2 = up.cross(e1).normalized()   # orbit basis perpendicular to up
view_ids = [a.only_view] if 0 <= a.only_view < a.views else range(a.views)
for k in view_ids:
    az = 2*math.pi*k/a.views+0.4; el = math.radians(8+12*(k % 2)); r = a.camdist*size
    if a.az_deg > -900.0: az = math.radians(a.az_deg)
    if a.el_deg > -900.0: el = math.radians(a.el_deg)
    horiz = math.cos(az)*e1+math.sin(az)*e2; loc = C+r*(math.cos(el)*horiz+math.sin(el)*up)
    fwd = (C-loc).normalized(); right = fwd.cross(up).normalized(); tup = right.cross(fwd).normalized()
    cd = bpy.data.cameras.new(f"c{k}"); cd.lens = 55; cam = bpy.data.objects.new(f"c{k}", cd); sc.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(loc) @ Matrix((right, tup, -fwd)).transposed().to_4x4()
    fname = a.out_name if a.out_name else f"furview_{k}.png"
    sc.camera = cam; sc.render.filepath = os.path.join(a.out, fname); bpy.ops.render.render(write_still=True)
if a.save_blend:
    os.makedirs(os.path.dirname(os.path.abspath(a.save_blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=a.save_blend)
    print(f"SAVED-BLEND {a.save_blend}", flush=True)
print("RENDER-PRO-DONE", flush=True)
