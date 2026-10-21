"""Render a groom at a real capture camera, for side-by-side comparison with the photo.

Generalizes render_surfpack_realcam.py (paper QA tool) to arbitrary rigs: the camera rotation is
taken from the RQ decomposition of the world->clip matrix (no world-up assumption) and the
principal point becomes a lens shift, so the render aligns with patch_graph_opt_geo.project_row
for any camera roll. Strands use the groom's photo-projected colors (hair BSDF, AgX), rendered
at the photo resolution with a transparent film. With --mesh the furless mesh is rendered under
the fur (per-vertex sRGB colours from --mesh_colors), so sparse fur does not show the backdrop.

blender -b --factory-startup --python patchlocks/qa/render_groom_realcam.py -- \
  --surfpack <groom.npz> --matrices <30000_matrices.pkl> --view 0012 --W 1748 --H 2320 \
  --out <dir> --tag fur --subsample 600000
"""
import argparse, os, sys, pickle
import numpy as np
import bpy
from mathutils import Matrix, Vector


def read_obj(path):
    vs, fs = [], []
    for ln in open(path):
        if ln.startswith("v "):
            vs.append(ln.split()[1:4])
        elif ln.startswith("f "):
            fs.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.asarray(vs, np.float32), np.asarray(fs, np.int64)


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def add_skin(mesh_path, colors_path, rough_path="", spec_path=""):
    """Furless mesh under the fur: per-vertex colour (sRGB .npy) or a neutral tone, and optionally a
    per-vertex roughness and specular level (.npy), e.g. glossy eyes and a dark nose on an otherwise
    matte skin (a bright, uniform world reflected at the default level greys a dark nose)."""
    v, f = read_obj(mesh_path)
    me = bpy.data.meshes.new("skin")
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(f), bool))
    col = srgb_to_linear(np.load(colors_path)) if colors_path else np.full((len(v), 3), 0.3, np.float32)
    assert len(col) == len(v), "mesh colours do not match the mesh"
    attr = me.color_attributes.new("skin_col", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.concatenate([col, np.ones((len(v), 1), np.float32)], 1).reshape(-1))
    obj = bpy.data.objects.new("skin", me)
    bpy.context.scene.collection.objects.link(obj)
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "skin_col"
    nt.links.new(at.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
    if rough_path:
        rough = np.load(rough_path).astype(np.float32).reshape(-1)
        assert len(rough) == len(v), "mesh roughness does not match the mesh"
        ra = me.attributes.new("skin_rough", "FLOAT", "POINT")
        ra.data.foreach_set("value", rough)
        rn = nt.nodes.new("ShaderNodeAttribute")
        rn.attribute_name = "skin_rough"
        nt.links.new(rn.outputs["Fac"], bsdf.inputs["Roughness"])
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.5
    if spec_path and "Specular IOR Level" in bsdf.inputs:
        spec = np.load(spec_path).astype(np.float32).reshape(-1)
        assert len(spec) == len(v), "mesh specular does not match the mesh"
        sa = me.attributes.new("skin_spec", "FLOAT", "POINT")
        sa.data.foreach_set("value", spec)
        sn = nt.nodes.new("ShaderNodeAttribute")
        sn.attribute_name = "skin_spec"
        nt.links.new(sn.outputs["Fac"], bsdf.inputs["Specular IOR Level"])
    obj.data.materials.append(mat)
    return len(v)


def rq3(Ain):
    P_ = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
    Ap = P_ @ Ain
    Q, R = np.linalg.qr(Ap.T)
    R = P_ @ R.T @ P_
    Q = P_ @ Q.T
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
            Q[i, :] *= -1
    return R, Q  # K, Rrot


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfpack", required=True)
    ap.add_argument("--res_scale", type=float, default=1.0,
                    help="render at W*s x H*s (same field of view) for close-up crops")
    ap.add_argument("--srgb_colors", action="store_true",
                    help="groom colors are sRGB (photo-projected); convert to linear for Blender")
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--mesh", default="", help="furless mesh (OBJ, groom frame) rendered under the fur")
    ap.add_argument("--mesh_colors", default="", help="per-vertex sRGB colours (.npy) for --mesh")
    ap.add_argument("--mesh_roughness", default="", help="per-vertex roughness (.npy) for --mesh (glossy eyes, nose)")
    ap.add_argument("--mesh_specular", default="", help="per-vertex specular level (.npy) for --mesh (0.5 = IOR 1.5)")
    ap.add_argument("--world_rgb", default="0.55,0.55,0.55")
    ap.add_argument("--world_strength", type=float, default=0.45, help="multiplied by --light_scale")
    ap.add_argument("--key_scale", type=float, default=1.0, help="multiplies the key and fill area lights only")
    ap.add_argument("--light_rig", default="world", choices=["world", "camera"],
                    help="world: key/fill offset along world axes, facing world -Z (as the paper QA renders); "
                         "camera: key above-right and fill below-left of each camera, both aimed at the groom")
    ap.add_argument("--fur_shader", default="hair", choices=["hair", "fiber"],
                    help="hair: Principled Hair BSDF (its white specular lobe pales a saturated coat); "
                         "fiber: diffuse fibre with a little translucency and a tinted sheen, keeps the photo colour")
    ap.add_argument("--view", required=True)
    ap.add_argument("--affine", default="1,1,0,0", help="sx,sy,tx,ty at full res")
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--photo", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="fur")
    ap.add_argument("--subsample", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--light_scale", type=float, default=0.6)
    ap.add_argument("--hair_ior", type=float, default=1.35)
    ap.add_argument("--view_transform", default="AgX")
    ap.add_argument("--exposure", type=float, default=0.0, help="view exposure in stops")
    ap.add_argument("--device", default="optix", choices=["optix", "cuda"],
                    help="cuda avoids OptiX's curve intersector (it can fault on some grooms)")
    ap.add_argument("--filter_width", type=float, default=0.0, help="pixel filter width in pixels; 0 = Blender's 1.5")
    ap.add_argument("--bounces", type=int, default=0, help="light bounces (diffuse/transmission); 0 = Blender's defaults")
    ap.add_argument("--radius_mul", type=float, default=1.0)
    ap.add_argument("--length_mul", type=float, default=1.0,
                    help="render-only strand length multiplier about each root")
    a = ap.parse_args(argv)
    if a.length_mul <= 0:
        ap.error("--length_mul must be > 0")
    os.makedirs(a.out, exist_ok=True)
    W, H = a.W, a.H

    # --- camera matrix -> (K, center, forward) ---
    # Verified (probe): building the cam basis from forward + world-up and sensor_fit
    # VERTICAL with pixel_aspect reproduces project_row to 0.00 px (no mirror, no shift).
    mats = pickle.load(open(a.matrices, "rb"))
    M = np.asarray(mats[a.view], np.float64)        # clip = [x,y,z,1] @ M
    P = M.T                                         # clip = P @ [x,y,z,1]
    Pxyw = np.array([P[0], P[1], P[3]])             # pinhole (x,y,w) 3x4
    Mp = Pxyw[:, :3]
    p4 = Pxyw[:, 3]
    C = -np.linalg.solve(Mp, p4)                    # camera center, world
    K, _R = rq3(Mp)
    K = K / K[2, 2]
    fx_ndc, fy_ndc = K[0, 0], K[1, 1]
    forward = P[3, :3] / np.linalg.norm(P[3, :3])   # viewing dir into scene

    # --- subsample strands (preserve length: same indices if seed shared) ---
    d = np.load(a.surfpack, allow_pickle=True)
    S = d["strands"].astype(np.float32)
    if abs(a.length_mul - 1.0) > 1e-8:
        S = S[:, :1] + (S - S[:, :1]) * a.length_mul
    radius = d["radius"].astype(np.float32).reshape(-1)
    color = d["color"].astype(np.float32) if "color" in d.files else np.full((len(S), 3), 0.6, np.float32)
    color_tip = d["color_tip"].astype(np.float32) if "color_tip" in d.files else color
    if a.srgb_colors:
        color, color_tip = srgb_to_linear(color), srgb_to_linear(color_tip)
    N0 = len(S)
    if a.subsample > 0 and N0 > a.subsample:
        idx = np.sort(np.random.default_rng(a.seed).choice(N0, a.subsample, replace=False))
        S = S[idx]; radius = radius[idx]; color = color[idx]; color_tip = color_tip[idx]
    N, Lp, _ = S.shape
    arc = float(np.median(np.linalg.norm(np.diff(S, axis=1), axis=2).sum(1)))
    size = float(np.linalg.norm(S.reshape(-1, 3).max(0) - S.reshape(-1, 3).min(0)))
    print(f"[{a.tag}] strands {N0}->{N} arc_median={arc:.5f} world_extent={size:.4f}", flush=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    # --- fur curves ---
    cv = bpy.data.hair_curves.new("fur")
    cv.add_curves([Lp] * N)
    cv.attributes["position"].data.foreach_set("vector", S.reshape(-1).astype(np.float32))
    t = np.linspace(0, 1, Lp, dtype=np.float32)[None].repeat(N, 0)
    rad = radius[:, None] * (1.0 - 0.82 * t) * a.radius_mul
    cv.attributes.new("radius", "FLOAT", "POINT").data.foreach_set("value", rad.reshape(-1).astype(np.float32))
    tg = np.linspace(0, 1, Lp, dtype=np.float32)[None, :, None]
    col_grad = color[:, None, :] * (1 - tg) + color_tip[:, None, :] * tg
    cv.attributes.new("strand_col", "FLOAT_COLOR", "POINT").data.foreach_set(
        "color", np.concatenate([col_grad, np.ones((N, Lp, 1), np.float32)], -1).reshape(-1).astype(np.float32))
    obj = bpy.data.objects.new("fur", cv)
    sc.collection.objects.link(obj)

    # --- hair-BSDF material (reused from render_patch_samecam) ---
    mat = bpy.data.materials.new("furRGB")
    mat.use_nodes = True
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    for node in list(nt.nodes):
        if node.type == "BSDF_PRINCIPLED":
            nt.nodes.remove(node)
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "strand_col"
    if a.fur_shader == "fiber":
        # diffuse fibre + 20 % translucency, sheen tinted by the strand colour (no white specular)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.6
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.15
        if "Sheen Weight" in bsdf.inputs:
            bsdf.inputs["Sheen Weight"].default_value = 0.4
            bsdf.inputs["Sheen Roughness"].default_value = 0.5
            nt.links.new(attr.outputs["Color"], bsdf.inputs["Sheen Tint"])
        tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
        nt.links.new(attr.outputs["Color"], tr.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = 0.2
        nt.links.new(bsdf.outputs[0], mix.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:
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
    if a.mesh:
        print(f"[{a.tag}] skin mesh {add_skin(a.mesh, a.mesh_colors, a.mesh_roughness, a.mesh_specular)} vertices", flush=True)

    # --- lighting (reused) ---
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = tuple(float(x) for x in a.world_rgb.split(",")) + (1.0,)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = a.world_strength * a.light_scale
    bpy.ops.object.light_add(type="AREA", location=(C[0] + 1.2 * size, C[1] - 1.4 * size, C[2] + 1.8 * size))
    key = bpy.context.object
    key.data.energy = 280 * a.light_scale * a.key_scale
    key.data.size = 2.0 * size
    bpy.ops.object.light_add(type="AREA", location=(C[0] - 1.8 * size, C[1] + 1.0 * size, C[2] + 1.2 * size))
    fill = bpy.context.object
    fill.data.energy = 70 * a.light_scale * a.key_scale
    fill.data.size = 2.8 * size

    # --- camera (VERIFIED 0.00px vs project_row): basis from forward + world-up ---
    # rotation from the RQ decomposition: rows of _R are camera right, image-up (NDC y is up)
    # and forward; no assumption about the world's up axis, so any camera roll is handled
    obj_ctr = S.reshape(-1, 3).mean(0)
    right, up, fwd = _R[0] / np.linalg.norm(_R[0]), _R[1] / np.linalg.norm(_R[1]), _R[2] / np.linalg.norm(_R[2])
    if np.dot(obj_ctr - C, fwd) < 0:
        fwd = -fwd
    if np.dot(np.cross(right, up), -fwd) < 0:   # Blender camera: X right, Y up, Z backward
        up = -up
    Rcw = np.stack([right, up, -fwd], axis=1)        # cam->world (cols: X,Y,Z of cam)
    mw = np.eye(4)
    mw[:3, :3] = Rcw
    mw[:3, 3] = C

    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 36.0
    cd.lens = fy_ndc * 18.0
    # principal point -> lens shift (Blender shifts the view window in units of the fit-axis size)
    pp = np.r_[C + fwd, 1.0] @ M
    cx = (pp[0] / pp[3] * 0.5 + 0.5) * W
    cy = (1.0 - (pp[1] / pp[3] * 0.5 + 0.5)) * H
    ycor = (fx_ndc / fy_ndc) * (float(W) / float(H))
    cd.shift_x = (W / 2.0 - cx) / (ycor * H)
    cd.shift_y = (cy - H / 2.0) / (ycor * H)
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    cam.matrix_world = Matrix(mw.tolist())
    sc.camera = cam
    if a.light_rig == "camera":
        # same lighting relative to every camera: key up-right, fill low-left, aimed at the groom
        for lt, (kr, ku) in [(key, (1.0, 1.2)), (fill, (-1.5, -0.5))]:
            pos = C + size * (kr * right + ku * up)
            lt.location = Vector(pos.tolist())
            lt.rotation_euler = Vector((obj_ctr - pos).tolist()).to_track_quat("-Z", "Y").to_euler()

    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.resolution_percentage = int(round(100 * a.res_scale))
    sc.render.pixel_aspect_x = 1.0
    sc.render.pixel_aspect_y = (fx_ndc / fy_ndc) * (float(W) / float(H))

    # --- engine ---
    sc.render.engine = "CYCLES"
    sc.cycles_curves.shape = "THICK"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    dev_type = a.device.upper()
    prefs.compute_device_type = dev_type
    prefs.refresh_devices()
    for dv in prefs.devices:
        dv.use = dv.type == dev_type
    sc.cycles.device = "GPU"
    sc.cycles.samples = a.samples
    if a.filter_width > 0:
        sc.cycles.filter_width = a.filter_width          # pixels; photos carry lens blur
    if a.bounces > 0:
        # dense light fur needs many scattering events between strands; Blender's default of 4
        # diffuse bounces leaves the inside of a white coat grey
        sc.cycles.max_bounces = a.bounces
        sc.cycles.diffuse_bounces = a.bounces
        sc.cycles.glossy_bounces = min(a.bounces, 8)
        sc.cycles.transmission_bounces = a.bounces
        sc.cycles.transparent_max_bounces = max(a.bounces, 8)
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPTIX"
    except Exception:
        pass
    sc.view_settings.view_transform = a.view_transform
    sc.view_settings.exposure = a.exposure
    sc.view_settings.gamma = 1
    sc.render.film_transparent = True
    sc.render.filter_size = 1.2
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = os.path.join(a.out, f"render_{a.tag}_{a.view}_raw.png")
    bpy.ops.render.render(write_still=True)
    # stash camera params for the python-side verify/warp/compose step
    np.savez(os.path.join(a.out, f"caminfo_{a.tag}_{a.view}.npz"),
             K=K, C=C, mw=mw, fx_ndc=fx_ndc, fy_ndc=fy_ndc, forward=fwd, arc=arc)
    print(f"[{a.tag}] rendered raw -> {sc.render.filepath} | lens={cd.lens:.3f} pax_y={sc.render.pixel_aspect_y:.4f}", flush=True)


if __name__ == "__main__":
    main()
