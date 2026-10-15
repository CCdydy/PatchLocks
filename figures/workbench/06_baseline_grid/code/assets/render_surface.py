#!/usr/bin/env python
"""Clay-shaded bare-surface renderer for the baseline grid (SMAL/NeuS/GenZoo columns).

Replicates render_dog_fur_pro's turntable camera EXACTLY (same az/el/lens/up,
studio.exr world) but takes --center/--size explicitly so a baseline mesh renders
in the identical frame as our fur result for that animal. No fur -> matte clay.

blender -b --factory-startup --python render_surface.py -- \
  --mesh <obj> --out <dir> --center cx,cy,cz --size S --camdist 2.05 --only_view 1 --views 4
"""
import bpy, sys, os, math
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

mesh_path = opt("--mesh"); out = opt("--out"); os.makedirs(out, exist_ok=True)
autoframe = "--autoframe" in argv   # derive center/size from the mesh's own bbox
center = [float(x) for x in opt("--center", "0,0,0").split(",")]
size = float(opt("--size", "1.0"))
camdist = float(opt("--camdist", "2.05"))
views = int(opt("--views", "4"))
only_view = int(opt("--only_view", "1"))
res = int(opt("--res", "1280"))
samples = int(opt("--samples", "48"))
up_key = opt("--up", "y")
clay = [float(x) for x in opt("--clay", "0.62,0.62,0.60").split(",")]
out_name = opt("--out_name", "")

BL_DATA = os.path.join(os.path.dirname(bpy.app.binary_path), "4.5", "datafiles")
hdri = os.path.join(BL_DATA, "studiolights", "world", "studio.exr")

# scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
try: sc.cycles.device = "GPU"
except Exception: pass
sc.cycles.samples = samples
sc.render.resolution_x = res; sc.render.resolution_y = res
sc.render.film_transparent = True
sc.view_settings.view_transform = "AgX"

# mesh + clay -- parse OBJ verts/faces directly and build via from_pydata (NO
# axis conversion, matches render_dog_fur_pro; Blender's obj_import rotates 90 deg).
verts, faces = [], []
for ln in open(mesh_path):
    if ln.startswith("v "):
        verts.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith("f "):
        faces.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:]])
if autoframe and verts:                              # frame the mesh by its own bbox
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    center = [(min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2]
    size = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2) ** 0.5
me = bpy.data.meshes.new("surf"); me.from_pydata(verts, [], faces); me.update()
obj = bpy.data.objects.new("surf", me); sc.collection.objects.link(obj)
mat = bpy.data.materials.new("clay"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (*clay, 1.0)
bsdf.inputs["Roughness"].default_value = 0.7
obj.data.materials.append(mat)

# world = studio HDRI (same as fur renders)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
nt = w.node_tree; nt.nodes.clear()
env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdri)
bg = nt.nodes.new("ShaderNodeBackground")
outn = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], outn.inputs["Surface"])

# camera — identical math to render_dog_fur_pro
_UP = {"x": (1,0,0), "-x": (-1,0,0), "y": (0,1,0), "-y": (0,-1,0), "z": (0,0,1), "-z": (0,0,-1)}[up_key]
C = Vector(center); up = Vector(_UP)
e1 = up.cross(Vector((1,0,0))); e1 = (up.cross(Vector((0,0,1))) if e1.length < 1e-3 else e1).normalized()
e2 = up.cross(e1).normalized()
view_ids = [only_view] if 0 <= only_view < views else range(views)
for k in view_ids:
    az = 2*math.pi*k/views + 0.4; el = math.radians(8 + 12*(k % 2)); r = camdist*size
    horiz = math.cos(az)*e1 + math.sin(az)*e2
    loc = C + r*(math.cos(el)*horiz + math.sin(el)*up)
    fwd = (C-loc).normalized(); right = fwd.cross(up).normalized(); tup = right.cross(fwd).normalized()
    cd = bpy.data.cameras.new(f"c{k}"); cd.lens = 55
    cam = bpy.data.objects.new(f"c{k}", cd); sc.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(loc) @ Matrix((right, tup, -fwd)).transposed().to_4x4()
    fname = out_name if out_name else f"furview_{k}.png"
    sc.camera = cam; sc.render.filepath = os.path.join(out, fname)
    bpy.ops.render.render(write_still=True)
print("SURFACE-DONE", flush=True)
