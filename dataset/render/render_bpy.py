import bpy
import os
import math
import mathutils
import argparse
import sys
from tqdm import tqdm


# ========== setting ==========
# INPUT_FOLDER = "small_assets/ai2thorhab-uncompressed/assets"
# OUTPUT_FOLDER = "small_assets/render"
INPUT_FOLDER = "/dataset/huangjialu/hssd_objects/objects"
OUTPUT_FOLDER = "/dataset/huangjialu/hssd_render/objects"
IMAGE_RESOLUTION = (512, 512)

class DummyStream:
    def write(self, x): pass
    def flush(self): pass

sys.stdout = DummyStream()
sys.stderr = DummyStream()
tqdm_out = sys.__stderr__ 

def enable_gpu_rendering():
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.refresh_devices()
    for device in prefs.devices:
        if device.type in {'CUDA', 'OPTIX'}:
            device.use = True
    bpy.context.scene.cycles.device = "GPU"

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes, bpy.data.lights, bpy.data.cameras,
        bpy.data.materials, bpy.data.images, bpy.data.curves
    ):
        for block in datablocks:
            datablocks.remove(block, do_unlink=True)

def setup_render(out_path):
    scene = bpy.context.scene
    scene.render.resolution_x = IMAGE_RESOLUTION[0]
    scene.render.resolution_y = IMAGE_RESOLUTION[1]
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.render.filepath = out_path
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

def setup_camera_light(center, size, max_dim, view='front'):
    margin_ratio = 2.3
    cam_dist = max_dim * margin_ratio
    tilt_deg = 15
    tilt_rad = math.radians(tilt_deg)

    if view == 'front':
        horiz_dist = cam_dist * math.cos(tilt_rad)
        vert_dist = cam_dist * math.sin(tilt_rad)
        cam_loc = (center[0], center[1] - horiz_dist, center[2] + vert_dist + 0.1 * size[2])
        cam_rot = (math.radians(90 - tilt_deg), 0, 0)
        light_offset = (0, max_dim * 0.7, max_dim * 0.3)

    elif view == 'top':
        cam_loc = (center[0], center[1], center[2] + cam_dist)
        cam_rot = (math.radians(0), 0, 0)
        light_offset = (0, 0, max_dim * 0.5)

    bpy.ops.object.camera_add(location=cam_loc, rotation=cam_rot)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    # 灯光
    light_loc = (cam_loc[0] + light_offset[0],
                 cam_loc[1] + light_offset[1],
                 cam_loc[2] + light_offset[2])
    bpy.ops.object.light_add(type='SUN', location=light_loc)
    sun = bpy.context.object
    sun.data.energy = 1.5

    direction = mathutils.Vector(center) - sun.location
    sun.rotation_mode = 'QUATERNION'
    sun.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

    return cam, sun

def compute_scene_bounds():
    objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not objs:
        return None, None, None

    min_corner = [min(obj.bound_box[i][j] * obj.scale[j] + obj.location[j] for obj in objs for i in range(8)) for j in range(3)]
    max_corner = [max(obj.bound_box[i][j] * obj.scale[j] + obj.location[j] for obj in objs for i in range(8)) for j in range(3)]

    center = [(min_corner[i] + max_corner[i]) / 2 for i in range(3)]
    size = [max_corner[i] - min_corner[i] for i in range(3)]
    max_dim = max(size)
    return center, size, max_dim

# ========== 主流程 ==========
def render_all_glbs(start, end):
    start, end = int(start), int(end)
    for filedir in os.listdir(INPUT_FOLDER)[start:end]:
        files = [f for f in os.listdir(f"{INPUT_FOLDER}/{filedir}") if f.endswith(".glb")]
        os.makedirs(f"{OUTPUT_FOLDER}/{filedir}", exist_ok=True)

        for fname in tqdm(files, desc="Rendering", ncols=80, file=tqdm_out):
            clear_scene()
            full_path = os.path.join(INPUT_FOLDER, filedir, fname)
            bpy.ops.import_scene.gltf(filepath=full_path)
            center, size, max_dim = compute_scene_bounds()

            # Front view
            output_path = os.path.join(OUTPUT_FOLDER, filedir, os.path.splitext(fname)[0] + "_front.png")
            if not os.path.exists(output_path):
                cam, sun = setup_camera_light(center, size, max_dim, view='front')
                setup_render(output_path)
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(cam, do_unlink=True)
                bpy.data.objects.remove(sun, do_unlink=True)

            # Top view
            output_path = os.path.join(OUTPUT_FOLDER, filedir, os.path.splitext(fname)[0] + "_top.png")
            if not os.path.exists(output_path):
                cam, sun = setup_camera_light(center, size, max_dim, view='top')
                setup_render(output_path)
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(cam, do_unlink=True)
                bpy.data.objects.remove(sun, do_unlink=True)


# ========== 启动 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--start", required=True, help="Start index.")
parser.add_argument("--end", required=True, help="End index.")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
enable_gpu_rendering()
render_all_glbs(args.start, args.end)
