import genesis as gs
import imageio
import os
import trimesh
import warnings; warnings.filterwarnings("ignore")
from utils.tool import extract_name_from_path
import numpy as np
from PIL import Image
import math
from genesis.utils.geom import xyz_to_quat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Process, Queue, Event
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from scipy.spatial.transform import Rotation as R
import mapbox_earcut as earcut
import random
random.seed(42)

COLORS_255 = [
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (70, 70, 70),    # building
    (102, 102, 156), # wall
    (190, 153, 153), # fence
    (153, 153, 153), # pole
    (250, 170, 30),  # traffic light
    (220, 220, 0),   # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152), # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),   # person
    (255, 0, 0),     # rider
    (0, 0, 142),     # car
    (0, 0, 70),      # truck
    (0, 60, 100),    # bus
    (0, 80, 100),    # train
    (0, 0, 230),     # motorcycle
    (119, 11, 32),   # bicycle
]

COLORS = [(r/255.0, g/255.0, b/255.0) for r, g, b in COLORS_255]

def auto_align_pos_bottom_center_ea(glb_path, placement, scale, z_shift, big_scale=1.0, if_book=False):
    """
    使用 .glb 的 bounding box 自动将放置点对齐到底面中心。
    """
    z_offset = get_bottom_offset(glb_path, scale, if_book)
    x = -placement["pos"][0]
    y = -placement["pos"][1]
    z = placement["pos"][2] + z_offset + z_shift
    return [x* big_scale, y * big_scale, z * big_scale]

def auto_align_pos_bottom(glb_path, pose, scale):
    """
    使用 .glb 的 bounding box 自动将放置点对齐到底面。
    """
    z_offset = get_bottom_offset(glb_path, scale, False)
    return z_offset

def get_bottom_offset(glb_path, scale=1.0, if_book=False):
    mesh = trimesh.load(glb_path)
    mesh.apply_scale(scale)
    bbox_min = mesh.bounds[0]
    if if_book:
        shift_z = -bbox_min[2]
    else:
        shift_z = -bbox_min[1]
    return shift_z

def visualize_ea_layout_from_paths(mesh_path, merged, ea_result, dataset, save_path='./output/ea_result.png'):
    gs.init(backend=gs.gpu, theme='dark')

    scene = gs.Scene(
        show_viewer=False,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(2.5, -2.5, 2.5),
            camera_lookat=(2.5, 2.5, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            ambient_light=(0.5*1.0, 0.5*0.843, 0.5*0.678),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    # Add floor and back wall
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Rough(roughness=1., color=(0.9, 0.9, 0.9)),
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0., -5., 0.), euler=(90., 0., 0.)),
        surface=gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    if mesh_path.endswith('raw_model.obj'):
        texture_path = mesh_path.replace("raw_model.obj", "texture.png")
        img = Image.open(texture_path).convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        final_img = white_bg.convert("RGB")
        texture_path = texture_path.replace(".png", ".jpg")
        final_img.save(texture_path, quality=95)

    z_shift = get_bottom_offset(mesh_path)
    scene.add_entity(
        gs.morphs.Mesh(
            file=mesh_path,
            scale=1.0,
            pos=[0, 0, z_shift],
            euler=(90.0, 0.0, 180.0),
            collision=False,
        ),
        # surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=texture_path))
    )
    for idx, region_dict in enumerate(merged):
        assets = region_dict["items"]
        pos_dict = ea_result[str(idx)]
        for name, placement in pos_dict.items():
            item_name = name.rsplit("_", 1)[0]
            asset = assets[item_name]
            glb_path = asset[1]
            # if glb_path == "hssd-models/objects/6/688e8dde29b7a0f6d35a36a478f05050b2e5c262.glb":
            #     z_shift = z_shift+0.05
            euler = (90.0, 0.0, 180.0)
            if_book = False
            if item_name == "standing_book":
                euler = (180.0, 0.0, -90.0)
                if_book = True
            elif item_name == "flat book":
                euler = (90.0, 0.0, 180.0)
                if_book = False
            pos = auto_align_pos_bottom_center_ea(glb_path, placement, asset[3], z_shift, big_scale=1.0, if_book=if_book)
            scene.add_entity(
                gs.morphs.Mesh(file=glb_path, scale=asset[3], pos=pos, euler=euler, collision=False)
            )

    
    cam = scene.add_camera(
        res=(640*2, 480*2),
        pos=(0.0, 6.8, 1.3),
        lookat=(0.0, 0.0, 1.1),
        fov=30,
        GUI=False,
    )

    scene.build()
    rgb, _, _, _ = cam.render(rgb=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, rgb)
    print(f"Render saved at: {save_path}")
    gs.destroy()


def write_text(path):
    img = mpimg.imread(path)

    fig, ax = plt.subplots()
    ax.imshow(img)

    cell_size = 205
    img_size = 512 * 2  # 1024

    mark_positions = [
        (0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (3., 0., 0.), (4., 0., 0.), (5., 0., 0.),
        (0., 1., 0.), (1., 1., 0.), (2., 1., 0.), (3., 1., 0.), (4., 1., 0.), (5., 1., 0.),
        (0., 2., 0.), (1., 2., 0.), (2., 2., 0.), (3., 2., 0.), (4., 2., 0.), (5., 2., 0.),
        (0., 3., 0.), (1., 3., 0.), (2., 3., 0.), (3., 3., 0.), (4., 3., 0.), (5., 3., 0.),
        (0., 4., 0.), (1., 4., 0.), (2., 4., 0.), (3., 4., 0.), (4., 4., 0.), (5., 4., 0.),
        (0., 5., 0.), (1., 5., 0.), (2., 5., 0.), (3., 5., 0.), (4., 5., 0.), (5., 5., 0.),
    ]

    for x, y, _ in mark_positions:
        x_pos = x * cell_size
        y_pos = img_size - y * cell_size
        ax.text(x_pos, y_pos, f'({int(x)}, {int(y)})', fontsize=10, color='black', ha='center', va='center')
    ax.axis('off')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=200)


def render_scene_worker(q, assets, name, floor_xy, scene, objs, cam_top, cam_front, data_path, init_done):
    os.makedirs("./output", exist_ok=True)
    # init scene
    objs, scene, cam_top, cam_front = render_scene(assets, name, scene, objs, cam_top, cam_front, stop=False, init=True, data_path=data_path)
    
    init_done.set()  

    while True:
        cmd = q.get()
        if cmd == "update":
            render_scene(assets, name, floor_xy, scene, objs, cam_top, cam_front, stop=False, init=False, data_path=data_path)
        elif cmd == "stop":
            render_scene(assets, name, floor_xy, scene, objs, cam_top, cam_front, stop=True, init=False, data_path=data_path)
            gs.destroy()
            break


def render_scene_sync(assets, name, floor_xy, scene=None, objs=[], cam_top=None, cam_front=None, stop=False, init=False, data_path='./objects', render_queue=None):

    if init:
        q = Queue()
        init_done = Event()
        p = Process(target=render_scene_worker, args=(q, assets, name, scene, objs, cam_top, cam_front, data_path, init_done))
        p.start()
        init_done.wait()
        return objs, scene, cam_top, cam_front, p, q
    else:
        if render_queue is not None:
            if stop:
                render_queue.put("stop")
            else:
                render_queue.put("update")
        else:
            render_scene(assets, name, floor_xy, scene, objs, cam_top, cam_front, stop=stop, init=False, data_path=data_path)
        return objs, scene, cam_top, cam_front, None, None

def create_wall(
        wall_corner1,
        wall_corner2, 
        file_name, 
        window_locations=[],
        window_size=(0.08, 1.5, 1.18), 
        height=15.0
):
    i = 0
    same_val = wall_corner1[1]
    if wall_corner1[0] == wall_corner2[0]:
        i = 1
        same_val = wall_corner1[0]
    vertices = [
        list(wall_corner1) + [0],
        list(wall_corner2) + [0],
        list(wall_corner2) + [height],
        list(wall_corner1) + [height]
    ]

    create_door_vertices = [
        np.array([wall_corner1[i], 0]),
        np.array([wall_corner2[i], 0]),
        np.array([wall_corner2[i], height]),
        np.array([wall_corner1[i], height])
    ]
    rings = [len(create_door_vertices)]
    for window_loc in window_locations:
        create_door_vertices.append(np.array([window_loc[i] - window_size[1] / 2, 1.59 - window_size[2] / 2]))
        create_door_vertices.append(np.array([window_loc[i] + window_size[1] / 2, 1.59 - window_size[2] / 2]))
        create_door_vertices.append(np.array([window_loc[i] + window_size[1] / 2, 1.59 + window_size[2] / 2]))
        create_door_vertices.append(np.array([window_loc[i] - window_size[1] / 2, 1.59 + window_size[2] / 2]))
        vertices.append([(window_loc[i] - window_size[1] / 2) if i == 0 else same_val, (window_loc[i] - window_size[1] / 2) if i == 1 else same_val, 1.59 - window_size[2] / 2])
        vertices.append([(window_loc[i] + window_size[1] / 2) if i == 0 else same_val, (window_loc[i] + window_size[1] / 2) if i == 1 else same_val, 1.59 - window_size[2] / 2])
        vertices.append([(window_loc[i] + window_size[1] / 2) if i == 0 else same_val, (window_loc[i] + window_size[1] / 2) if i == 1 else same_val, 1.59 + window_size[2] / 2])
        vertices.append([(window_loc[i] - window_size[1] / 2) if i == 0 else same_val, (window_loc[i] - window_size[1] / 2) if i == 1 else same_val, 1.59 + window_size[2] / 2])
        rings.append(rings[-1] + 4)

    faces = earcut.triangulate_float32(np.stack(create_door_vertices), np.array(rings))
    faces.resize([len(faces) // 3, 3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces.tolist())
    mesh.export(file_name)


def _front_wall_indices(floor_xy):
    """Return boundary edges nearest the side camera at negative world Y."""
    if not floor_xy:
        return set()
    min_y = min(float(p[1]) for p in floor_xy)
    span = max(float(p[1]) for p in floor_xy) - min_y
    tol = max(1e-6, span * 1e-6)
    return {
        i
        for i, p1 in enumerate(floor_xy)
        if abs(float(p1[1]) - min_y) <= tol
        and abs(float(floor_xy[(i + 1) % len(floor_xy)][1]) - min_y) <= tol
    }


def render_scene(assets_np, name, floor_xy, door_windows, scene=None, objs=[], cam_top=None, cam_front=None, stop=False, init=False, data_path='./objects', if_seg=False):
    name = name.replace(' ', '_')
    
    if init:
        min_x = min([p[0] for p in floor_xy])
        max_x = max([p[0] for p in floor_xy])
        min_y = min([p[1] for p in floor_xy])
        max_y = max([p[1] for p in floor_xy])
        room_w = max_x - min_x
        room_h = max_y - min_y
        room_size = max(room_w, room_h) 
        fov = 60
        margin = 1.3
        z = (room_size / 2) / math.tan(math.radians(fov / 2)) * margin
        if room_size > 0.7:
            emissive=(0.35, 0.35, 0.35)
            intensity = 0.9
        else:
            emissive=(0.5, 0.5, 0.5)
            intensity = 1.3
            
        env_surface = gs.surfaces.Smooth(
            color=(255.0, 255.0, 255.0),
            emissive=emissive
        )
        
        gs.init(backend=gs.gpu, theme='dark')
        scene = gs.Scene(
            show_viewer = False,
            show_FPS = False,
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = False,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
            
            # renderer=gs.renderers.Rasterizer(),
            renderer=gs.renderers.RayTracer(env_surface = env_surface,
                env_radius  = 300.0,
                env_pos     = (0.0, 0.0, 0.0),
                env_euler   = (0.0, 0.0, 0.0),
                lights      = [
                    {'pos':(room_w/2,room_h/2,13.0),'color':(255.0, 229.5, 204.0),'intensity':intensity,'radius':0.4}
                ],
                tracing_depth=20
                )
            )

        wall_dir = [(90., 0., 0.), (90., 0., 270.), (90., 0., 180.), (90., 0., 90.)]
        front_wall_ids = _front_wall_indices(floor_xy)
        side_hidden_entities = []
        z_shift = auto_align_pos_bottom("repeated_objects/door/200-3.glb", [0.0, 0.0, 0.0], 1.0)
        door_pos = (door_windows["door_location"]["center"][0], door_windows["door_location"]["center"][1], z_shift)
        door_euler = wall_dir[door_windows["door_location"]["wall_id"]]
        door_entity = scene.add_entity(
            gs.morphs.Mesh(file="repeated_objects/door/200-3.glb", scale=1.0, pos=door_pos, euler=door_euler, collision=False),
            surface=gs.surfaces.Default(color=(0.39, 0.28, 0.19))
            # surface=gs.surfaces.Default(color=(0.52, 0.27, 0.11))
        )
        if door_windows["door_location"]["wall_id"] in front_wall_ids:
            side_hidden_entities.append(door_entity)

        z_shift = auto_align_pos_bottom("repeated_objects/window/window.obj", [0.0, 0.0, 0.0], 1.0)
        for window in door_windows["window_locations"]:
            window_pos = (window["center"][0], window["center"][1], z_shift+1.0)
            window_euler = wall_dir[window["wall_id"]]
            window_entity = scene.add_entity(
                gs.morphs.Mesh(file="repeated_objects/window/window.obj", scale=1.0, pos=window_pos, euler=window_euler, collision=False),
            )
            if window["wall_id"] in front_wall_ids:
                side_hidden_entities.append(window_entity)
        
        # Add floor
        floor = scene.add_entity(
            gs.morphs.Plane(),
            surface = gs.surfaces.Rough(roughness=1.0, color=(0.8, 0.8, 0.8)),
        )
        for i in range(len(floor_xy)):
            window_locs = []
            wall_corner1 = floor_xy[i]
            wall_corner2 = floor_xy[(i + 1) % len(floor_xy)]
            file_name = f"{name}_wall_{i}.obj"
            for win in door_windows["window_locations"]:
                if win["wall_id"] == i:
                    window_locs.append(win["center"])
            create_wall(wall_corner1, wall_corner2, file_name, window_locs)

            # wall (y=min_y, center along x)
            wall = scene.add_entity(
                gs.morphs.Mesh(file=file_name, scale=1.0, pos=(0., 0., 0.), euler=(0., 0., 0.), collision=False),
                surface = gs.surfaces.Rough(roughness=1.0, color=(0.85, 0.85, 0.85)),
            )
            if i in front_wall_ids:
                side_hidden_entities.append(wall)

        # objects on the floor
        for name_, asset in assets_np.items():
            base_name = name_.rsplit("_", 1)[0]
            mesh_key = base_name if base_name in data_path else name_
            if mesh_key not in data_path:
                print(f"[Warning] mesh path not found for '{name_}'.")
                continue
            z_shift = auto_align_pos_bottom(data_path[mesh_key], asset.pos, 1.0)
            position = tuple(asset.pos + np.array([0, 0, z_shift]))
            euler = (90., 0., 90. + asset.degree_rot[0])
            if data_path[mesh_key].endswith('raw_model.obj'):
                texture_path = data_path[mesh_key].replace("raw_model.obj", "texture.png")
                img = Image.open(texture_path).convert("RGBA")
                white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                white_bg.paste(img, mask=img.split()[3])
                final_img = white_bg.convert("RGB")
                texture_path = texture_path.replace(".png", ".jpg")
                final_img.save(texture_path, quality=95)

                objs.append(scene.add_entity(
                    gs.morphs.Mesh(file=data_path[mesh_key], scale=asset.scale, pos=position, euler=euler, collision=False),
                    surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=texture_path))
                ))
            else:
                objs.append(scene.add_entity(
                    gs.morphs.Mesh(file=data_path[mesh_key], scale=asset.scale, pos=position, euler=euler, collision=False),
                ))

        base_res = 2048
        aspect_ratio = room_w / room_h if room_h != 0 else 1.0

        if aspect_ratio >= 1:
            res_w = int(base_res)
            res_h = int(base_res / aspect_ratio)
        else:
            res_w = int(base_res * aspect_ratio)
            res_h = int(base_res)
            

        cam_top = scene.add_camera(
            res    = (res_w, res_h),
            pos    = ((min_x + max_x)/2, (min_y + max_y)/2, z),
            lookat = ((min_x + max_x)/2, (min_y + max_y)/2, 0),
            fov    = 60,
            GUI    = False,
        )

        cam_side = scene.add_camera(
            res    = (res_w, res_h),
            pos    = ((min_x + max_x)/2, min_y-max_y/1.8, z/2.0),
            lookat = ((min_x + max_x)/2, (min_y + max_y)/2, z/4.0-0.3),
            fov    = 60,
            GUI    = False,
        )
        scene.build()
        # cam_top.start_recording()
        rgb, _, _, _ = cam_top.render(rgb=True)
        # rgb_f, _, _, _ = cam_front.render(rgb=True)
        imageio.imwrite(f'{name}_top.png', rgb)
        for entity in side_hidden_entities:
            entity.set_pos((0.0, 0.0, -100.0))
        if side_hidden_entities:
            scene.step()
        rgb_side, _, _, _ = cam_side.render(rgb=True)
        imageio.imwrite(f'{name}_side.png', rgb_side)

        if stop:
            print("destroying scene")
            gs.destroy()
            return
        return objs, scene, cam_top, cam_side
    
    else:
        i = 0
        for name_, asset in assets_np.items():
            base_name = name_.rsplit("_", 1)[0]
            mesh_key = base_name if base_name in data_path else name_
            if mesh_key not in data_path:
                print(f"[Warning] mesh path not found for '{name_}'.")
                continue
            z_shift = auto_align_pos_bottom(data_path[mesh_key], asset.pos, 1.0)
            position = tuple(asset.pos + np.array([0, 0, z_shift]))
            euler = np.array([90., 0., 90. + asset.degree_rot[0]])
            quat = xyz_to_quat(euler)
            objs[i].set_pos(position)
            objs[i].set_quat(quat)
            i += 1
        scene.step()
        rgb, _, _, _ = cam_top.render(rgb=True)
        # rgb_f, _, _, _ = cam_front.render(rgb=True)
        imageio.imwrite(f'{name}_after_optimization_top.png', rgb)
        # imageio.imwrite(f'./output/{name}_after_optimization_front.png', rgb_f)
        # write_text(f'./output/{name}_after_optimization_top.png')
        if not stop:
            return objs, scene, cam_top, cam_front
        else:
            cam_top.stop_recording(save_to_filename=f'{name}_video.mp4', fps=10)
            # for obj in objs:
            #     collision_pair = obj.detect_collision()
            rgb, _, _, _ = cam_top.render(rgb=True)
            imageio.imwrite(f'{name}_after_optimization_top.png', rgb)
            # rgb_f, _, _, _ = cam_front.render(rgb=True)
            # imageio.imwrite(f'./output/{name}_after_optimization_front.png', rgb_f)

        return None


def _resolve_render_output_file(output_path, name, view_suffix):
    """Build output PNG path and ensure parent dirs exist (supports nested ``name``)."""
    out_path = os.path.join(output_path, f"{name}_{view_suffix}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    return out_path


def render_scene_fin(assets_np, door_windows, name, ids_floor, result_small, merged_small, data_path, floor_xy, output_path, top=True):
    name = name.replace(' ', '_')

    min_x = min([p[0] for p in floor_xy])
    max_x = max([p[0] for p in floor_xy])
    min_y = min([p[1] for p in floor_xy])
    max_y = max([p[1] for p in floor_xy])
    room_w = max_x - min_x
    room_h = max_y - min_y
    room_size = max(room_w, room_h) 
    front_wall_ids = _front_wall_indices(floor_xy)
    fov = 60
    margin = 1.3
    z = (room_size / 2) / math.tan(math.radians(fov / 2)) * margin
    
    gs.init(backend=gs.gpu, theme='dark')
    if top:
        if room_size > 0.7:
            emissive=(0.3, 0.3, 0.3)
            intensity = 0.5
        else:
            emissive=(0.5, 0.5, 0.5)
            intensity = 1.3
        env_surface = gs.surfaces.Smooth(
            color=(255.0, 229.5, 204.0),
            emissive=emissive
        )
        scene = gs.Scene(
            show_viewer = False,
            show_FPS = False,
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = False,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.RayTracer(env_surface = env_surface,
                env_radius  = 300.0,
                env_pos     = (0.0, 0.0, 0.0),
                env_euler   = (0.0, 0.0, 0.0),
                lights      = [
                    {'pos':(room_w/2,room_h/2,14.0),'color':(255.0, 229.5, 204.0),'intensity':intensity,'radius':0.4}
                ],
                tracing_depth=25
                )
        )
    else:
        env_surface = gs.surfaces.Smooth(
            color=(255.0, 229.5, 204.0),
            emissive=(0.1, 0.1, 0.1)        
        )
        scene = gs.Scene(
            show_viewer = False,
            show_FPS = False,
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = False,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.RayTracer(env_surface = env_surface,
                env_radius  = 500.0,
                env_pos     = (0.0, 0.0, 0.0),
                env_euler   = (0.0, 0.0, 0.0),
                lights      = [
                    {'pos':(room_w/2,room_h/2,12.0),'color':(255.0, 229.5, 204.0),'intensity':0.37,'radius':0.30}
                ],
                tracing_depth=25
                )
        )

    # Add floor
    floor = scene.add_entity(
        gs.morphs.Box(size=(room_w, room_h, 0.01), pos=(room_w/2, room_h/2, -0.005)),
        surface = gs.surfaces.Rough(roughness=1.0, color=(0.8, 0.8, 0.8)),
    )
    wall_dir = [(90., 0., 0.), (90., 0., 270.), (90., 0., 180.), (90., 0., 90.)]
    if top:
        z_shift = auto_align_pos_bottom("repeated_objects/door/200-3.glb", [0.0, 0.0, 0.0], 1.0)
        door_pos = (door_windows["door_location"]["center"][0], door_windows["door_location"]["center"][1], z_shift)
        door_euler = wall_dir[door_windows["door_location"]["wall_id"]]
        scene.add_entity(
            gs.morphs.Mesh(file="repeated_objects/door/200-3.glb", scale=1.0, pos=door_pos, euler=door_euler, collision=False),
            surface=gs.surfaces.Rough(roughness=1.0, color=(0.39, 0.28, 0.19))
        )

        z_shift = auto_align_pos_bottom("repeated_objects/window/window.obj", [0.0, 0.0, 0.0], 1.0)
        for window in door_windows["window_locations"]:
            window_pos = (window["center"][0], window["center"][1], z_shift+1.0)
            window_euler = wall_dir[window["wall_id"]]
            scene.add_entity(
                gs.morphs.Mesh(file="repeated_objects/window/window.obj", scale=1.0, pos=window_pos, euler=window_euler, collision=False),
            )
        
        for i in range(len(floor_xy)):
            window_locs = []
            wall_corner1 = floor_xy[i]
            wall_corner2 = floor_xy[(i + 1) % len(floor_xy)]
            file_name = f"{output_path}/{name}_wall_{i}.obj"
            for win in door_windows["window_locations"]:
                if win["wall_id"] == i:
                    window_locs.append(win["center"])
            create_wall(wall_corner1, wall_corner2, file_name, window_locs)

            # wall (y=min_y, center along x)
            wall = scene.add_entity(
                gs.morphs.Mesh(file=file_name, scale=1.0, pos=(0., 0., 0.), euler=(0., 0., 0.), collision=False),
                surface = gs.surfaces.Rough(roughness=1.0, color=(0.85, 0.85, 0.85)),
            )
    else:
        if door_windows["door_location"]["wall_id"] not in front_wall_ids:
            z_shift = auto_align_pos_bottom("repeated_objects/door/200-3.glb", [0.0, 0.0, 0.0], 1.0)
            door_pos = (door_windows["door_location"]["center"][0], door_windows["door_location"]["center"][1], z_shift)
            door_euler = wall_dir[door_windows["door_location"]["wall_id"]]
            scene.add_entity(
                gs.morphs.Mesh(file="repeated_objects/door/200-3.glb", scale=1.0, pos=door_pos, euler=door_euler, collision=False),
                surface=gs.surfaces.Default(color=(0.33, 0.24, 0.16))
            )

        z_shift = auto_align_pos_bottom("repeated_objects/window/window.obj", [0.0, 0.0, 0.0], 1.0)
        for window in door_windows["window_locations"]:
            if window["wall_id"] in front_wall_ids:
                continue
            window_pos = (window["center"][0], window["center"][1], z_shift+1.0)
            window_euler = wall_dir[window["wall_id"]]
            scene.add_entity(
                gs.morphs.Mesh(file="repeated_objects/window/window.obj", scale=1.0, pos=window_pos, euler=window_euler, collision=False),
            )
        
        for i in range(1, len(floor_xy)):
            if i in front_wall_ids:
                continue
            window_locs = []
            wall_corner1 = floor_xy[i]
            wall_corner2 = floor_xy[(i + 1) % len(floor_xy)]
            file_name = f"{output_path}/{name}_wall_{i}.obj"
            for win in door_windows["window_locations"]:
                if win["wall_id"] == i:
                    window_locs.append(win["center"])
            create_wall(wall_corner1, wall_corner2, file_name, window_locs)

            # wall (y=min_y, center along x)
            wall = scene.add_entity(
                gs.morphs.Mesh(file=file_name, scale=1.0, pos=(0., 0., 0.), euler=(0., 0., 0.), collision=False),
                surface = gs.surfaces.Rough(roughness=1.0, color=(0.85, 0.85, 0.85)),
            )

    objs = []
    labels_idx = {}
    parent_z_shifts = {}
    # objects on the floor
    for name_full, asset in assets_np.items():
        n = name_full.rsplit("_", 1)[0]
        z_shift = auto_align_pos_bottom(data_path[n], asset['pos'], asset['scale'])
        parent_z_shifts[name_full] = z_shift
        position = tuple(asset['pos'] + np.array([0, 0, z_shift]))
        euler = (90., 0., 90. + asset['phy'][0])

        if data_path[n].endswith('raw_model.obj'):
            texture_path = data_path[n].replace("raw_model.obj", "texture.png")
            img = Image.open(texture_path).convert("RGBA")
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            white_bg.paste(img, mask=img.split()[3])
            final_img = white_bg.convert("RGB")
            texture_path = texture_path.replace(".png", ".jpg")
            final_img.save(texture_path, quality=95)

            objs.append(scene.add_entity(
                gs.morphs.Mesh(file=data_path[n], scale=asset['scale'], pos=position, euler=euler, collision=False),
                surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=texture_path))
            ))
        else:
            objs.append(scene.add_entity(
                gs.morphs.Mesh(file=data_path[n], scale=asset['scale'], pos=position, euler=euler, collision=False),
            ))

        if n in ids_floor:
            if n not in labels_idx:
                labels_idx[n] = []
            labels_idx[n].append(name_full)

    def _item_glb_path_and_scale(item):
        if isinstance(item, dict):
            return str(item["name"]), float(item["scale"])
        return str(item[1]), float(item[3])

    def _iter_merged_regions(merged):
        if isinstance(merged, dict):
            def _sort_key(k):
                try:
                    return int(k)
                except (TypeError, ValueError):
                    return k
            for k in sorted(merged.keys(), key=_sort_key):
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    idx = k
                yield idx, merged[k]
        else:
            for idx, region in enumerate(merged):
                yield idx, region

    for id in ids_floor:
        ea_result = result_small[id]
        merged = merged_small[id]
        for region_idx, region_dict in _iter_merged_regions(merged):
            assets = region_dict["items"]
            pos_dict = ea_result[str(region_idx)]
            for name_, placement in pos_dict.items():
                item_name = name_.rsplit("_", 1)[0]
                if item_name not in assets:
                    continue
                asset = assets[item_name]
                glb_path, local_scale = _item_glb_path_and_scale(asset)
                
                euler = (0.0, 0.0, 180.0)
                if_book = False
                l_idxs = labels_idx[id]
                for l_idx in l_idxs:
                    parent_z_shift = parent_z_shifts.get(l_idx, 0.0)
                    r_big = R.from_euler('XYZ', (0., 0., 90. + assets_np[l_idx]['phy'][0]), degrees=True)
                    scale = assets_np[l_idx]['scale'] * local_scale
                    if item_name == "standing_book":
                        if_book = True
                        pos_local = auto_align_pos_bottom_center_ea(glb_path, placement, scale, parent_z_shift, big_scale=assets_np[l_idx]['scale'], if_book=if_book)
                        
                        pos_local = np.array([-pos_local[0], -pos_local[1], pos_local[2]]) 
                        pos_world = r_big.apply(pos_local) + np.array(assets_np[l_idx]['pos'])
                        euler_local = (180.0, 0.0, -90.0)
                        r_local = R.from_euler('XYZ', euler_local, degrees=True)
                        r_world = r_big * r_local
                        euler = tuple(r_world.as_euler('XYZ', degrees=True))
                    else:
                        if_book = False
                        pos_local = auto_align_pos_bottom_center_ea(glb_path, placement, scale, parent_z_shift, big_scale=assets_np[l_idx]['scale'], if_book=if_book)
                        pos_local = np.array([-pos_local[0], -pos_local[1], pos_local[2]]) 
                        pos_world = r_big.apply(pos_local) + np.array(assets_np[l_idx]['pos'])
                        euler = (90.0, 0.0, 90. + assets_np[l_idx]['phy'][0])

                    scene.add_entity(
                        gs.morphs.Mesh(file=glb_path, scale=scale, pos=pos_world, euler=euler, collision=False)
                    )

    base_res = 1280
    aspect_ratio = room_w / room_h if room_h != 0 else 1.0

    if aspect_ratio >= 1:
        res_w = int(base_res)
        res_h = int(base_res / aspect_ratio)
    else:
        res_w = int(base_res * aspect_ratio)
        res_h = int(base_res)
        
    # cam_top = scene.add_camera(
    #     res    = (res_w, res_h),
    #     pos    = ((min_x + max_x)/2, (min_y + max_y)/2, z),
    #     lookat = ((min_x + max_x)/2, (min_y + max_y)/2, 0),
    #     fov    = 60,
    #     GUI    = False,
    # )
    cam_top = scene.add_camera(
        res    = (res_w, res_h),
        pos    = ((min_x + max_x)/2, (min_y + max_y)/2, z),
        lookat = ((min_x + max_x)/2, (min_y + max_y)/2, 0),
        fov    = 60,
        GUI    = False,
    )
    cam_side = scene.add_camera(
        res    = (res_w, res_h),
        pos    = ((min_x + max_x)/2, min_y-max_y/1.8, z/2.0),
        lookat = ((min_x + max_x)/2, (min_y + max_y)/2, z/4.0-0.3),
        fov    = 60,
        GUI    = False,
    )
    scene.build()

    # Render top view image
    if top:
        rgb, _, _, _ = cam_top.render(rgb=True)
        imageio.imwrite(_resolve_render_output_file(output_path, name, "top"), rgb)
    else:
        rgb, _, _, _ = cam_side.render(rgb=True)
        imageio.imwrite(_resolve_render_output_file(output_path, name, "side"), rgb)
