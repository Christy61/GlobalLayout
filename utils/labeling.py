'''
Here using the genesis to generate the toy scene, which contains the following assets:


4. A top-down view of the current scene, with a marked global frame, 1-meter grid, labeled assets, and front-facing orientation arrows. Walls are also labeled with orientation arrows.
5. A side view of the current scene, with the global frame and 1-meter grid.
6. A top-down view of each new asset in an empty scene, facing the positive X-axis, labeled with its name and front-facing arrow.

'''

import genesis as gs
import imageio
import warnings; warnings.filterwarnings("ignore")
from multiprocessing import Process
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import multiprocessing
import os
from pathlib import Path
multiprocessing.set_start_method('spawn', force=True)

def two_views_of_current_scene(mode, floor_xy, obj_path=None, rot=(90., 0, 0.), name=None, scale=1.0, pose=np.eye(4)):
    gs.init(backend=gs.gpu, theme='dark')

    scene = gs.Scene(
        show_viewer = False,
        viewer_options = gs.options.ViewerOptions(
            res           = (1280, 960),
            camera_pos    = (3.5, 0.0, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 40,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame  = False,
            show_cameras     = False,
            plane_reflection = True,
            ambient_light    = (0.2*1.0, 0.2*0.843, 0.2*0.678),
        ),
        # renderer=gs.renderers.Rasterizer(),
        renderer=gs.renderers.RayTracer(),
    )


    # floor_xy: list of (x, y) tuples, e.g. [(0.0, 0.0), (6.0, 0.0), (6.0, 7.0), (0.0, 7.0)]
    # Compute wall positions and orientations from floor_xy
    height = 1.5
    # Add floor
    floor = scene.add_entity(
        gs.morphs.Plane(),
        surface = gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    min_x = min([p[0] for p in floor_xy])
    max_x = max([p[0] for p in floor_xy])
    min_y = min([p[1] for p in floor_xy])
    max_y = max([p[1] for p in floor_xy])
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    room_w = max_x - min_x
    room_h = max_y - min_y
    room_size = max(room_w, room_h) 
    fov = 60
    margin = 1.1
    z = (room_size / 2) / math.tan(math.radians(fov / 2)) * margin
    # left wall (x=min_x, center along y)
    wall = scene.add_entity(
        gs.morphs.Plane(pos=(min_x, center_y, 0.), euler=(90., 0., 90.)),
        surface = gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    # right wall (x=max_x, center along y)
    wall = scene.add_entity(
        gs.morphs.Plane(pos=(max_x, center_y, 0.), euler=(90., 0., 270.)),
        surface = gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    # back wall (y=max_y, center along x)
    wall = scene.add_entity(
        gs.morphs.Plane(pos=(center_x, max_y, 0.), euler=(90., 0., 180.)),
        surface = gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    # front wall (y=min_y, center along x)
    wall = scene.add_entity(
        gs.morphs.Plane(pos=(center_x, min_y, 0.), euler=(90., 0., 0.)),
        surface = gs.surfaces.Rough(roughness=1., color=(0.8, 0.8, 0.8)),
    )
    # Add origin dot at floor_xy[0]
    origin_dot = scene.add_entity(
        gs.morphs.Sphere(radius=0.05, pos=(floor_xy[0][0], floor_xy[0][1], 0.), euler=(0., 0., 0.)),
        surface=gs.surfaces.Rough(color=(0., 0., 1.)),
    )
    if mode=='top_down':
        # Compute grid bounds from floor_xy
        min_x = min([p[0] for p in floor_xy])
        max_x = max([p[0] for p in floor_xy])
        min_y = min([p[1] for p in floor_xy])
        max_y = max([p[1] for p in floor_xy])
        mark_positions = []
        for x in range(int(np.ceil(min_x)), int(np.floor(max_x))+1):
            for y in range(int(np.ceil(min_y)), int(np.floor(max_y))+1):
                mark_positions.append((float(x), float(y), 0.))
        for pos in mark_positions:
            scene.add_entity(
                gs.morphs.Sphere(radius=0.05, pos=pos, euler=(0., 0., 0.)),
                surface=gs.surfaces.Rough(color=(1., 0., 0.)),
            )

        # top-down view of the scene
        cam = scene.add_camera(
            res    = (512, 512),
            pos    = ((min_x + max_x)/2, (min_y + max_y)/2, z),
            lookat = ((min_x + max_x)/2, (min_y + max_y)/2, 0),
            fov    = 60,
            GUI    = False,
        )

        scene.build()
        rgb, _, _, _ = cam.render(rgb=True)
        imageio.imwrite(f'./input_cache_areas/top_down_cs.png', rgb)

    if mode == 'obj':
        # top-down view of the scene
        cam = scene.add_camera(
            res    = (512, 512),
            pos    = ((min_x + max_x)/2, (min_y + max_y)/2, z),
            lookat = ((min_x + max_x)/2, (min_y + max_y)/2, 0),
            fov    = 60,
            GUI    = False,
        )
        obj = scene.add_entity(
            gs.morphs.Mesh(file=obj_path, scale=scale, pos=((min_x + max_x)/2, (min_y + max_y)/2, 0.), euler=rot),
        )
        scene.build()
        bounds = obj.get_AABB()
        min_z = bounds[0][2]
        translation_z = -min_z
        new_pos = [ (min_x + max_x)/2, (min_y + max_y)/2, 0 ] + np.array([0, 0, translation_z.item()])
        obj.set_pos(new_pos)
        scene.step()
        size = bounds[1] - bounds[0]
        with open(f'./input_cache_areas/{name}.txt', 'w') as f:
            f.write(f"{str(scale)}-{str(size.cpu().tolist())}:{str(translation_z.item())}")

        rgb, _, _, _ = cam.render(rgb=True)
        imageio.imwrite(f'./input_cache_areas/top_down_obj_{name}.png', rgb)


def rendering_views(labels, dataset_path, scale_dict, floor_xy):

    # here, I have to render them in parallel to avoid gs been re-initialized.
    # TODO; As a result the terminal output is a mess, try to find a way to suppress the output or clean it.
    
    processes = []
    # img_path = "toy_example/input_img.png"
    # mask_path_dict = {label: f"output/input_img/{label}_mask.png" for label in labels}
    # mesh_path_dict = {label: f"output/input_img/Mesh/tex_obj_{label}.glb" for label in labels}
    # scale_dict, pose_dict = run_foundationpose_scale(img_path, mask_path_dict, mesh_path_dict)
    pose_dict = {label: np.eye(4) for label in labels}
    label_used = []
    for label in labels:
        if label in label_used:
            continue
        p = Process(target=two_views_of_current_scene, args=('obj', floor_xy, dataset_path[label], (90., 0, 90.), label.lower(), scale_dict[label], pose_dict[label]))
        label_used.append(label)
        processes.append(p)
        p.start()
    
    p = Process(target=two_views_of_current_scene, args=('top_down', floor_xy))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()
        if p.exitcode not in (0, None):
            print(f"[labeling][Warning] render subprocess exited with code {p.exitcode}")
    
    os.makedirs("./input_cache_areas", exist_ok=True)
    for obj_name in labels:
        # using matplotlib to write the name of object and the front-facing arrow on the image.
        img_path = Path(f'./input_cache_areas/top_down_obj_{obj_name}.png')
        if not img_path.exists():
            print(f"[labeling][Warning] missing object render cache, skip label image: {img_path}")
            continue
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        x_pos = 256
        y_pos = 128
        ax.text(x_pos, y_pos, obj_name, fontsize=9, color='black')
        ax.arrow(512/2, 512/2, 30, 0, head_width=6, head_length=6, fc='blue', ec='blue')
        ax.axis('off')
        fig.savefig(f'./input_cache_areas/top_down_obj_{obj_name.lower()}_text.png', bbox_inches='tight', pad_inches=0.0, dpi=200)
