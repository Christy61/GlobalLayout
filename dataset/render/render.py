import os
import subprocess
from loguru import logger
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def render_glb(start, end, timeout):
    package_folder = os.path.dirname(__file__)
    convert_command = f"{package_folder}/blender-4.4.3-linux-x64/blender --background --python {package_folder}/render_bpy.py -- --start {start} --end {end} "
    try:
        subprocess.run(convert_command, check=True, shell=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        logger.error(f"transform failed: {e}")
        return False
    return True


def get_object_dirs(objects_root):
    return [os.path.join(objects_root, d) for d in os.listdir(objects_root)
            if os.path.isdir(os.path.join(objects_root, d))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects_root', type=str, default='hssd_objects/objects')
    parser.add_argument('--timeout', type=int, default=800)
    args = parser.parse_args()

    object_dirs = get_object_dirs(args.objects_root)
    total = len(object_dirs)
    batch_size = 7

    def render_wrapper(idx):
        start = idx
        end = idx + 1
        return render_glb(start, end, args.timeout)

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(render_wrapper, i) for i in range(total)]
        for future in as_completed(futures):
            if not future.result():
                logger.error("A render task failed.")


if __name__ == '__main__':
    main()
