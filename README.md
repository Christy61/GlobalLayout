# GlobalLayout

Code of "Global Graph-Validated Optimization for VLM-based 3D Indoor Scene Generation"
<img width="2839" height="1022" alt="image" src="https://github.com/user-attachments/assets/10ecd02b-1a37-4534-9229-2ba26e2186a9" />

GlobalLayout generates indoor layouts from scene descriptions and candidate 3D assets. The main entry point is `run.py`.

## Environment and datasets

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Place all datasets in the project root:

```text
GlobalLayout/
├── 3D-FUTURE-model/
├── hssd-models/
│   └── objects/
├── hssd_render/
│   └── objects/
├── test_asset_dir/
└── objaverse/
```

The benchmark tasks use `./3D-FUTURE-model` by default. HSSD, test assets, and Objaverse assets must likewise be available under `./hssd-models`, `./test_asset_dir`, and `./objaverse`, respectively. Rendered HSSD previews are read from `./hssd_render`.

Copy the environment variable template and add your own OpenAI API key:

```bash
cp .env.example .env
```

## Data preprocessing

### Rendering HSSD asset previews

The HSSD retrieval pipeline reads front- and top-view preview images from `./hssd_render`. After placing the source GLB files under `./hssd-models/objects`, install the bundled Blender runtime:

```bash
cd dataset/render
bash install_blender.sh
cd ../..
```

Before rendering, set `INPUT_FOLDER` and `OUTPUT_FOLDER` in `dataset/render/render_bpy.py` to the absolute paths of the source and output directories, respectively. For example:

```python
INPUT_FOLDER = "/path/to/GraphLayout/hssd-models/objects"
OUTPUT_FOLDER = "/path/to/GraphLayout/hssd_render/objects"
```

Then run the renderer from the project root:

```bash
python dataset/render/render.py \
  --objects_root ./hssd-models/objects \
  --timeout 800
```

The script renders the assets in parallel with Blender/Cycles and preserves the HSSD subdirectory structure. Each source asset produces two transparent PNG previews:

```text
hssd-models/objects/<group>/<asset_id>.glb
                  ↓
hssd_render/objects/<group>/<asset_id>_front.png
hssd_render/objects/<group>/<asset_id>_top.png
```

Rendering requires an NVIDIA GPU supported by Blender CUDA or OptiX. Existing preview files are skipped, so the same command can be rerun after an interruption. Increase `--timeout` if individual assets need more than 800 seconds to render.

### Precomputing placeable surfaces

Small-object placement can use a locally generated index of valid horizontal support surfaces. Run the precomputation from the project root after placing 3D-FUTURE and Objaverse in the directories shown above. The following command processes both datasets with 16 worker processes:

```bash
python utils/precompute_placeable_assets.py \
  --threed_front_root ./3D-FUTURE-model \
  --objaverse_root ./objaverse \
  --datasets 3d_front objaverse \
  --workers 16 \
  --output_dir ./assets_feature/placeable
```

Adjust `--workers` to the CPU and memory available on the machine. The command reads meshes from `./3D-FUTURE-model` and `./objaverse` and writes:

```text
assets_feature/placeable/
├── placeable_assets.json
├── 3d_front/<asset_id>/support_regions_3d_*.png
└── objaverse/<asset_id>/support_regions_3d_*.png
```

`run.py` loads `./assets_feature/placeable/placeable_assets.json` by default when placing small objects. If the file is absent, it falls back to computing support regions at runtime with the legacy `find_plane` pipeline. Use `--resume` to continue an interrupted precomputation without reprocessing entries already stored in the output JSON.

## Precomputed asset features

Download the precomputed retrieval indexes and extract both archives in the project root:

- [3D-FUTURE asset features](https://drive.google.com/open?id=1ie3b_t8O4xwnkoDG_YzPuFUfxIRHgPNl)
- [HSSD asset features](https://drive.google.com/open?id=13uSHagpLML3btN4-uOdpZG7z1totYnMf)

```bash
tar -xzf assets_feature_core.tar.gz
tar -xzf assets_feature_hssd.tar.gz
```

The first archive creates the core files under `./assets_feature`; the second creates `./assets_feature_hssd`. Placeable-surface data is not included and must be generated locally as described below.

## Example

Run one bedroom scene from the project root:

```bash
python run.py --scene_json_file benchmark_tasks/bedroom/bedroom_0.json
```

This example expects its dataset at `./3D-FUTURE-model`, `./objaverse` and `./hssd-models`. Results are written to `./results_areas` by default.
