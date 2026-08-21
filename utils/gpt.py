from openai import OpenAI
import base64
import time
import os
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

from utils.task_assets import format_asset_categories_for_prompt
from utils.openai_credentials import resolve_openai_api_key


# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda"
# )

# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.float16, device_map="auto", load_in_8bit=True, 
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

class GPT:
    def __init__(self, cfg, if_test=False):
        self.cfg = cfg
        self.api_key = resolve_openai_api_key(getattr(cfg, "gpt_api_key", None))
        self.cfg.gpt_api_key = self.api_key
        if if_test:
            self.version = "gpt-4o"
        else:
            self.version = cfg.gpt_version
        self.max_retries = 8

    def __call__(self, content_system, content_user, output_add=None):
        client = OpenAI(api_key=self.api_key)

        attempts = 0
        while attempts < self.max_retries:
            try:
                if output_add:
                    output_add.extend([
                        {
                            "role": "user",
                            "content": content_user
                        }
                    ])
                    messages = output_add.copy()
                else:
                    messages = [
                                {
                                    "role": "system",
                                    "content": content_system
                                },
                                {
                                    "role": "user",
                                    "content": content_user
                                }
                            ]
                gpt_seed = int(getattr(self.cfg, "gpt_seed", 1234) or 1234)
                output = client.chat.completions.create(
                    model=self.version,
                    messages=messages,
                    max_tokens=3000,
                    temperature=0.01,
                    seed=gpt_seed,
                    )
                return output.choices[0].message.content
            
            except Exception as e:
                attempts += 1
                print(f"Error querying {self.version} API: {e}")
                if attempts < self.max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to query {self.version} API after {self.max_retries} attempts.")
                    return None

    # def __call__(self, content_system, content_user, output_add=None):
    #     start_time = time.time()
    #     messages = [
    #                 {
    #                     "role": "system",
    #                     "content": content_system
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": content_user
    #                 }
    #             ]

    #     # Preparation for inference
    #     inputs = processor.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_generation_prompt=True,
    #         return_dict=True,
    #         return_tensors="pt"
    #     )
    #     inputs = inputs.to(model.device)

    #     # Inference: Generation of the output
    #     generated_ids = model.generate(**inputs, max_new_tokens=5000)
    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]
    #     output_text = processor.batch_decode(
    #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )

    #     end_time = time.time()
    #     execution_time = end_time - start_time
        
    #     print(f"api time: {execution_time:.2f} s")
    #     return output_text[0]


    def encode_image(self, image_path, use_transformers=False):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            str: Encoded image
        """
        if use_transformers:
            return image_path
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        

    def create_door(self, task, floor_v):
        task_description = task["task_description"]

        with open("system_prompts/room.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/floor_plan.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                task_description=task_description, 
                floor_v=floor_v)

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            }
        ]

        return content_system, content_user
    

    def get_areas(self, task, floor_v, output_dir, room_type, assets_categories=None):
        layout_criteria = task["layout_criteria"]
        room_description = task["task_description"]
        categories_text = format_asset_categories_for_prompt(assets_categories)
        with open("system_prompts/area.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/get_areas.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                layout_criteria=layout_criteria, 
                room_description=room_description,
                floor_v=floor_v,
                room_type=room_type,
                categories=categories_text,
                )
    
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        imageA = self.encode_image(f"{output_dir}/floor_plan.png")
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": "Floor image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{imageA}" 
                }
            }
        ]

        return content_system, content_user
    
    def get_objects(self, task, area_data, output_dir, room_type, assets_categories=None):
        layout_criteria = task["layout_criteria"]
        room_description = task["task_description"]
        categories_text = format_asset_categories_for_prompt(assets_categories)
        with open("system_prompts/objects.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/get_objects.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                layout_criteria=layout_criteria, 
                room_description=room_description,
                areas=area_data,
                room_type=room_type,
                categories=categories_text,
                )

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        imageA = self.encode_image(f"{output_dir}/floor_plan.png")
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": "Floor image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{imageA}" 
                }
            }
        ]

        return content_system, content_user

    def get_rug(self, task, area_data, output_dir, room_type, floor_objects):
        layout_criteria = task["layout_criteria"]
        room_description = task["task_description"]
        with open("system_prompts/objects.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/get_rug.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                layout_criteria=layout_criteria,
                room_description=room_description,
                areas=area_data,
                room_type=room_type,
                obj=floor_objects,
            )

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            }
        ]
        return content_system, content_user

    def get_small_assets(self, exp_name, description, output_dir):
        with open("system_prompts/small_assets.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/small_assets.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                description=description)
            
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        
        
        imageA = self.encode_image(f"{exp_name}/support_regions_3d_front.png")
        imageB = self.encode_image(f"{exp_name}/support_regions_3d_angled.png")
        render_path = f"{output_dir}/ccea/layout_floor_top.png"
        if not os.path.exists(render_path):
            render_path = f"{output_dir}/ccea/layout_top.png"
        render_image = self.encode_image(render_path)
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": 'Image A:\n'
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{imageA}" 
                }
            },
            {
                "type": "text",
                "text": 'Image B:\n'
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{imageB}" 
                }
            },
            {
                "type": "text",
                "text": "Rendering image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{render_image}"
                }
            }
        ]

        return content_system, content_user

    
    # def define_optim_func(self, wall, obj, task, current_code, areas, output_dir, use_init=True):
    def define_optim_func(
        self,
        wall,
        obj,
        task,
        current_code,
        areas,
        output_dir,
        extra_rules: str = "",
        render_images=None,
        user_prompt_name: str = "constraints",
        system_prompt_name: str = "constraints",
    ):
        layout_criteria = task["layout_criteria"]
        room_description = task["task_description"]
        with open(f"system_prompts/{system_prompt_name}.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()
        with open(f"user_prompts/{user_prompt_name}.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                wall=str(wall),
                obj=str(obj),
                area=str(areas),
                current_code=current_code,
                layout_criteria=layout_criteria,
                room_description=room_description,
                extra_rules=extra_rules,
            )
            
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        imageA = self.encode_image(f"{output_dir}/floor_plan.png")
        
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": "Floor plan: (includes: 1. A door marked in green and windows marked in blue. 2. dark blue fill indicate existing objects)\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{imageA}"
                }
            }
        ]

        name_used = []
        for name in obj:
            name = name.rsplit('_', 1)[0]
            if name in name_used:
                continue
            obj_img = self.encode_image(f"./input_cache_areas/top_down_obj_{name}_text.png")
            content_user.append(
                {
                    "type": "text",
                    "text": f"Top view image with positive X-axis in room bound of {name} is as follows."
                }
            )
            content_user.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{obj_img}"
                    }
                }
            )
            name_used.append(name)

        if render_images:
            for label, path in render_images:
                try:
                    img_b64 = self.encode_image(path)
                except Exception:
                    continue
                content_user.append({
                    "type": "text",
                    "text": f"{label}:\n"
                })
                content_user.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                })
        return content_system, content_user
    
    def get_reflection(
        self,
        task,
        wall,
        obj,
        current_code,
        log_text,
        output_dir=None,
        low_outdegree_nodes=None,
    ):
        layout_criteria = task["layout_criteria"]
        room_description = task["task_description"]
        with open("system_prompts/constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        def _normalize_node_token(x):
            if x is None:
                return None
            return str(x).strip()

        def _is_new_asset(asset_key, asset_val, low_nodes):
            if not low_nodes:
                return False
            candidates = set()
            key_str = _normalize_node_token(asset_key)
            if key_str:
                candidates.add(key_str)
                if "_" in key_str:
                    candidates.add(key_str.rsplit("_", 1)[0])
            if hasattr(asset_val, "id"):
                asset_id = _normalize_node_token(getattr(asset_val, "id", None))
                if asset_id:
                    candidates.add(asset_id)
            elif isinstance(asset_val, dict):
                asset_id = _normalize_node_token(asset_val.get("id"))
                if asset_id:
                    candidates.add(asset_id)
            return any(c in low_nodes for c in candidates)

        low_nodes = set()
        if isinstance(low_outdegree_nodes, (list, tuple, set)):
            for node in low_outdegree_nodes:
                token = _normalize_node_token(node)
                if token:
                    low_nodes.add(token)
        elif low_outdegree_nodes:
            token = _normalize_node_token(low_outdegree_nodes)
            if token:
                low_nodes.add(token)

        obj_existing = {}
        obj_new = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _is_new_asset(k, v, low_nodes):
                    obj_new[k] = v
                else:
                    obj_existing[k] = v
        else:
            obj_existing = obj

        with open("user_prompts/conflict_reflection.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                layout_criteria=layout_criteria,
                room_description=room_description,
                wall=str(wall),
                obj=str(obj),
                obj_existing=str(obj_existing),
                obj_new=str(obj_new),
                current_code=current_code,
                log_text=log_text,
            )

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
        ]

        if output_dir:
            imageA = self.encode_image(f"{output_dir}/floor_plan.png")
            content_user.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{imageA}"
                    }
                }
            )

            name_used = []
            render_path = "./input_cache_areas"
            asset_keys = []
            if isinstance(obj_existing, dict):
                asset_keys.extend(obj_existing.keys())
            if isinstance(obj_new, dict):
                asset_keys.extend(obj_new.keys())
            for name in asset_keys:
                name_base = name.rsplit("_", 1)[0]
                if name_base in name_used:
                    continue
                obj_img_path = os.path.join(render_path, f"top_down_obj_{name_base}_text.png")
                if not os.path.exists(obj_img_path):
                    continue
                obj_img = self.encode_image(obj_img_path)
                content_user.append(
                    {
                        "type": "text",
                        "text": f"Top view image with positive X-axis in room bound of {name_base} is as follows."
                    }
                )
                content_user.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{obj_img}"
                        }
                    }
                )
                name_used.append(name_base)

        return content_system, content_user

    def get_reflection_small(
        self,
        region_bounds,
        obj_list,
        current_code,
        output_dir,
        name,
        low_outdegree_nodes,
        conflict_lines,
        task=None,
        region_info=None,
    ):
        with open("system_prompts/small_constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        if region_info is None:
            region_info = getattr(self, "region_info", None)
        if region_info is None:
            region_info = []

        with open("user_prompts/conflict_reflection_small.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                region_info=region_info,
                current_code=current_code,
                region_bounds=region_bounds,
                obj_list=str(obj_list),
                log_text=(
                    "\n".join(conflict_lines)
                    if isinstance(conflict_lines, (list, tuple))
                    else str(conflict_lines)
                ),
            )

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        support_front_path = f"{output_dir}/support_regions_3d_front.png"
        support_angled_path = f"{output_dir}/support_regions_3d_angled.png"
        output_dir_o = output_dir.rsplit("/", 1)[0]
        render_path = os.path.join(output_dir_o, "layout_top.png")
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            }
        ]
        if os.path.exists(support_front_path):
            imageA = self.encode_image(support_front_path)
            content_user.extend([
                {
                    "type": "text",
                    "text": "Image A: Sideview segmentation diagram."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{imageA}"
                    }
                },
            ])
        else:
            print(f"[GPT][Warning] missing small support image: {support_front_path}")
        if os.path.exists(support_angled_path):
            imageB = self.encode_image(support_angled_path)
            content_user.extend([
                {
                    "type": "text",
                    "text": "Image B: Top-down view segmentation diagram."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{imageB}"
                    }
                },
            ])
        else:
            print(f"[GPT][Warning] missing small support image: {support_angled_path}")
        if os.path.exists(render_path):
            render_image = self.encode_image(render_path)
            content_user.extend([
                {
                    "type": "text",
                    "text": "Global render image:\n"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{render_image}"
                    }
                }
            ])

        return content_system, content_user
    
    def group_constraints(self, obj, task, groups, current_constraints):
        room_description = task["task_description"]
        layout_criteria = task["layout_criteria"]
        with open("system_prompts/group_constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/group_constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(
                room_description=room_description,
                layout_criteria=layout_criteria,
                obj=str(obj),
                groups=str(groups),
                existing_constraints=current_constraints)
            
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            }
        ]

        return content_system, content_user
    
    
    def define_small_optim_func(self, obj, region_data, output_dir):
        with open("system_prompts/small_constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()

        with open("user_prompts/small_constraints.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(obj=obj)

        # obj / region_data should only contain open-surface regions (clearance == 1.0).
        if isinstance(region_data, dict):
            open_region_payload = {
                k: v for k, v in region_data.items()
                if v is not None and k in obj
            }
        else:
            open_region_payload = region_data
    
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        render_path = f"{output_dir}/ccea/layout_floor_top.png"
        if not os.path.exists(render_path):
            render_path = f"{output_dir}/ccea/layout_top.png"
        render_image = self.encode_image(render_path)

        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": 'open_region:\n' + str(open_region_payload)
            },
            {
                "type": "text",
                "text": "global render image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{render_image}"
                }
            }
        ]

        return content_system, content_user
    
    def find_big_object(self, objects_in_areas, task, output_dir):
        with open("system_prompts/big_objects.txt", "r", encoding="utf-8") as f:
            prompting_text_system = f.read()
        with open("user_prompts/find_big_objects.txt", "r", encoding="utf-8") as f:
            prompting_text_user = f.read().format(room_description=task["task_description"])
    
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        render_path = f"{output_dir}/ccea/layout_floor_top.png"
        if not os.path.exists(render_path):
            render_path = f"{output_dir}/ccea/layout_top.png"
        render_image = self.encode_image(render_path)
        content_user = [
            {
                "type": "text",
                "text": prompting_text_user
            },
            {
                "type": "text",
                "text": 'objects_in_areas: \n' + str(objects_in_areas)
            },
            {
                "type": "text",
                "text": "global render image:\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{render_image}"
                }
            }
        ]

        return content_system, content_user
    
    
