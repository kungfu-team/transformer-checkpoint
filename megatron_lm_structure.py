import collections
import json
import os

import torch

from key_order import gen_key_order
from rank_map import gen_rank_map


def create_value_dict(value):
    if value is None:
        return None
    if isinstance(value, (collections.OrderedDict, dict)):
        elements = {}
        for key, val in value.items():
            elements[key] = create_value_dict(val)
        return elements
    if isinstance(value, torch.Tensor):
        return {"tensor": list(value.size())}
    if isinstance(value, (list, tuple)):
        elements = []
        for val in value:
            elements.append(create_value_dict(val))
        return elements
    if isinstance(value, (int, float, str, bool)):
        return value

    return "NotPrimitiveNorTensor"


def create_ckpt_dict(ckpt):
    elements = {}

    for key, val in ckpt.items():
        elements[key] = create_value_dict(val)

    return elements


def gen_structure(
    job_id: str,
    base_dir: str,
    pp_size: int,
    tp_size: int,
    dp_size: int,
    step: int,
    model: str,
    model_size: str,
    precision: str,
):
    out_dir = "megatron-lm"
    out_dir = os.path.join(out_dir, precision)
    out_dir = os.path.join(out_dir, f"{model}/{model_size}")
    out_dir = os.path.join(out_dir, f"pp{pp_size:02d}/mp{tp_size:02d}/dp{dp_size:02d}")
    os.makedirs(out_dir, exist_ok=True)
    size = pp_size * tp_size * dp_size
    gpus_container = 4
    num_containers = size // gpus_container

    for container in range(num_containers):
        container_path = os.path.join(base_dir, f"{container}/ckpt")
        for entry in os.scandir(container_path):
            if not entry.is_dir() or entry.name == "tensorboard":
                continue
            rank = int(entry.name)
            print(f"rank {rank}")
            rank_input_path = os.path.join(entry.path, f"iter_{step:07d}")

            if not os.path.isdir(rank_input_path):
                continue

            rank_output_path = os.path.join(out_dir, f"rank{rank:02d}")
            os.makedirs(rank_output_path)

            for entry in os.scandir(rank_input_path):
                if not entry.is_dir():
                    continue

                ckpt_path = os.path.join(entry.path, "model_optim_rng.pt")

                ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
                ckpt_dict = create_ckpt_dict(ckpt)
                new_name = entry.name.split(".")[0] + ".json"
                out_path = os.path.join(rank_output_path, new_name)

                with open(out_path, "w", encoding="utf-8") as dict_file:
                    json.dump(ckpt_dict, dict_file, indent=4)


def main():
    job_id = "gen-para-config"
    base_dir = os.path.join(
        os.path.expanduser("~"), f"Tenplex/repo/tenplex/benchmark/training/{job_id}"
    )
    pp_size = 1
    tp_size = 2
    dp_size = 2
    step = 50
    model = "gpt"
    model_size = "large"
    precision = "fp32"
    gen_structure(
        job_id, base_dir, pp_size, tp_size, dp_size, step, model, model_size, precision
    )

    framework = "megatron-lm"
    gen_key_order(framework, model, model_size, precision, pp_size, tp_size, dp_size)

    gen_rank_map(
        framework, model, model_size, precision, pp_size, tp_size, dp_size, base_dir
    )


if __name__ == "__main__":
    main()