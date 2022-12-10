import json
import os
import re


def main():
    framework = 'megatron-lm'
    pp_size = 3
    mp_size = 2
    dp_size = 2
    total_size = pp_size * mp_size * dp_size
    direc = f'{framework}/fp16/seq_1024/pp{pp_size:02d}/mp{mp_size:02d}/dp{dp_size:02d}'

    mapping = dict()

    for rank in range(total_size):
        rank_dir = os.path.join(direc, f'rank{rank:02d}')
        if not os.path.exists(rank_dir):
            continue

        layer_numbers = []
        pp_rank = None
        for entry in os.scandir(rank_dir):
            #  print(f"entry name {entry.name}")
            pattern = r'mp_rank_(\d+)_(\d+).json'
            mat = re.match(pattern, entry.name)
            if mat is None:
                raise ValueError("Match is None")
            mp_rank = int(mat.group(1))
            pp_rank = int(mat.group(2))

            with open(entry.path, "r") as json_f:
                ckpt = json.load(json_f)

            encoder = ckpt["model"]["language_model"]["encoder"]
            keys = encoder.keys()

            layers = set()
            for key in keys:
                #  print(f"layer key {key}")
                pattern = r"layers\.(\d+)\.(.*)"
                mat = re.match(pattern, key)
                if mat is None:
                    print(f"NO match for {key}")
                    continue
                    #  raise ValueError("Match is None")
                layer_num = int(mat.group(1))
                layers.add(layer_num)

            mapping[pp_rank] = list(layers)

    import pprint
    pprint.pprint(mapping)

    #  with open(f'{direc}/layer_map.json', 'w') as json_file:
    #      json.dump(mapping, json_file, indent=4)


if __name__ == "__main__":
    main()
