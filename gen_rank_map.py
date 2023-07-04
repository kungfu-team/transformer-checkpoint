import json
import os
import re


def main():
    framework = 'megatron-lm'
    model = 'bert'
    model_size = 'base'
    pp_size = 2
    mp_size = 2
    dp_size = 1
    total_size = pp_size * mp_size * dp_size
    direc = f'{framework}/{model}/{model_size}/pp{pp_size:02d}/mp{mp_size:02d}/dp{dp_size:02d}'

    mapping = dict()

    for rank in range(total_size):
        rank_dir = os.path.join(direc, f'rank{rank:02d}')
        if not os.path.exists(rank_dir):
            mapping[rank] = {'mp_rank': 0, 'pp_rank': 0, 'dp_rank': 0}
            continue
        for entry in os.scandir(rank_dir):
            if pp_size > 1:
                pattern = r'mp_rank_(\d+)_(\d+)'
            else:
                pattern = r'mp_rank_(\d+)'
            ma = re.match(pattern, entry.name)
            if ma is None:
                continue
                #  raise ValueError("Match is None")
            mp_rank = int(ma.group(1))
            if pp_size > 1:
                pp_rank = int(ma.group(2))
                mapping[rank] = {
                    'mp_rank': mp_rank,
                    'pp_rank': pp_rank,
                    'dp_rank': 0
                }
            else:
                mapping[rank] = {
                    'mp_rank': mp_rank,
                    'pp_rank': 0,
                    'dp_rank': 0
                }
            break

    with open(f'{direc}/rank_map.json', 'w') as json_file:
        json.dump(mapping, json_file, indent=4)

    print("You must set the DP ranks manually!!!")


if __name__ == "__main__":
    main()
