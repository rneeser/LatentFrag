import argparse
import os
import subprocess

from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--in_dir', action='store', type=str,  required=True)
p.add_argument('--out_dir', action='store', type=str,  required=True)
p.add_argument('--reduce', action='store', type=str,  required=True)
p.add_argument('--starting', action='store', type=str,  required=False, default=None)


def main(in_dir, out_dir, reduce, starting=None):
    os.makedirs(out_dir, exist_ok=True)
    processed_files = set([
        fname
        for fname in os.listdir(out_dir)
        if fname.endswith('.pdb') and fname.count('_') == 1
    ])
    print(f'Found {len(processed_files)} processed files')

    files_to_process = []
    for fname in os.listdir(in_dir):
        if not fname.endswith('.pdb') or fname.count('_') != 1 or fname in processed_files:
            continue
        if starting is not None and not fname.startswith(starting):
            continue
        files_to_process.append(fname)

    print(f'Protonating {len(files_to_process)} chains')
    for fname in tqdm(files_to_process):
        input_path = f'{in_dir}/{fname}'
        output_path = f'{out_dir}/{fname}'
        subprocess.run(f'{reduce} {input_path} > {output_path} 2> /dev/null', shell=True)


if __name__ == '__main__':
    args = p.parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        reduce=args.reduce,
        starting=args.starting,
    )
