import argparse
import os
import subprocess
from glob import glob
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument('--in_dir', action='store', type=str,  required=True)
p.add_argument('--out_dir', action='store', type=str,  required=True)
p.add_argument('--reduce', action='store', type=str,  required=True)
p.add_argument('--starting', action='store', type=str,  required=False, default=None)
p.add_argument('--process_crossdocked', action='store_true', default=False)


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


# def protonate_crossdocked(in_dir, out_dir, reduce, starting=None):
#     os.makedirs(out_dir, exist_ok=True)
#     processed_files = glob(f"{out_dir}/*/*rec.pdb")
#     print(f'Found {len(processed_files)} processed files')

#     all_files = glob(f"{in_dir}/*/*rec.pdb")
#     in2out = {}
#     files_to_process = []
#     for fname in all_files:
#         corresponding_fname = fname.replace(in_dir, out_dir)
#         if corresponding_fname in processed_files:
#             continue
#         if starting is not None and not corresponding_fname.startswith(starting):
#             continue
#         files_to_process.append(fname)
#         in2out[fname] = corresponding_fname

#     print(f'Protonating {len(files_to_process)} chains')
#     for fname in tqdm(files_to_process):
#         output_path = in2out[fname]
#         # create the output directory if it doesn't exist
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         subprocess.run(f'{reduce} {fname} > {output_path} 2> /dev/null', shell=True)


def process_file(args):
    fname, output_path, reduce = args
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run(f'{reduce} {fname} > {output_path} 2> /dev/null', shell=True)


def protonate_crossdocked(in_dir, out_dir, reduce, starting=None, num_processes=None):
    os.makedirs(out_dir, exist_ok=True)
    processed_files = set(glob(f"{out_dir}/*/*rec.pdb"))
    print(f'Found {len(processed_files)} processed files')
    all_files = glob(f"{in_dir}/*/*rec.pdb")

    files_to_process = []
    for fname in all_files:
        corresponding_fname = fname.replace(in_dir, out_dir)
        if corresponding_fname in processed_files:
            continue
        if starting is not None and not corresponding_fname.startswith(starting):
            continue
        files_to_process.append((fname, corresponding_fname, reduce))

    print(f'Protonating {len(files_to_process)} chains')

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)))


if __name__ == '__main__':
    args = p.parse_args()
    if not args.process_crossdocked:
        main(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            reduce=args.reduce,
            starting=args.starting,
        )
    else:
        protonate_crossdocked(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            reduce=args.reduce,
            starting=args.starting,
       )
