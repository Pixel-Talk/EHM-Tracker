import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tyro,time,shlex
import subprocess
from rich.progress import track
from src.utils.rprint import rlog as log
from src.configs.argument_config import ArgumentConfig
from src.data_prepare_pipeline import DataPreparePipeline
from src.configs.data_prepare_config import DataPreparationConfig

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def build_command(args, part_lst, gpu_id):
    cmd_parts = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'python', os.path.basename(os.path.abspath(__file__))
    ]
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None  and arg_name not in["part_lst","visible_gpus","output_dir","is_vfhq","n_divide"]:
            if isinstance(arg_value, bool) :
                if  arg_value:
                    cmd_parts.append(f'--{arg_name}')
            elif isinstance(arg_value, list):
                cmd_parts.append(f'--{arg_name} {" ".join(map(str, arg_value))}')
            else:
                cmd_parts.append(f'--{arg_name} {shlex.quote(str(arg_value))}')
            
    cmd_parts.extend([
        '-p', part_lst,
        '-n', '1',
        '-v', '0',
        '--output_dir', args.output_dir
    ])
    
    return ' '.join(cmd_parts)

def find_videos_in_path(path):
    videos = []
    if os.path.isfile(path):
        if path.split('.')[-1].lower() in ('mp4', 'avi', 'mkv'):
            videos.append(path)
    elif os.path.isdir(path):
        videos.extend(
            [os.path.join(path, f) for f in sorted(os.listdir(path)) 
             if f.split('.')[-1].lower() in ('mp4', 'avi', 'mkv')]
        )
    else:
        print(f"[Warning] Path '{path}' is not a valid file or directory. Skipping.")
        
    return videos

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    all_dirs = []
    all_dirs.extend(find_videos_in_path(args.in_root))
    if len(args.more_in_root)>0:
        for m_in_root in args.more_in_root:
            all_dirs.extend(find_videos_in_path(m_in_root))
            
    visible_gpus = [int(x) for x in args.visible_gpus.split(',') if x.strip() != '']
    if args.part_lst is None or args.part_lst.lower() == 'nan':
        per_num = len(all_dirs) // int(args.n_divide) + 1
        all_procs = []
        counter = 0
        for iii, i in enumerate(range(0, len(all_dirs), per_num)):
            part_lst = f'{i},{i + per_num}'
            gpu_id   = iii % len(visible_gpus)

            cmd=build_command(args, part_lst, visible_gpus[gpu_id])
            log(cmd)
            all_procs.append(subprocess.Popen(cmd, shell=True))
            counter += 1
            if counter % len(visible_gpus) == 0:
                # Sleep 30 seconds after every visible_gpus processes to avoid memory overload during warm-up
                print("start sleeping for 30 seconds.......")
                time.sleep(30)
                print("finish sleeping.......")
        for p in all_procs:
            p.wait()
    else:
        # specify configs for inference
        pdata_cfg = partial_fields(DataPreparationConfig, args.__dict__)

        data_prepare_pipeline = DataPreparePipeline(
            data_prepare_cfg=pdata_cfg,
        )
        part_lst = [int(x) for x in args.part_lst.split(',')]
        for i in range(len(part_lst)):
            if part_lst[i] == 0: part_lst[i] = None
        all_dirs = all_dirs[part_lst[0]: part_lst[1]]

        args.source_dir = all_dirs
        data_prepare_pipeline.execute(args)


if __name__ == '__main__':
    main()
