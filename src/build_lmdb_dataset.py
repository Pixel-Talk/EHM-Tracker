
"""

Integrate all tracked data under folders into a dataset

"""


import os,sys
c_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(c_dir)
sys.path.append(os.path.dirname(c_dir))
import argparse,json
from tqdm import tqdm
from src.utils.lmdb import LMDBEngine
from src.utils.io import write_dict_pkl, load_dict_pkl
from split_dataset import split_train_valid


def build_lmdb_dataset(data_folders,save_path):
    out_lmdb_dir = os.path.join(save_path, 'img_lmdb')
    all_optim_track_smplx_dir=os.path.join(save_path, 'optim_tracking_ehm.pkl')
    all_id_share_params_dir=os.path.join(save_path, 'id_share_params.pkl')
    os.makedirs(out_lmdb_dir,exist_ok=True)
    
    all_lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
    videos_info={}
    all_optim_track_fp_smplx={}
    all_id_share_params={}
    for data_folder in tqdm(data_folders,desc="Traverse each folder"):
        for video_id in tqdm(os.listdir(data_folder),desc="Traverse each tracked video data"):
            
            optim_track_fp_smplx = os.path.join(data_folder,video_id, 'optim_tracking_ehm.pkl')
            
            if os.path.exists(optim_track_fp_smplx):
                t_img_lmdb_path=os.path.join(data_folder,video_id, 'img_lmdb')
                t_id_params_path=os.path.join(data_folder,video_id, 'id_share_params.pkl')
                
                t_optim_track_fp_smplx=load_dict_pkl(optim_track_fp_smplx)
                t_id_share_params=load_dict_pkl(t_id_params_path)
                t_lmdb_engine=LMDBEngine(t_img_lmdb_path)
                
                frames_key=list(t_optim_track_fp_smplx.keys())
                videos_info[video_id]={"frames_num":len(frames_key),"frames_keys":frames_key}
                all_optim_track_fp_smplx[video_id]={}
                all_id_share_params[video_id]=t_id_share_params
                for frame_id in frames_key:
                    all_lmdb_engine.dump(f'{video_id}/{frame_id}/body_image', payload=t_lmdb_engine[f'{frame_id}/body_image'], type='image')
                    all_lmdb_engine.dump(f'{video_id}/{frame_id}/body_mask', payload=t_lmdb_engine[f'{frame_id}/body_mask'], type='image')
                    all_optim_track_fp_smplx[video_id][frame_id]={'flame_coeffs':t_optim_track_fp_smplx[frame_id]['flame_coeffs'],
                        'smplx_coeffs':t_optim_track_fp_smplx[frame_id]['smplx_coeffs'],
                        'body_crop':t_optim_track_fp_smplx[frame_id]['body_crop'],
                        'head_crop':t_optim_track_fp_smplx[frame_id]['head_crop'],
                        'left_hand_crop':t_optim_track_fp_smplx[frame_id]['left_hand_crop'],
                        'right_hand_crop':t_optim_track_fp_smplx[frame_id]['right_hand_crop'],}
                    
                t_lmdb_engine.close()
    all_lmdb_engine.close()
    write_dict_pkl(all_id_share_params_dir, all_id_share_params)
    write_dict_pkl(all_optim_track_smplx_dir, all_optim_track_fp_smplx)
    videos_info_path = os.path.join(save_path, 'videos_info.json')
    
    with open(videos_info_path, 'w', encoding='utf-8') as json_file:
        json.dump(videos_info, json_file, ensure_ascii=False, indent=4)
    print("Finish building Dataset !")
    
    dataset_frames=split_train_valid(videos_info,num_valid=1)
    dataset_frames_path = os.path.join(save_path, 'dataset_frames.json')
    with open(dataset_frames_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset_frames, json_file, ensure_ascii=False, indent=4)

def save_crop_image(t_optim_track_fp_smplx,image):
    import numpy as np
    import torchvision.utils as vutils
    head_crop_size=512
    hand_crop_size=512
    head_box_o=np.array([[0.0,0.0,1.0],[head_crop_size,0.0,1.0],[0.0,head_crop_size,1.0],[head_crop_size,head_crop_size,1.0]])#x,y
    hand_box_o=np.array([[0.0,0.0,1.0],[hand_crop_size,0.0,1.0],[0.0,hand_crop_size,1.0],[hand_crop_size,hand_crop_size,1.0]])#x,y
    
    body_crop=t_optim_track_fp_smplx['body_crop']
    head_crop=t_optim_track_fp_smplx['head_crop']
    left_hand_crop=t_optim_track_fp_smplx['left_hand_crop']
    right_hand_crop=t_optim_track_fp_smplx['right_hand_crop']
    
    head_box=body_crop['M_o2c-hd']@head_crop['M_c2o']@head_box_o[:,:,None]
    left_hand_box=body_crop['M_o2c-hd']@left_hand_crop['M_c2o']@hand_box_o[:,:,None]
    right_hand_box=body_crop['M_o2c-hd']@right_hand_crop['M_c2o']@hand_box_o[:,:,None]
    
    head_left,head_right=int(head_box.min(axis=0)[0]),int(head_box.max(axis=0)[0])
    head_top,head_bottom=int(head_box.min(axis=0)[1]),int(head_box.max(axis=0)[1])
    
    left_hand_left,left_hand_right=int(left_hand_box.min(axis=0)[0]),int(left_hand_box.max(axis=0)[0])
    left_hand_top,left_hand_bottom=int(left_hand_box.min(axis=0)[1]),int(left_hand_box.max(axis=0)[1])
    right_hand_left,right_hand_right=int(right_hand_box.min(axis=0)[0]),int(right_hand_box.max(axis=0)[0])
    right_hand_top,right_hand_bottom=int(right_hand_box.min(axis=0)[1]),int(right_hand_box.max(axis=0)[1])
    
    head_image=image[:,head_top:head_bottom,head_left:head_right]/255
    left_hand_image=image[:,left_hand_top:left_hand_bottom,left_hand_left:left_hand_right]/255
    right_hand_image=image[:,right_hand_top:right_hand_bottom,right_hand_left:right_hand_right]/255
    
    vutils.save_image(head_image.float(), "z_img_temp/head.jpg")
    vutils.save_image(left_hand_image.float(), "z_img_temp/left_hand.jpg")
    vutils.save_image(right_hand_image.float(), "z_img_temp/right_hand.jpg")
    pass

if __name__ == "__main__":
    # Set up argument parsing.
    parser = argparse.ArgumentParser(description="Recursively traverse specified folders and their subfolders, find and process specific files.")
    
    # Add an argument for folder paths. nargs='+' means that at least one folder path must be provided.
    parser.add_argument('--data_folders', nargs='+',default=[], help="A list of data folder paths to traverse")
    parser.add_argument('--save_path')
    # Parse the arguments.
    args = parser.parse_args()
    build_lmdb_dataset(args.data_folders,args.save_path)