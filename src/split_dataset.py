import os
import random
import argparse
import json,re

def split_train_valid(videos_info, num_valid=5):
    
    def extract_base_id(video_id):
        return re.sub(r'\d', '', video_id)
    base_ids = set(extract_base_id(video_id) for video_id in videos_info.keys())
    valid_base_ids = set(random.sample(list(base_ids), min(num_valid, len(base_ids))))
    train_data = []
    valid_data = []
    
    for video_id, info in videos_info.items():
        frames_key = info['frames_keys']
        base_id = extract_base_id(video_id)
        if base_id in valid_base_ids:
            valid_data.extend([f'{video_id}/{frame}' for frame in frames_key])
        else:
            train_data.extend([f'{video_id}/{frame}' for frame in frames_key])
    print(f"valid_data_length: {len(valid_data)}")
    return {'train': train_data, 'valid': valid_data}

if __name__ == "__main__":
    # Set up argument parsing.
    parser = argparse.ArgumentParser(description="Recursively traverse specified folders and their subfolders, find and process specific files.")
    parser.add_argument('--data_path')
    parser.add_argument('--num_valid', type=int, default=1)
    # Parse the arguments.
    args = parser.parse_args()
    random.seed(10)
    with open(os.path.join(args.data_path, 'videos_info.json'), 'r') as f:
        videos_info = json.load(f)
    dataset_frames=split_train_valid(videos_info,num_valid=args.num_valid)
    dataset_frames_path = os.path.join(args.data_path, 'dataset_frames.json')
    with open(dataset_frames_path, 'w', encoding='utf-8') as json_file:
        json.dump(dataset_frames, json_file, ensure_ascii=False, indent=4)