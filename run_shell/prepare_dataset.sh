export PYTHONPATH='.'
# Tracking with your own video dataset
# n means number of total threads
# v means which gpu to use, e.g. 0,1,2,3
python tracking_video.py \
 --in_root /path/to/video_dataset \
  --output_dir  /path/to/tracked_video \
 --check_hand_score 0.45 \
 --tracking_with_interval \
 -n 8 -v 0,1,2,3

#combine all dataset
python -m src.build_lmdb_dataset \
 --data_folders /path/to/tracked_video \
 --save_path /path/to/combined_dataset

# python -m src.build_lmdb_dataset \
#  --data_folders results/example_video \
#  --save_path results/example_video_dataset

#split dataset to train and val
#num_valid: number of videos for validation
python -m src.split_dataset \
 --data_path /path/to/combined_dataset \
 --num_valid 100

# python -m src.split_dataset \
# --data_path  results/example_video_dataset \
# --num_valid 1