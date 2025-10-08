export PYTHONPATH='.'
python tracking_video.py \
    --in_root assets/examples/videos \
    --output_dir results/example_video  \
    --save_vis_video --save_images \
    --check_hand_score 0.0 -n 1 -v 0