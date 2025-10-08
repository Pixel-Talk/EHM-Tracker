export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=0 python -m src.tracking_single_image \
    --input_dir assets/examples/images \
    --output_dir results/example_image  \
    --save_vis_video --save_visual_render