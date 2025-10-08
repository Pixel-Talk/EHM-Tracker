from . import DWposeDetector, draw_pose
import argparse
import tqdm
import os
import cv2
import json

def convert_json(pose_dict):
    return pose_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract dw pose info from image dir')
    parser.add_argument('--image_dir',  type=str, default='',  help='Input image dir')
    parser.add_argument('--write_json', type=str, default='',  help='Output json dir')
    parser.add_argument('--display',    type=str, default='0', help='Display result or not')
    parser.add_argument('--model_pose', type=str, default='',  help='Output pose mode')
    parser.add_argument('--face', action='store_true',  help='Output pose with face')
    parser.add_argument('--hand', action='store_true',  help='Output pose with hand')
    parser.add_argument('--number_people_max',   type=int, default=1, help='Max peple in one flame')
    parser.add_argument('--net_resolution',      type=str, default='320x176')
    parser.add_argument('--face_net_resolution', type=str, default='200x200')
    parser.add_argument('--write_video',     type=str, default='', help='Output video path')
    parser.add_argument('--write_video_fps', type=str, default='30')
    parser.add_argument('--write_images',    type=str, default='',  help='Output images dir')
    parser.add_argument('--render_pose',     type=str, default='0', help='Render pose on the original image or not')

    args = parser.parse_args()

    dwpose = DWposeDetector()

    all_names = sorted(os.listdir(args.image_dir))

    os.makedirs(args.write_json, exist_ok=True)

    for img_name in tqdm(all_names):
        img_fp  = os.path.join(args.image_dir, img_name)
        json_fp = os.path.join(args.write_json, img_name.split('.')[0] + '.json')

        img = cv2.imread(img_fp)

        pose = dwpose(img)
        pose = convert_json(pose)

        with open(json_fp, 'w') as fid:
            json.dump(pose, fid)







