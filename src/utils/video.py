"""
functions for processing video
"""
import os
import os.path as osp
import numpy as np
import subprocess
import imageio
import cv2
import json
from rich.progress import track
from .rprint import rprint as print
from .helper import prefix, seconds_to_hms

def exec_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
def images2video(images, wfp, **kwargs):
    fps = kwargs.get('fps', 25)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    n = len(images)
    if image_mode.lower() == 'bgr':
        for i in track(range(n), description='converting', transient=True):
            images[i] = (cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))

    imageio.mimsave(wfp, images, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size)

    # print(f':smiley: Dump to {wfp}\n', style="bold green")
    print(f'Dump to {wfp}\n')


def video2gif(video_fp, fps=30, size=256):
    if osp.exists(video_fp):
        d = osp.split(video_fp)[0]
        fn = prefix(osp.basename(video_fp))
        palette_wfp = osp.join(d, 'palette.png')
        gif_wfp = osp.join(d, f'{fn}.gif')
        # generate the palette
        cmd = f'ffmpeg -i {video_fp} -vf "fps={fps},scale={size}:-1:flags=lanczos,palettegen" {palette_wfp} -y'
        exec_cmd(cmd)
        # use the palette to generate the gif
        cmd = f'ffmpeg -i {video_fp} -i {palette_wfp} -filter_complex "fps={fps},scale={size}:-1:flags=lanczos[x];[x][1:v]paletteuse" {gif_wfp} -y'
        exec_cmd(cmd)
    else:
        print(f'video_fp: {video_fp} not exists!')


def merge_audio_video(video_fp, audio_fp, wfp):
    if osp.exists(video_fp) and osp.exists(audio_fp):
        cmd = f'ffmpeg -i {video_fp} -i {audio_fp} -c:v copy -c:a aac {wfp} -y'
        exec_cmd(cmd)
        print(f'merge {video_fp} and {audio_fp} to {wfp}')
    else:
        print(f'video_fp: {video_fp} or audio_fp: {audio_fp} not exists!')


def blend(img: np.ndarray, mask: np.ndarray, background_color=(255, 255, 255)):
    mask_float = mask.astype(np.float32) / 255.
    background_color = np.array(background_color).reshape([1, 1, 3])
    bg = np.ones_like(img) * background_color
    img = np.clip(mask_float * img + (1 - mask_float) * bg, 0, 255).astype(np.uint8)
    return img


def concat_frames(I_p_lst, driving_rgb_lst, img_rgb):
    # TODO: add more concat style, e.g., left-down corner driving
    out_lst = []
    for idx, _ in track(enumerate(I_p_lst), total=len(I_p_lst), description='Concatenating result...'):
        source_image_drived = I_p_lst[idx]
        image_drive = driving_rgb_lst[idx]

        # resize images to match source_image_drived shape
        h, w, _ = source_image_drived.shape
        image_drive_resized = cv2.resize(image_drive, (w, h))
        img_rgb_resized = cv2.resize(img_rgb, (w, h))

        # concatenate images horizontally
        frame = np.concatenate((image_drive_resized, img_rgb_resized, source_image_drived), axis=1)
        out_lst.append(frame)
    return out_lst


class VideoWriter:
    def __init__(self, **kwargs):
        self.fps = kwargs.get('fps', 30)
        self.wfp = kwargs.get('wfp', 'video.mp4')
        self.video_format = kwargs.get('format', 'mp4')
        self.codec = kwargs.get('codec', 'libx264')
        self.quality = kwargs.get('quality')
        self.pixelformat = kwargs.get('pixelformat', 'yuv420p')
        self.image_mode = kwargs.get('image_mode', 'rgb')
        self.ffmpeg_params = kwargs.get('ffmpeg_params')

        self.writer = imageio.get_writer(
            self.wfp, fps=self.fps, format=self.video_format,
            codec=self.codec, quality=self.quality,
            ffmpeg_params=self.ffmpeg_params, pixelformat=self.pixelformat
        )

    def write(self, image):
        if self.image_mode.lower() == 'bgr':
            self.writer.append_data(image[..., ::-1])
        else:
            self.writer.append_data(image)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def change_video_fps(input_file, output_file, fps=25, codec='libx264', crf=13, video_len_limit=180):
    change_vid_len = get_video_duration(input_file) > video_len_limit and video_len_limit > 0
    if change_vid_len:
        output_file_t = output_file.replace('.mp4', '-temp.mp4')
        cmd = f"ffmpeg -ss 00:00:00 -to {seconds_to_hms(video_len_limit)} -i {input_file} -c:v {codec} {output_file_t} -y -loglevel warning"
        exec_cmd(cmd)
        input_file = output_file_t

    cmd = f"ffmpeg -i {input_file} -c:v {codec} -crf {crf} -r {fps} {output_file} -y -loglevel warning"
    exec_cmd(cmd)

    if change_vid_len: exec_cmd(f'rm {output_file_t}')


def get_fps(filepath):
    import ffmpeg
    probe = ffmpeg.probe(filepath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['avg_frame_rate'])
    return fps


def get_video_duration(file_path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    return duration


def extract_audio(video_fp, audio_fp):
    os.makedirs(os.path.dirname(audio_fp), exist_ok=True)
    extract_wav_cmd = f'ffmpeg -i "{video_fp}" -vn -acodec pcm_s16le -ar 44100 "{audio_fp}" -y -loglevel warning'
    exec_cmd(extract_wav_cmd)
