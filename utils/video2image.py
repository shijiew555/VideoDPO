import os
import moviepy.editor as mp
import imageio

def video2gif(video_path, gif_path, fps=10):
    """
    Convert video to gif
    :param video_path: video file path
    :param gif_path: gif file path
    :param fps: frame per second
    :return: None
    """
    clip = mp.VideoFileClip(video_path)
    clip.write_gif(gif_path, fps=fps)
    

def main(root_folder):
    """
    Convert all videos in the root folder to gifs and save them in the output folder
    :param root_folder: root folder containing videos
    :param output_folder: output folder to save gifs
    :return: None
    """
    # use os.walk 
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                gif_path = os.path.join(root, file.replace('.mp4', '.gif'))
                print(f"Converting {video_path} to {gif_path}")
                video2gif(video_path, gif_path)

if __name__ == '__main__':
    root_folder = 'assets/'
    main(root_folder)