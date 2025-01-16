"""
This file just converts .mp4 file into a .gif file. Do not worry about this.
"""

from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx

if __name__ == "__main__":
    video = VideoFileClip("defend.mp4")
    video = video.fx(vfx.speedx, factor=3)
    video.write_gif("defend.gif")
    video.close()