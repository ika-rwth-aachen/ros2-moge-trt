#!/bin/bash

# Convert MP4 video to GIF using ffmpeg
# ffmpeg -i mogev2_long2.mp4 -ss 4 -frames 15 -vf "fps=2,scale=720:-1:flags=lanczos" -c:v gif mogev2_short2.gif

ffmpeg -i mogev2_long2.mp4 -ss 4 -vf "fps=8,scale=1000:-2" -pix_fmt yuv420p -c:v libx264 -crf 23 -vframes 150 mogev2_short2.mp4
