
import os
import csv
from moviepy.video.io.VideoFileClip import VideoFileClip

# Define the directory containing the video files
directory = "/home/yuningc/Masking-video/EgoVLP/MPII"
output_directory = "/home/yuningc/Masking-video/EgoVLP/MPII_clip"
# Read the CSV file
csv_files = ["/home/yuningc/Masking-video/EgoVLP/MPII_train.csv","/home/yuningc/Masking-video/EgoVLP/MPII_test.csv"]
for csv_file in csv_files:
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            subject, filename, start_frame, end_frame, _, _, _  = row
            
            # Construct the full path to the video file
            filepath = os.path.join(directory, filename+".avi")
            output_filename = f"{filename}_{start_frame}_{end_frame}.avi"
            output_filepath = os.path.join(output_directory, output_filename)
            if os.path.exists(output_filepath):
                continue
            
            # Load the video
            video = VideoFileClip(filepath)
            
            # Convert start_frame and end_frame to integers
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            
            # Clip the segment
            segment = video.subclip(start_frame / video.fps, end_frame / video.fps)
            
            # print(start_frame / video.fps, end_frame / video.fps)
            
            # Save the clipped segment
            segment.write_videofile(output_filepath, codec="libx264")  # Use codec="rawvideo" for .avi files
            # break
'''

import os
import csv

# Define the directory containing the output files
directory = "/home/yuningc/Masking-video/EgoVLP/MPII"

# Read the CSV file
csv_file = "/home/yuningc/Masking-video/EgoVLP/MPII_train.csv"
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        subject, filename, start_frame, end_frame, _, _, _ = row
        
        # Define the output file name with .avi extension
        output_filename = f"{filename}_{start_frame}_{end_frame}.avi"
        output_filepath = os.path.join(directory, output_filename)

        # Check if the output file exists and delete it if necessary
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
'''