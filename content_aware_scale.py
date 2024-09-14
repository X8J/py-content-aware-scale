import cv2
import numpy as np
import os
from seam_carving import resize
import time
from multiprocessing import Pool, cpu_count
import argparse

def seam_carve(image, scale_x, scale_y):
    """
    resize image using seam carving based on scaling factors.
    """
    new_width = int(image.shape[1] * scale_x)
    new_height = int(image.shape[0] * scale_y)
    print(f"resizing image to {new_width}x{new_height} ({scale_x*100:.1f}% width, {scale_y*100:.1f}% height)")
    
    carved_image = resize(image, (new_height, new_width))
    
    if carved_image is None or carved_image.size == 0:
        print("error: the carved image is empty or none.")
    
    return carved_image

def process_frame(frame, scale_x, scale_y):
    """
    process a single frame with seam carving.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    carved_frame = seam_carve(frame_rgb, scale_x, scale_y)
    if carved_frame is None:
        return None
    return cv2.cvtColor(carved_frame, cv2.COLOR_RGB2BGR)

def process_batch(frames, scale_x, scale_y):
    """
    process a batch of frames in parallel.
    """
    return [process_frame(frame, scale_x, scale_y) for frame in frames]

def process_video(input_path, output_path, scale_x=1.0, scale_y=1.0, progress_file='progress.txt'):
    """
    process video frame-by-frame and save output.
    """
    if not os.path.exists(input_path):
        print(f"error: video file {input_path} not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"error: couldn't open the video file {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        print("error: couldn't read the first frame of the video.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    carved_frame = seam_carve(frame_rgb, scale_x, scale_y)

    if carved_frame is None:
        print("warning: skipping video processing due to an error with the first frame.")
        return

    new_height, new_width, _ = carved_frame.shape
    print(f"processed frame dimensions: {new_width}x{new_height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ensure output file path is correct
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    batch_size = 32
    frame_buffer = []
    total_processed_frames = 0

    start_time = time.time()
    pool = Pool(cpu_count())

    with open(progress_file, 'w') as progress_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_buffer:
                    processed_frames = pool.starmap(process_frame, [(f, scale_x, scale_y) for f in frame_buffer])
                    for processed_frame in processed_frames:
                        if processed_frame is not None:
                            out.write(processed_frame)
                            total_processed_frames += 1
                break

            frame_buffer.append(frame)
            if len(frame_buffer) >= batch_size:
                processed_frames = pool.starmap(process_frame, [(f, scale_x, scale_y) for f in frame_buffer])
                for processed_frame in processed_frames:
                    if processed_frame is not None:
                        out.write(processed_frame)
                        total_processed_frames += 1
                frame_buffer = []

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress_percentage = (current_frame / total_frames) * 100
            progress_file.write(f"{progress_percentage:.2f}\n")

    pool.close()
    pool.join()

    cap.release()
    out.release()

    end_time = time.time()
    total_time = end_time - start_time
    average_fps = total_processed_frames / total_time if total_time > 0 else 0
    average_spf = total_time / total_processed_frames if total_processed_frames > 0 else 0
    print(f"processing complete in {total_time:.2f} seconds. Average FPS: {average_fps:.2f}, Average SPF: {average_spf:.2f}")

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="content-aware scaling using seam carving.")
    parser.add_argument('input_video', type=str, help='path to the input video file')
    parser.add_argument('output_video', type=str, help='path to save the scaled output video file')
    parser.add_argument('--scale_x', type=float, default=1.0, help='scaling factor for width (default: 1.0)')
    parser.add_argument('--scale_y', type=float, default=1.0, help='scaling factor for height (default: 1.0)')
    parser.add_argument('--progress_file', type=str, default='progress.txt', help='path to save progress file (default: progress.txt)')

    args = parser.parse_args()

    input_video = args.input_video
    output_video = args.output_video
    scale_x = args.scale_x
    scale_y = args.scale_y
    progress_file = args.progress_file

    process_video(input_video, output_video, scale_x, scale_y, progress_file)

if __name__ == "__main__":
    main()
