from transformers import pipeline
import os
import cv2
from PIL import Image

VIDEO_DIR = "videos"
CAPTION_DIR = "captions"
FRAME_RATE = 1  # frames per second

# Create output directory
os.makedirs(CAPTION_DIR, exist_ok=True)

# Use a small vision-language model
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def extract_frames(video_path, rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps / rate)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)
        count += 1
    cap.release()
    return frames

def caption_video(video_file):
    path = os.path.join(VIDEO_DIR, video_file)
    frames = extract_frames(path, FRAME_RATE)
    captions = []
    for i, frame in enumerate(frames):
        try:
            result = captioner(frame)
            caption = result[0]['generated_text']
            print(f"Caption {i+1}: {caption}")
            captions.append(caption)
        except Exception as e:
            print(f"Error on frame {i+1}: {e}")
    return captions

def main():
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.lower().endswith(('.mp4', '.mov', '.avi')):
            continue
        print(f"Processing {video_file}...")
        captions = caption_video(video_file)
        if not captions:
            print("No captions generated.")
            continue
        out_path = os.path.join(CAPTION_DIR, f"{os.path.splitext(video_file)[0]}.txt")
        with open(out_path, 'w') as f:
            for i, cap in enumerate(captions):
                f.write(f"Frame {i+1}: {cap}\n")
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
