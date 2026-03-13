import cv2
from pathlib import Path

def get_video_files(base_path: str) -> list[dict]:
    """Scan dataset folder, return list of {path, label, category} dicts."""
    base = Path(base_path)
    dataset = []

    for video_path in (base / "Anomaly-Videos-Part-1").rglob("*.mp4"):
        dataset.append({
            "path": video_path,
            "label": "Anomaly",
            "category": video_path.parent.name
        })

    for video_path in (base / "Normal_Videos_for_Event_Recognition").rglob("*.mp4"):
        dataset.append({
            "path": video_path,
            "label": "Normal",
            "category": "Normal"
        })
    return dataset

def extract_frames(video_path, num_frames: int) -> tuple:
    """
    Sample num_frames evenly from a video file.
    Returns (frames, total_frames, fps, duration_s).
    Returns ([], 0, 0, 0) if the file can't be read.
    """
    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 1
    duration_s   = round(total_frames / fps, 2)

    if total_frames <= 0:
        cap.release()
        return [], 0, 0, 0

    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]

    frames = []
    for frame_id in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames, total_frames, fps, duration_s