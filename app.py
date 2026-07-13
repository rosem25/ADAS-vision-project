"""
ADAS Lite - FastAPI Streaming Backend
=====================================
Streams processed video frames (lane detection + object detection + warnings)
live to the browser as MJPEG, avoiding the mp4v codec playback issues that
break in VS Code / most browsers with plain saved .mp4 files.

Run with: uvicorn main:app --reload
Then open: http://localhost:8000
"""
import os
import uuid
import shutil
import tempfile
import time

import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import csv

from adas_pipeline import detect_lanes, detect_objects_yolo, detect_objects_fallback, check_warning, annotate_frame, USE_YOLO

app = FastAPI(title="ADAS Lite")
templates = Jinja2Templates(directory="templates")

# In-memory store: video_id -> {path, log}
SESSIONS = {}
UPLOAD_DIR = tempfile.mkdtemp(prefix="adas_uploads_")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"using_yolo": USE_YOLO})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    SESSIONS[video_id] = {"path": save_path, "log": [], "conf": 0.5}
    return {"video_id": video_id}

@app.post("/set_conf/{video_id}")
async def set_conf(video_id: str, request: Request):
    data = await request.json()
    if video_id in SESSIONS and "conf" in data:
        SESSIONS[video_id]["conf"] = float(data["conf"])
    return {"status": "ok"}


def frame_generator(video_id: str):
    session = SESSIONS[video_id]
    cap = cv2.VideoCapture(session["path"])
    frame_idx = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(UPLOAD_DIR, f"{video_id}_annotated.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    session["out_path"] = out_path

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        annotated, lane_count, lane_departure = detect_lanes(frame)

        conf_threshold = session.get("conf", 0.5)
        if USE_YOLO:
            detections = detect_objects_yolo(frame, conf_threshold=conf_threshold)
        else:
            detections = detect_objects_fallback(frame, conf_threshold=conf_threshold)

        warning, _ = check_warning(detections, frame.shape)
        
        fps_value = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        annotate_frame(annotated, detections, warning, fps_value, lane_departure)

        session["log"].append({
            "frame": frame_idx,
            "lane_lines_detected": lane_count,
            "num_objects": len(detections),
            "object_labels": ";".join(d["label"] for d in detections),
            "warning_triggered": bool(warning),
            "lane_departure": bool(lane_departure),
            "fps": fps_value,
        })
        
        out.write(annotated)

        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        frame_idx += 1

    cap.release()
    out.release()
    session["done"] = True


@app.get("/stream/{video_id}")
def stream(video_id: str):
    if video_id not in SESSIONS:
        return JSONResponse({"error": "unknown video_id"}, status_code=404)
    return StreamingResponse(
        frame_generator(video_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/log/{video_id}")
def get_log(video_id: str):
    if video_id not in SESSIONS:
        return JSONResponse({"error": "unknown video_id"}, status_code=404)
    session = SESSIONS[video_id]
    log = session["log"]
    total_warnings = sum(1 for row in log if row["warning_triggered"])
    total_lane_departures = sum(1 for row in log if row.get("lane_departure"))
    avg_objects = (sum(row["num_objects"] for row in log) / len(log)) if log else 0
    avg_fps = (sum(row.get("fps", 0) for row in log) / len(log)) if log else 0
    return {
        "frames_processed": len(log),
        "total_warnings": total_warnings,
        "total_lane_departures": total_lane_departures,
        "avg_objects_per_frame": round(avg_objects, 2),
        "avg_fps": round(avg_fps, 1),
        "done": session.get("done", False),
        "rows": log[-50:],  # last 50 rows to keep payload small
    }


@app.get("/download/video/{video_id}")
def download_video(video_id: str):
    session = SESSIONS.get(video_id)
    if not session or not session.get("done") or not os.path.exists(session.get("out_path", "")):
        return JSONResponse({"error": "not found or not done"}, status_code=404)
    return FileResponse(session["out_path"], media_type="video/mp4", filename="annotated.mp4")


@app.get("/download/csv/{video_id}")
def download_csv(video_id: str):
    session = SESSIONS.get(video_id)
    if not session or not session.get("done"):
        return JSONResponse({"error": "not found or not done"}, status_code=404)
    
    output = io.StringIO()
    log = session["log"]
    if log:
        writer = csv.DictWriter(output, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detections_log.csv"}
    )


@app.get("/health")
def health():
    return {"status": "ok", "using_yolo": USE_YOLO}