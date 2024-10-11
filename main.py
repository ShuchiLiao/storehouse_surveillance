from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from model import LiveAlertSystem
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import logging

logging.getLogger().setLevel(logging.WARNING)

# 存储正在使用的摄像头资源
active_camera = {}
models = {}

# 定义线程池
executor = ThreadPoolExecutor(max_workers=20)  # 最大线程数可以根据系统资源调整
# 加载模型
model = YOLO('model/best.pt', verbose=False)

live_alert_system = LiveAlertSystem(model, cooldown=60)

async def process_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception(f"无法打开摄像头 {camera_id}")

    active_camera[camera_id] = cap

    frame_count = 0
    skip_frames = 10

    def stream_video():

        nonlocal frame_count

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames == 0:
                # 每5帧调用模型进行推理
                frame = live_alert_system.predict_and_alert(frame)

            # 编码帧并返回
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        cap.release()

    # 通过线程池处理摄像头推理任务
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, stream_video)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for camera_id, cap in active_camera.items():
        if cap.isOpened():
            cap.release()
            print(f"Camera {camera_id} released")
    print("All camera released")

app = FastAPI(lifespan=lifespan)



@app.get("/")
async def root():
    return {"message": "API is running. Use the camera."}


@app.get("/camera/{camera_id}")
async def camera_stream(camera_id: int):
    # 动态处理不同摄像头的请求
    return StreamingResponse(await process_camera(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


