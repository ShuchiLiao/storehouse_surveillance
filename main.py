from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from model import LiveAlertSystem
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import logging
import paho.mqtt.client as mqtt
from utils import put_chinese_text
import json

logging.getLogger().setLevel(logging.WARNING)

# 存储正在使用的摄像头资源
active_camera = {}
models = {}

# 定义线程池
executor = ThreadPoolExecutor(max_workers=20)  # 最大线程数可以根据系统资源调整
# 加载模型
model = YOLO('model/best.pt', verbose=False)

live_alert_system = LiveAlertSystem(model, cooldown=60)

# 设置mqtt 地址
mqtt_client = mqtt.Client()
mqtt_client.connect("host.docker.internal", 1883, 60)
# mqtt_client.connect("127.0.0.1", 1883, 60)
mqtt_topic = "/ai"

async def process_camera(stream_url):
    cap = cv2.VideoCapture(f"{stream_url}?rtsp_transport=tcp")
    # # 设置帧率为5fps
    # cap.set(cv2.CAP_PROP_FPS, 2)  # 将帧率设置为 5
    # # 验证帧率设置是否成功
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Current FPS: {fps}")

    if not cap.isOpened():
        raise Exception(f"无法打开摄像头 {stream_url}")

    active_camera[stream_url] = cap

    frame_count = 0
    skip_frames = 10

    last_person_count = 0
    last_alert_msg = ""

    def stream_video():

        nonlocal frame_count, last_person_count, last_alert_msg, stream_url

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames == 0:
                # 每x帧调用模型进行推理
                frame, person_count, alert_msg, mqtt_msgs = live_alert_system.predict_and_alert(stream_url,frame)
                last_person_count = person_count
                if alert_msg:
                    last_alert_msg = " ".join(sorted(alert_msg))
                    alert_msg = " ".join(sorted(alert_msg))
                else:
                    last_alert_msg = ""
                    alert_msg = ""
                # print(person_count)
                # print(alert_msg)
                # 在帧上显示检测到的人员
                if mqtt_msgs:
                    for msg in mqtt_msgs:
                        if isinstance(msg, dict):
                            # 将字典序列化为 JSON 字符串
                            payload = json.dumps(msg)
                            print(f"Publishing to MQTT: {payload}")
                            mqtt_client.publish(mqtt_topic, payload)
                        else:
                            # 如果不是字典，直接转换为字符串并发布
                            mqtt_client.publish(mqtt_topic, str(msg))
            else:
                person_count = last_person_count
                alert_msg = last_alert_msg

            frame = put_chinese_text(frame, f"人员数量: {person_count}", (10, 30),
                                     font_path='Fonts/simhei.ttf', font_size=25, color=(0, 255, 0))

            # 如果有警告，显示警告信息
            frame = put_chinese_text(frame, f"警告: {alert_msg}", (10, 70),
                                     font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))
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
    for stream_url, cap in active_camera.items():
        if cap.isOpened():
            cap.release()
            print(f"Stream {stream_url} released")
    print("All Stream released")

app = FastAPI(lifespan=lifespan)



@app.get("/")
async def root():
    return {"message": "API is running. Use the stream URL."}


@app.get("/camera/{stream_url:path}")
async def camera_stream(stream_url: str):
    # 动态处理不同摄像头的请求
    return StreamingResponse(await process_camera(stream_url), media_type="multipart/x-mixed-replace; boundary=frame")


