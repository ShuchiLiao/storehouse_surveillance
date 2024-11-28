from fastapi import FastAPI
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import logging
import paho.mqtt.client as mqtt
import cv2
import asyncio
import json
import time
from model import DetectEvent, DetectPerson, DetectHelmet

# 配置日志记录到文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_log.log", encoding="utf-8"),
    ],
)

# 存储正在使用的摄像头资源和状态
active_camera = {}
camera_stats = {}

executor = ThreadPoolExecutor(max_workers=50)

# 加载模型
model_fall = YOLO('./model/best_fall.pt', verbose=False)
model_fire = YOLO('./model/best_fire.pt', verbose=False)
model_person = YOLO('./model/best_person.pt', verbose=False)
model_helmet = YOLO('./model/best_helmet.pt', verbose=False)
model_smoke = YOLO('./model/best_smoke.pt', verbose=False)

detect_fall = DetectEvent('摔倒','fall', model_fall, cooldown=300)
detect_smoke = DetectEvent('吸烟', 'smoke', model_smoke, cooldown=300)
detect_fire = DetectEvent('火源', 'fire', model_fire, cooldown=300)
detect_solo = DetectPerson('单独作业', 'solo', model_person, cooldown=300)
detect_helmet = DetectHelmet('未戴头盔', 'no-helmet', model_helmet, cooldown=300)

# 设置 MQTT 地址
mqtt_client = mqtt.Client()
mqtt_client.connect("host.docker.internal", 1883, 60)
mqtt_topic = "/ai"
mqtt_msgs = []

async def process_camera_inference(stream_url):
    cap = cv2.VideoCapture(f"{stream_url}?rtsp_transport=tcp")

    if not cap.isOpened():
        logging.error(f"无法打开摄像头 {stream_url}")
        raise Exception(f"无法打开摄像头 {stream_url}")

    active_camera[stream_url] = cap
    camera_stats[stream_url] = {"frame_count": 0, "processed_frame_count": 0}
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_processed_time = time.time()

    while cap.isOpened():
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            logging.warning(f"摄像头 {stream_url} 未能捕获帧")
            break

        # 更新帧计数
        camera_stats[stream_url]["frame_count"] += 1
        cur_time = time.time()

        if cur_time - last_processed_time >= 1:  # 每秒进行一次推理
            last_processed_time = cur_time
            # logging.info(f"摄像头 {stream_url} 第 {camera_stats[stream_url]['frame_count']}帧")
            # logging.info(f"摄像头 {stream_url} 开始推理第 {camera_stats[stream_url]['processed_frame_count'] + 1} 帧")
            camera_stats[stream_url]["processed_frame_count"] += 1

            # alert_msg = ""

            # 判断是否有跌倒：
            frame, mqtt_fall, fall_alert = detect_fall.detect_and_alert(stream_url, frame, 0.85)
            # 判断是否有吸烟：
            frame, mqtt_smoke, smoke_alert = detect_smoke.detect_and_alert(stream_url, frame, 0.7)
            # 判断是否有火源：
            frame, mqtt_fire, fire_alert = detect_fire.detect_and_alert(stream_url, frame, 0.65)
            # 判断是否单独作业：
            frame, mqtt_solo, person_count, solo_alert = detect_solo.detect_and_alert(stream_url, frame, 0.6)
            # 判断是否戴头盔：
            frame, mqtt_helmet, no_helmet_alert = detect_helmet.detect_and_alert(stream_url, frame, 0.6)

            if mqtt_fall:
                mqtt_msgs.append(mqtt_fall)
            if mqtt_smoke:
                mqtt_msgs.append(mqtt_smoke)
            if mqtt_fire:
                mqtt_msgs.append(mqtt_fire)
            if mqtt_solo:
                mqtt_msgs.append(mqtt_solo)
            if mqtt_helmet:
                mqtt_msgs.append(mqtt_helmet)

            # 发布 MQTT 消息
            for msg in mqtt_msgs:
                if isinstance(msg, dict):
                    payload = json.dumps(msg)
                    mqtt_client.publish(mqtt_topic, payload)
                    # logging.info(f"摄像头 {stream_url} 发布 MQTT 消息: {payload}")
                else:
                    mqtt_client.publish(mqtt_topic, str(msg))
            mqtt_msgs.clear()

            # if alert_msg:
            #     logging.info(f"摄像头 {stream_url} 警告: {alert_msg}")

    cap.release()
    del active_camera[stream_url]
    del camera_stats[stream_url]
    logging.info(f"摄像头 {stream_url} 推理任务已结束")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for stream_url, cap in active_camera.items():
        if cap.isOpened():
            cap.release()
            logging.info(f"摄像头 {stream_url} 已释放")
    active_camera.clear()
    camera_stats.clear()
    logging.info("所有摄像头资源已释放")


app = FastAPI(lifespan=lifespan)


@app.get("/camera/{stream_url:path}")
# @app.get("/camera/{stream_url:path}")
async def start_camera_inference(stream_url: str):
    if stream_url in active_camera:
        return {"message": f"摄像头 {stream_url} 识别已在运行"}
    asyncio.create_task(process_camera_inference(stream_url))
    return {"message": f"摄像头 {stream_url} 识别已启动"}


@app.get("/camera/status")
async def camera_status():
    return {"active_cameras": camera_stats}


@app.get("/")
async def root():
    return {"message": "API is running. Use the POST or GET /camera/{stream_url} endpoint to start inference."}
