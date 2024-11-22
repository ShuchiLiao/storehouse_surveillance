import os
from datetime import datetime, timedelta
import cv2
from sentry_sdk.profiler import frame_id
from utils import put_chinese_text


class DetectEvent:
    def __init__(self, event, cls, model, cooldown=60):
        self.event = event
        self.cls = cls
        self.model = model
        self.cooldown = cooldown

        self.event_status = False
        self.event_screen_time = datetime.now()

    def detect_and_alert(self, stream_url, frame, confidence):
        results = self.model(frame, conf=confidence, verbose=False)
        # results is a list=[result], each result stands for result for each cls
        mqtt_msg = None

        for result in results:
            # if the class is not detected, result.boxes is None, otherwise, boxes is a list[box] of
            # detected objects.
            if result.boxes:
                for box in result.boxes:
                    xyxy = box.xyxy.tolist()[0]

                        # 绘制边界框和标签
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    # cv2.putText(frame, f"{event}", (int(xyxy[0]), int(xyxy[1]) - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    put_chinese_text(frame, f"{self.event.strip()}", (int(xyxy[0]), int(xyxy[1]) - 10),
                             font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))

                mqtt_msg = self.alert_and_screenshot(stream_url, frame)

        self.reset_event_status()

        return frame, mqtt_msg

    def alert_and_screenshot(self, stream_url, frame):
        if not self.event_status:
            # 生成截图文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_name = f"screenshots/{self.cls}_{timestamp}.jpg"
            self.event_status = True
            # 更新截图时间
            self.event_screen_time = datetime.now()
            # 保存截图
            cv2.imwrite(screenshot_name, frame)
            # mqtt_信息
            mqtt_msg = {
                "stream":stream_url,
                "time": timestamp,
                "eventType":self.event,
                "screenshot":screenshot_name
            }
            return mqtt_msg

    def reset_event_status(self):
        """重置事件状态，避免频繁保存相同事件的截图。"""
        current_time = datetime.now()
        if self.event_status:  # 事件已处理
            time_since_last_screenshot = current_time - self.event_screen_time
            if time_since_last_screenshot > timedelta(seconds=self.cooldown):
                self.event_status = False  # 重置状态



class DetectPerson(DetectEvent):
    def detect_and_alert(self, stream_url, frame, confidence):
        results = self.model(frame, conf=confidence, verbose=False)
        mqtt_msg = None
        person_count = 0

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    xyxy = box.xyxy.tolist()[0]

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                    put_chinese_text(frame, "工作人员", (int(xyxy[0]), int(xyxy[1]) - 10),
                             font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))

                    person_count += 1

            if person_count == 1:
                mqtt_msg = self.alert_and_screenshot(stream_url, frame)

        self.reset_event_status()

        return frame, mqtt_msg, person_count

class DetectHelmet(DetectEvent):
    def detect_and_alert(self, stream_url, frame, confidence):
        results = self.model(frame, conf=confidence, verbose=False)
        mqtt_msg = None
        no_helmet = False

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    cls_idx = int(box.cls)
                    if cls_idx == 1:
                        no_helmet = True
                        xyxy = box.xyxy.tolist()[0]
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        # cv2.putText(frame, f"{event}", (int(xyxy[0]), int(xyxy[1]) - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        put_chinese_text(frame, f"{self.event.strip()}", (int(xyxy[0]), int(xyxy[1]) - 10),
                                 font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))

            if no_helmet:
                mqtt_msg = self.alert_and_screenshot(stream_url, frame)

        self.reset_event_status()

        return frame, mqtt_msg



