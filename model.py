import os
from datetime import datetime, timedelta
import cv2
from utils import put_chinese_text



class LiveAlertSystem:
    def __init__(self, model, cooldown=10):
        self.model = model
        self.cooldown = cooldown

        #events and timestamps

        self.event_status = {
            "solo": False,
            "no_helmet": False,
            "fire": False,
            "fall": False,
            "smoking": False,
        }

        self.event_screenshot_time = {
            "solo": datetime.now(),
            "no_helmet": datetime.now(),
            "fire": datetime.now(),
            "fall": datetime.now(),
            "smoking": datetime.now()
        }

        self.alert_message = ""

        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')

    def predict_and_alert(self, frame):
        # 使用模型进行推断
        results = self.model(frame)
        person_count = 0
        alert = False
        event = ""
        cls = ""
        alert_count = 0

        # 遍历检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy.tolist()[0]
                class_id = int(box.cls.tolist()[0])
                # confidence = box.conf.tolist()[0]

               # ID-class: {0:fall, 1:fire, 2:helmet, 3:no-helmet, 4:person, 5:smoking}
                if class_id == 4 or class_id == 2:
                    cls = "工作人员"
                    person_count += 1
                    alert = False
                else:
                    alert = True
                    alert_count += 1

                if class_id == 5: # 检测到吸烟
                    cls = " 吸烟 "
                    event = "smoking"

                elif class_id == 3:  # 未戴头盔
                    cls = " 未戴头盔 "
                    event = "no_helmet"

                elif class_id == 1:  # 明火
                    cls = " 明火 "
                    event = "fire"

                elif class_id == 0:  # 摔倒
                    cls = " 摔倒 "
                    event = "fall"

                    # 绘制边界框和标签
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                # cv2.putText(frame, f"{event}", (int(xyxy[0]), int(xyxy[1]) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                print(f"Text position: {(int(xyxy[0]), int(xyxy[1]) - 10)}")
                put_chinese_text(frame, f"{cls.strip()}", (int(xyxy[0]), int(xyxy[1]) - 10),
                         font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))
                if alert:
                    self.alert(event, cls)
                    self.save_screenshot(frame, event)


        # 在帧上显示检测到的人员数量
        frame = put_chinese_text(frame, f"人员数量: {person_count}", (10, 30),
                                 font_path='Fonts/simhei.ttf', font_size=25, color=(0, 255, 0))

        if person_count == 1:
            alert_count += 1
            self.alert("solo", "单独作业")
            self.save_screenshot(frame, "solo")


        # 如果有警告，显示警告信息
        if alert_count > 0:
            frame = put_chinese_text(frame, f"警告: {self.alert_message}", (10, 70),
                                     font_path='Fonts/simhei.ttf', font_size=25, color=(255, 0, 0))

        # 重置事件状态（根据冷却时间）
        self.reset_event_status()

        return frame

    def save_screenshot(self, frame, event):
        # 生成截图文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshots/{event}_{timestamp}.jpg"
        # 更新截图时间
        self.event_screenshot_time[event] = datetime.now()
        # 保存截图
        cv2.imwrite(filename, frame)
        # print(f"截图已保存: {filename}")

    # def alert_and_screenshot(self, frame, event, msg):
    #     if not self.event_status[event]:
    #         self.alert_message += msg
    #         self.save_screenshot(frame, event)
    #         self.event_status[event] = True
    def alert(self,event, msg):
        print(event)
        if not self.event_status[event]:
            self.alert_message += msg
            self.event_status[event] = True

    def reset_event_status(self):
        """重置事件状态，避免频繁保存相同事件的截图。"""
        current_time = datetime.now()
        for event, status in self.event_status.items():
            if status:  # 事件已处理
                time_since_last_screenshot = current_time - self.event_screenshot_time[event]
                if time_since_last_screenshot > timedelta(seconds=self.cooldown):
                    self.event_status[event] = False  # 重置状态
                    self.event_screenshot_time[event] = current_time  # 更新截图时间
                    self.alert_message = "" # 重置警告



