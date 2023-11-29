from pathlib import Path
from PIL import Image, ImageTk
import torch
import tkinter
import cv2
import threading
import numpy as np
import time
from tkinter import font
from tkinter import ttk
import speech_recognition as sr
from gtts import gTTS
import os
import playsound


# YOLO 모델 로드 
model = torch.hub.load('ultralytics/yolov5','custom',path='best.pt') #custom 모델 사용

# 전역 변수로 선언
save_img_event = threading.Event()
saved_image_path = "screenshot.jpg"
# 카메라 처리
def display_camera_feed():
    cap = cv2.VideoCapture(0)  # 카메라 번호 0을 사용 (첫 번째 카메라)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        # 이미지 리사이즈
        img = cv2.resize(frame, (500, 450))

        # OpenCV 이미지를 Tkinter PhotoImage로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)

        # 라벨 업데이트
        camera_label.config(image=img_tk, anchor='w')
        camera_label.image = img_tk

        if save_img_event.is_set():
            cv2.imwrite(saved_image_path, frame)
            update_saved_image()
            save_img_event.clear()  # 이벤트 리셋

        # # 'q' 키를 누르면 루프 종료
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    # 사용한 자원 해제
    cap.release()

def update_saved_image():
    # 저장된 이미지를 PhotoImage로 변환하여 Label에 표시하며 크기 조절
    saved_img = Image.open(saved_image_path)
    resized_img = saved_img.resize((300, 300), Image.LANCZOS)  # 원하는 크기를 가로와 세로에 지정
    saved_img_tk = ImageTk.PhotoImage(resized_img)
    
    detect_label.config(image=saved_img_tk)
    detect_label.image = saved_img_tk

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)

class YoloThread(threading.Thread):
    sum = 0
    def run(self):
        save_img_event.set()  # 이벤트 설정

        im = "screenshot.jpg"  # file, Path, PIL.Image, OpenCV, nparray, list
        results = model(im)  # inference
        
        try:
            pred = results.pred[0]
            # class_ids 배열에서 첫 번째 값을 사용
            first_class_id = pred[:, 5][0].cpu().numpy()
            detected_value = int(model.names[int(first_class_id)])  # Assuming the detected value is an integer
            self.sum += detected_value  # Add the detected value to the sum
            
            yolo_label.config(text=str(detected_value))
            sum_label.config(text=str(self.sum))

        except IndexError:
            # 감지된 객체가 없을 경우 예외 처리
            yolo_label.config(text="No object detected")

# TTS 버튼을 클릭할 때 호출되는 콜백 함수
def tts_button_callback():
    global yolo_thread  # yolo_thread가 YoloThread 클래스의 인스턴스라고 가정
    # YoloThread 인스턴스에서 현재 합계 값을 가져오기
    current_sum = yolo_thread.sum
    # 합계 값을 문자열로 변환하고 speak 함수에 전달
    speak(str(current_sum))


# Tkinter 창 설정
window = tkinter.Tk()
window.title("지폐 인식 시스템")
window.geometry("1200x720+0+0")
window.resizable(False, False)

# 폰트 설정
ilabel_font = font.Font(family='Helvetica', size=20, weight='bold')
Yolo_font = font.Font(family='Helvetica', size=20, weight='bold')
Sum_font = font.Font(family='Helvetica', size=20, weight='bold')

# 버튼 생성

yolo_thread = YoloThread()
Mbutton = tkinter.Button(window, text="지폐 인식", command=lambda: threading.Thread(target=yolo_thread.run).start(), height=15, width=65)
Mbutton.grid(row=0, column=0, sticky='nsew')  # 버튼 위치를 왼쪽 상단으로

Tbutton = tkinter.Button(window, text="TTS", height=15, width=65,command=tts_button_callback)
Tbutton.grid(row=0, column=1, sticky='nsew')  # 버튼 위치를 오른쪽 상단으로

window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)

# 라벨 생성
ilabel = tkinter.Label(window, text="인식된 돈")
ilabel.place(x=830, y=450)
ilabel['font'] = ilabel_font

Slabel = tkinter.Label(window, text= "합계")
Slabel.place(x=870, y=530)
Slabel['font'] = Sum_font

yolo_label = tkinter.Label(window, text="")  # 인식된 물체 정보
yolo_label.place(x=970, y=450)
yolo_label['font'] = Yolo_font

detect_label = tkinter.Label(window, text="")  # 스크린샷 정보
detect_label.place(x=520, y=330)

sum_label = tkinter.Label(window,text="") #합계 정보
sum_label.place(x=970, y=530)
sum_label['font'] = Sum_font

camera_label = tkinter.Label(window)  # 카메라 영상
camera_label.place(x=0, y=270)

# 카메라 영상을 표시하는 스레드 시작
# cap = cv2.VideoCapture(0)  # 카메라 번호 0을 사용 (첫 번째 카메라)
camera_thread = threading.Thread(target=display_camera_feed)
camera_thread.start()

# Tkinter 창 실행
window.mainloop()
