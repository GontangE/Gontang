import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import cv2
import numpy as np
from keras.models import load_model

# 모델 로드
model = load_model('model.h5')

class MnistApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # 카메라 초기화
        self.capture = cv2.VideoCapture(0)
        self.my_camera = Image()
        
        # 카메라 부분 레이아웃
        camera_layout = BoxLayout(size_hint=(1, 0.7))
        camera_layout.add_widget(self.my_camera)
        
        # 하단 레이아웃 (버튼과 예측 결과)
        bottom_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.3))
        
        # 예측 결과를 표시할 레이블 추가
        self.label = Label(text="Prediction: ", size_hint=(1, 0.2))
        bottom_layout.add_widget(self.label)
        
        # 촬영 버튼 추가
        button = Button(text="Capture and Predict", size_hint=(1, 0.8))
        button.bind(on_press=self.capture_and_predict)
        bottom_layout.add_widget(button)
        
        self.layout.add_widget(camera_layout)
        self.layout.add_widget(bottom_layout)
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.my_camera.texture = image_texture

    def capture_and_predict(self, instance):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite('captured.png', frame)
            
            img = cv2.imread('captured.png', cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (28, 28))
                img = img.reshape(1, 28, 28, 1)
                img = img.astype('float32') / 255
                
                # 예측 수행
                prediction = model.predict(img)
                predicted_digit = np.argmax(prediction)
                
                # 예측 결과 업데이트
                self.label.text = f"Prediction: {predicted_digit}"
            else:
                self.label.text = "Error: Couldn't load the captured image."

if __name__ == '__main__':
    MnistApp().run()
