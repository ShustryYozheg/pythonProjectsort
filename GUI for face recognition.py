import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

import self as self
from PIL import Image, ImageTk
from tensorflow import keras
import dlib

# Загрузите модель нейронной сети
model = keras.models.load_model('family_face_recognition_model.h5')

# Папка с фотографиями, которые нужно классифицировать
input_folder = 'D:\\for py\\Photos1'

# Создайте детектор лиц с использованием dlib
face_detector = dlib.get_frontal_face_detector()

class FamilyPhotoClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Family Photo Classifier")
        self.root.geometry("800x800")

        # Создайте виджет для отображения изображения
        self.photo_label = ttk.Label(root)
        self.photo_label.pack(fill="both", expand=True)

        # Создайте кнопки для "Правильно" и "Неправильно"
        self.correct_button = ttk.Button(root, text="Правильно", command=self.correct_photo)
        self.correct_button.pack(side="left", padx=10)
        self.incorrect_button = ttk.Button(root, text="Неправильно", command=self.incorrect_photo)
        self.incorrect_button.pack(side="right", padx=10)

        # Создайте виджет для отображения предположения
        self.prediction_label = ttk.Label(root, text="")
        self.prediction_label.pack()

        # Подготовьте список файлов для классификации
        self.image_files = [filename for filename in os.listdir(input_folder) if filename.endswith('.jpg') or filename.endswith('.png')]
        self.current_image_index = 0

        # Отобразите первое изображение
        self.display_current_image()

        # Создайте кнопку "Закрыть"
        self.close_button = ttk.Button(root, text="Закрыть", command=self.close_app)
        self.close_button.pack()

    def display_current_image(self):
     if self.current_image_index < len(self.image_files):
        image_path = os.path.join(input_folder, self.image_files[self.current_image_index])
        image = cv2.imread(image_path)

        # Преобразуйте изображение в оттенки серого для детектора лиц
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружьте лицо на изображении
        faces = face_detector(gray_image)

        # Если обнаружено лицо, нарисуйте прямоугольник вокруг него
        if len(faces) > 0:
            (x, y, w, h) = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_image = image[y:y+h, x:x+w]
        else:
            face_image = image  # Если лицо не обнаружено, используйте всё изображение

        # Преобразуйте изображение в оттенки серого и измените размер до 100x100
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (100, 100))
        face_image = np.expand_dims(face_image, axis=-1)

        # Масштабируйте изображение для отображения
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((800, 800))  # Масштабирование изображения
        photo = ImageTk.PhotoImage(image=image)
        self.photo_label.config(image=photo)
        self.photo_label.image = photo

        # Используйте модель для распознавания лиц
        predicted_labels = model.predict(np.array([face_image]))

        # Выведите предположение о лице
        family_members = ['Marina', 'Sergey', 'Yarik', 'Kirill']
        if predicted_labels.argmax() < len(family_members):
            prediction = f'На фото: {family_members[predicted_labels.argmax()]}'
        else:
            prediction = 'На фото: Неизвестный член семьи'

        self.prediction_label.config(text=prediction)

    def correct_photo(self):
        # Обработка правильной классификации
        if self.current_image_index < len(self.image_files):
            print(f'Фотография "{self.image_files[self.current_image_index]}" классифицирована как правильная.')
            self.current_image_index += 1
            self.display_current_image()

    def incorrect_photo(self):
        # Обработка неправильной классификации
        if self.current_image_index < len(self.image_files):
            print(f'Фотография "{self.image_files[self.current_image_index]}" классифицирована как неправильная.')
            self.current_image_index += 1
            self.display_current_image()

    def close_app(self):
        # Закрыть приложение
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FamilyPhotoClassifierApp(root)
    root.mainloop()
