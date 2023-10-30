import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Путь к папке, в которой находятся папки с фотографиями членов семьи
family_members_root_folder = 'D:\\for py\\Family'

# Получаем список имен подпапок (имен членов семьи)
family_members_folders = os.listdir(family_members_root_folder)

# Папка, в которой будут сохранены данные для обучения
output_folder = 'D:\\for py\\Family data'

# Создаем папку для сохранения данных
os.makedirs(output_folder, exist_ok=True)

# Инициализируем списки для хранения изображений и меток
images = []
labels = []

# Проходим по каждой папке с фотографиями членов семьи
for label, family_member_folder in enumerate(family_members_folders):
    folder_path = os.path.join(family_members_root_folder, family_member_folder)

    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)

            # Загружаем изображение с помощью OpenCV
            image = cv2.imread(image_path)

            # Предобрабатываем изображение
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))
            image = np.expand_dims(image, axis=-1)

            # Добавляем изображение и метку в списки
            images.append(image)
            labels.append(label)

# Преобразуем списки в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Сохраняем данные в файлы
np.save(os.path.join(output_folder, 'images.npy'), images)
np.save(os.path.join(output_folder, 'labels.npy'), labels)

# Загружаем данные из файлов
images = np.load(os.path.join(output_folder, 'images.npy'))
labels = np.load(os.path.join(output_folder, 'labels.npy'))

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Создадим нейронную сеть для классификации лиц
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),  # Входной слой (сплющиваем изображение)
    keras.layers.Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами
    keras.layers.Dense(len(family_members_folders), activation='softmax')  # Выходной слой с количеством классов равным количеству членов семьи
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель на обучающем наборе данных
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Сохраняем обученную модель
model.save('family_face_recognition_model.h5')

# Загружаем обученную модель
model = keras.models.load_model('family_face_recognition_model.h5')

# Указываем абсолютный путь к папке с фотографиями на диске D
input_folder = 'D:\\for py\\Photos1'

# Создаем папки для членов семьи и папку "Другие фото" в указанном абсолютном пути
family_members = ['Marina', 'Sergey', 'Yarik', 'Kirill']
for member in family_members:
    os.makedirs(os.path.join(input_folder, member), exist_ok=True)
os.makedirs(os.path.join(input_folder, 'Другие фото'), exist_ok=True)

# Проходим по всем фотографиям в папке
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Предобрабатываем изображение
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))
        image = np.expand_dims(image, axis=-1)

        # Используем модель для распознавания лиц
        predicted_labels = model.predict(np.array([image]))

        # Выводим информацию о том, кого модель видит на фотографии
        print(f'На фотографии "{filename}" обнаружен: {family_member}')

        # Присваиваем каждому члену семьи метку (или "Другие фото", если не распознано)
        family_member = None
        if predicted_labels.argmax() < len(family_members):
            family_member = family_members[predicted_labels.argmax()]

        # Перемещаем фотографию в соответствующую папку
        if family_member is not None:
            destination_folder = os.path.join(input_folder, family_member)
            shutil.move(image_path, os.path.join(destination_folder, filename))
        else:
            # Если не удалось определить члена семьи, перемещаем фотографию в папку "Другие фото"
            destination_folder = os.path.join(input_folder, 'Другие фото')
            shutil.move(image_path, os.path.join(destination_folder, filename))
