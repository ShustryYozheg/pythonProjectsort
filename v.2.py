import os
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

            # Преобразуем изображение в чб и изменяем размер, если необходимо
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (100, 100))  # Можете изменить размер по вашему усмотрению

            # Добавляем изображение и метку в списки
            images.append(image)
            labels.append(label)

# Преобразуем списки в массивы NumPy
images = np.array(images)
labels = np.array(labels)

# Сохраняем данные в файлы
np.save(os.path.join(output_folder, 'images.npy'), images)
np.save(os.path.join(output_folder, 'labels.npy'), labels)


# Загрузим данные из файлов
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

