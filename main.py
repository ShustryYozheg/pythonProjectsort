import os
import face_recognition
import shutil

# Путь к папке с вашими фотографиями (позитивными образцами)
positive_samples_folder = 'D:\\for py\\positive'

# Папка, в которой будут сохранены фотографии с вашим лицом
output_folder = 'D:\\for py\\positive\\done'

# Загрузим все фотографии из папки и получим кодировки лиц для каждой из них
positive_face_encodings = []

for filename in os.listdir(positive_samples_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(positive_samples_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            # Извлекаем кодировку лица (первого найденного лица на фотографии)
            face_encoding = face_encodings[0]
            positive_face_encodings.append(face_encoding)

# Пройдем по всем фотографиям в папке, в которой вы хотите найти фотографии с вашим лицом
for filename in os.listdir('D:\\for py\\Photos1'):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join('D:\\for py\\Photos1', filename)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_encoding in face_encodings:
            # Сравним лицо на фотографии с вашими позитивными образцами
            matches = face_recognition.compare_faces(positive_face_encodings, face_encoding)

            if any(matches):
                print(f"Найдено ваше лицо на фотографии: {filename}")
                # Переместим или скопируем фотографию с вашим лицом в выходную папку
                shutil.copy(image_path, os.path.join('D:\\for py', filename))
                break
