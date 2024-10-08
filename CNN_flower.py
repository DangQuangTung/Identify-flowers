import numpy as np
from keras_preprocessing import image  
import cv2 
import os
import tensorflow as tf 
from keras import layers 
import time
from keras_preprocessing.image import ImageDataGenerator  

# Định nghĩa danh sách các lớp
classes = ['a_kudopul', 'b_cucvantho', 'c_campion', 'd_movet', 'e_uudambala']

# Tạo các đối tượng tạo dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=8.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('c:\\DOAN\\train',
                                                 target_size=(64, 64),
                                                 batch_size=12,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('c:\\DOAN\\test',
                                             target_size=(64, 64),
                                             batch_size=12,
                                             class_mode='categorical')

# Xây dựng mô hình CNN
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))  # Thay đổi số đơn vị đầu ra thành 5

# Biên dịch mô hình
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
cnn.fit(x=training_set, validation_data=test_set, epochs=30)

# Khởi tạo camera
vid = cv2.VideoCapture(0)
time.sleep(10)
print("Kết nối camera thành công")

i = 0
while True:
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    
    # Lưu frame
    cv2.imwrite('c:\\DOAN\\final' + str(i) + ".jpg", frame)
    
    # Tải và tiền xử lý hình ảnh
    test_image = image.load_img('c:\\DOAN\\final' + str(i) + ".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    # Dự đoán lớp
    result = cnn.predict(test_image)
    prediction = classes[np.argmax(result)]
    print(prediction)
    
    # Xóa hình ảnh đã lưu
    os.remove('c:\\DOAN\\final' + str(i) + ".jpg")
    
    i += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
