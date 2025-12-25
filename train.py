import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# =============================
# 1. LOAD MNIST (DIGITS)
# =============================
(x_train_d, y_train_d), (x_test_d, y_test_d) = tf.keras.datasets.mnist.load_data()

# Normalize & reshape
x_train_d = x_train_d / 255.0
x_test_d = x_test_d / 255.0
x_train_d = x_train_d.reshape(-1, 28, 28, 1)
x_test_d = x_test_d.reshape(-1, 28, 28, 1)

# =============================
# 2. LOAD EMNIST LETTERS (TFDS)
# =============================
(ds_train_a, ds_test_a), _ = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))  # ✅ force correct shape
    label = label + 9   # 1–26 → 10–35
    return image, label


ds_train_a = ds_train_a.map(preprocess)
ds_test_a = ds_test_a.map(preprocess)

# Convert TFDS → NumPy
x_train_a, y_train_a = [], []
for img, lbl in tfds.as_numpy(ds_train_a):
    x_train_a.append(img)
    y_train_a.append(lbl)

x_test_a, y_test_a = [], []
for img, lbl in tfds.as_numpy(ds_test_a):
    x_test_a.append(img)
    y_test_a.append(lbl)

x_train_a = np.array(x_train_a)
y_train_a = np.array(y_train_a)
x_test_a = np.array(x_test_a)
y_test_a = np.array(y_test_a)

# =============================
# 3. COMBINE DIGITS + LETTERS
# =============================
x_train = np.concatenate((x_train_d, x_train_a))
y_train = np.concatenate((y_train_d, y_train_a))

x_test = np.concatenate((x_test_d, x_test_a))
y_test = np.concatenate((y_test_d, y_test_a))

# One-hot encode
num_classes = 36
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# =============================
# 4. BUILD CNN
# =============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================
# 5. TRAIN
# =============================
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# =============================
# 6. EVALUATE
# =============================
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# =============================
# 7. SAVE MODEL
# =============================
model.save("alphanumeric_cnn_model.h5")
print("Model saved successfully!")

# =============================
# 8. PREDICT FROM CUSTOM IMAGE
# =============================
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def decode_label(label):
    if label < 10:
        return str(label)
    else:
        return chr(label + 55)  # 10 -> A

# 🔹 GIVE YOUR IMAGE PATH HERE
img_path = r"F:/Git/images/Number.jfif"

# Load & preprocess image
img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Predicted Character:", decode_label(predicted_class))

# Show image
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {decode_label(predicted_class)}")
plt.axis('off')
plt.show()
