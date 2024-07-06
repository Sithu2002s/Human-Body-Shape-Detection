import cv2
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns

# Base dataset path
dataset_path = 'C:/Users/Sithumi/OneDrive - National Institute of Business Management/Desktop/dl_hnd232_a_02_03_10/SHAPE'

# paths for train, val, and test directories
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# Creating directories
for path in [train_path, val_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Body shape categories
body_shapes = ['Apple', 'Hourglass', 'Inverted_Triangle', 'Rectangle', 'Spoon']

# Creating subdirectories for each body shape in train, val, and test directories
for shape in body_shapes:
    for path in [train_path, val_path, test_path]:
        shape_path = os.path.join(path, shape)
        if not os.path.exists(shape_path):
            os.makedirs(shape_path)

# Function to split and move images
def split_data(shape, source_dir, train_dir, val_dir, test_dir, val_size=0.2, test_size=0.1):
    files = os.listdir(source_dir)
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size), random_state=42)

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, shape, file))
    
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, shape, file))
    
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, shape, file))

# Split the dataset for each body shape
for shape in body_shapes:
    source_dir = os.path.join(dataset_path, shape)
    if os.path.exists(source_dir):
        split_data(shape, source_dir, train_path, val_path, test_path)
    else:
        print(f"Directory {source_dir} does not exist.")

# Custom augmentation functions
def adjust_brightness(image):
    return tf.image.random_brightness(image, max_delta=0.2)

def adjust_contrast(image):
    return tf.image.random_contrast(image, lower=0.8, upper=1.2)

def adjust_saturation(image):
    return tf.image.random_saturation(image, lower=0.8, upper=1.2)

def adjust_hue(image):
    return tf.image.random_hue(image, max_delta=0.1)

def custom_augmentation(image):
    image = adjust_brightness(image)
    image = adjust_contrast(image)
    image = adjust_saturation(image)
    image = adjust_hue(image)
    return image

# Image data generator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=custom_augmentation
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Loading data from directories
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Disable shuffling for test set to align predictions with true labels
)

# Use a pre-trained model (VGG16) and fine-tune it
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=100
)

# Save the model
model.save('C:/Users/Sithumi/OneDrive - National Institute of Business Management/Desktop/dl_hnd232_a_02_03_10/best_model.h5')

# Loading the model
model = load_model('C:/Users/Sithumi/OneDrive - National Institute of Business Management/Desktop/dl_hnd232_a_02_03_10/best_model.h5')

# Evaluate the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Getting true labels and predictions
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=body_shapes, yticklabels=body_shapes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=body_shapes)
print(class_report)

# Precision, Recall, and F1 Score
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Function to preprocess image for prediction
def preprocess_image(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict body shape
def predict_body_shape(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return body_shapes[np.argmax(prediction)]

# Load the TensorFlow model
model_path = 'C:/Users/Sithumi/OneDrive - National Institute of Business Management/Desktop/dl_hnd232_a_02_03_10/frozen_inference_graph.pb'
pbtxt_path = 'C:/Users/Sithumi/OneDrive - National Institute of Business Management/Desktop/dl_hnd232_a_02_03_10/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(model_path, pbtxt_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect humans in the frame
    classes, confidences, boxes = net.detect(frame, confThreshold=0.5)
    human_detected = False

    for i in range(len(boxes)):
        class_id = int(classes[i])
        confidence = float(confidences[i])
        box = boxes[i]
        if class_id == 1:  # COCO class id for 'person' is 1
            human_detected = True
            (x, y, w, h) = box
            human_roi = frame[y:y+h, x:x+w]

            # Predict body shape
            predicted_shape = predict_body_shape(human_roi)

            # Display the resulting frame with the prediction
            cv2.putText(frame, f"Predicted body shape: {predicted_shape}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break

    if not human_detected:
        # Display "No human detected"
        cv2.putText(frame, "No human detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Webcam Body Shape Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
