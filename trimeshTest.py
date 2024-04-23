from stl import mesh
import trimesh
import numpy as np
import os
import seaborn as sns
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Input, Flatten, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize


n_points = 15000


def sample_points_from_mesh(mesh, n_points):
    points = mesh.sample(n_points)
    return points


def load_point_clouds_from_mesh(folder, label, n_points, max_files=None):
    all_points = []
    labels = []
    files = [f for f in os.listdir(folder) if f.endswith('.stl')]
    if max_files is not None:
        files = files[:max_files]
    for f in files:
        mesh_path = os.path.join(folder, f)
        mesh = trimesh.load(mesh_path, force='mesh')
        points = sample_points_from_mesh(mesh, n_points)
        all_points.append(points)
        labels.append(label)
    print("folder meshed")
    return np.array(all_points), np.array(labels)


def create_pointnet_model():
    input_points = Input(shape=(n_points, 3))
    x = Conv1D(64, 1, activation='relu')(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=n_points)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=input_points, outputs=x)


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.show()


print("Start")

# Пути к папкам с данными
defect_folder = './dataset/defect'
non_defect_folder = './dataset/non_defect'
augmented_defect_folder = './dataset/augmented/defect'
augmented_non_defect_folder = './dataset/augmented/non_defect'

print("Load")
# Загрузка точечных облаков из данных
X_defect, y_defect = load_point_clouds_from_mesh(defect_folder, 1, n_points=n_points)
X_non_defect, y_non_defect = load_point_clouds_from_mesh(non_defect_folder, 0, n_points=n_points)
X_aug_defect, y_aug_defect = load_point_clouds_from_mesh(augmented_defect_folder, 1, n_points=n_points, max_files=250)
X_aug_non_defect, y_aug_non_defect = load_point_clouds_from_mesh(augmented_non_defect_folder, 0, n_points=n_points, max_files=250)

print("Combo")
# Комбинирование всех данных в один набор
X = np.concatenate((X_defect, X_non_defect, X_aug_defect, X_aug_non_defect), axis=0)
y = np.concatenate((y_defect, y_non_defect, y_aug_defect, y_aug_non_defect))
#X = np.concatenate((X_defect, X_non_defect), axis=0)
#y = np.concatenate((y_defect, y_non_defect))
y = to_categorical(y)


print("Model")
# Создание модели
model = create_pointnet_model()

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32)

# Визуализация истории обучения
plot_training_history(history)

# Сохранение модели
model.save('3d_model_defect_detection_2304.h5')

model.summary()

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Predictions for evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_classes))

# ROC Curve and AUC
y_test_bin = label_binarize(y_true, classes=[0, 1])
y_pred_bin = label_binarize(y_pred_classes, classes=[0, 1])
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
