from stl import mesh
import trimesh
import numpy as np
import os
import seaborn as sns
import tensorflow as tf
from keras import layers, regularizers
from keras.utils import to_categorical
from keras.models import Model
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dense, Input, Flatten, Dropout
#from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, fbeta_score
from sklearn.preprocessing import label_binarize
from tensorflow.python.keras.callbacks import EarlyStopping

# Параметры
n_points = 5000
num_classes = 2
folder_max_files = 330
model_batch_size = 64
k_num = 5


# Вспомогательные функции для построения модели
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


# Определение OrthogonalRegularizer
class OrthogonalRegularizer(regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {"num_features": self.num_features, "l2reg": self.l2reg}


# Определение T-net
def tnet(inputs, num_features):
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


# Определение модели PointNet
def create_pointnet_model(num_points, num_classes):
    inputs = Input(shape=(num_points, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model


# Вспомогательная функция для выборки точек из меша
def sample_points_from_mesh(mesh, n_points):
    points = mesh.sample(n_points)
    return points


# Загрузка точечных облаков из файлов мешей в папке
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


# Функция для визуализации истории обучения
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
X_aug_defect, y_aug_defect = load_point_clouds_from_mesh(augmented_defect_folder, 1, n_points=n_points, max_files=folder_max_files)
X_aug_non_defect, y_aug_non_defect = load_point_clouds_from_mesh(augmented_non_defect_folder, 0, n_points=n_points, max_files=folder_max_files)

print("Combo")
# Комбинирование всех данных в один набор
X = np.concatenate((X_defect, X_non_defect, X_aug_defect, X_aug_non_defect), axis=0)
y = np.concatenate((y_defect, y_non_defect, y_aug_defect, y_aug_non_defect))
y = to_categorical(y, num_classes=num_classes)

# Проверка данных
print("Check data distribution before training:")
print("Defect count:", len(y_defect))
print("Non-defect count:", len(y_non_defect))
print("Augmented defect count:", len(y_aug_defect))
print("Augmented non-defect count:", len(y_aug_non_defect))

labels, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
print("Labels distribution in combined data:", dict(zip(labels, counts)))

# K-кратная кросс-валидация
kf = KFold(n_splits=k_num, shuffle=True, random_state=17)
all_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = create_pointnet_model(n_points, num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping], batch_size=model_batch_size)

    # Визуализация истории обучения
    plot_training_history(history)

    # Сохранение модели
    model_save_path = f'model_fold_{fold}.keras'
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    # Оценка модели на валидационных данных
    loss, accuracy = model.evaluate(X_val, y_val)
    all_scores.append(accuracy)

    # Предсказания на валидационных данных
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred_classes))

    # F2-Score
    f2 = fbeta_score(y_true, y_pred_classes, beta=2)
    print(f'F2-Score: {f2:.4f}')

    # Визуализация распределения предсказанных вероятностей для положительного класса
    sns.histplot(y_pred[:, 1], bins=50, kde=True)
    plt.xlabel('Predicted Probability for Positive Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')
    plt.show()

    # ROC Curve and AUC
    y_val_bin = label_binarize(y_true, classes=[0, 1])
    y_pred_bin = y_pred[:, 1]
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_pred_bin)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Средняя точность по всем фолдам
mean_accuracy = np.mean(all_scores)
print(f'Mean accuracy: {mean_accuracy:.4f}')
