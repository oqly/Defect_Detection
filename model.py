import sqlite3
from datetime import datetime
import numpy as np
import trimesh
import tensorflow as tf
from keras import regularizers
from keras.models import load_model


currentModel = 'model_fold_094_f096.keras'
#model_fold_096_2.keras
#model_dd_2405.keras
#model_dd_1105_098_100ds_32bs.keras
#model_dd_2705_1.keras

# Регистрация пользовательского регуляризатора
@tf.keras.utils.register_keras_serializable()
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


class DefectDetectionModel:
    def __init__(self):
        self.model = load_model(currentModel, custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})
        self.conn = sqlite3.connect('analysis_results.db')
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    result TEXT NOT NULL,
                    model_version TEXT NOT NULL
                )
                ''')
        self.conn.commit()

    def sample_points_from_mesh(self, file_path, n_points=5000):
        mesh = trimesh.load(file_path)
        points = mesh.sample(n_points)
        return points

    def analyze(self, file_path):
        points = self.sample_points_from_mesh(file_path)
        points = np.expand_dims(points, axis=0)
        prediction = self.model.predict(points)
        result = 'Defective' if np.argmax(prediction) == 1 else 'Non-Defective'
        self.save_result(file_path, result)
        return result

    def save_result(self, file_name, result):
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_version = currentModel
        self.cursor.execute('''
                INSERT INTO analysis_results (file_name, analysis_date, result, model_version)
                VALUES (?, ?, ?, ?)
                ''', (file_name, analysis_date, result, model_version))
        self.conn.commit()
