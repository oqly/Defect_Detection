import trimesh
import numpy as np
import os


def smoothstep(x, min_val, max_val):
    t = np.clip((x - min_val) / (max_val - min_val), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def twist_model(mesh, angle, twist_height_percent=0.08, smoothness=0.5, axis=(0, 0, 1)):
    # Positive angle - counter clockwise,
    # Negative angle - clockwise,

    # Переводим  уголв радианы с помощью numpy
    angle_rad = np.radians(angle)

    # Определяем граничную высоту для скручивания
    min_z, max_z = mesh.bounds[:, 2]
    twist_height = min_z + (max_z - min_z) * twist_height_percent

    # Создаем матрицу поворота
    rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, axis)

    # Применяем поворот к вершинам модели, с плавным переходом для верхней части
    for i, vertex in enumerate(mesh.vertices):
        t = smoothstep(vertex[2], twist_height, max_z)
        if vertex[2] >= twist_height:
            angle_t = angle_rad * smoothstep(vertex[2], twist_height, max_z)**smoothness
            rotation_matrix_t = trimesh.transformations.rotation_matrix(angle_t, axis)
            mesh.vertices[i] = np.dot(rotation_matrix_t[:3, :3], vertex) + rotation_matrix_t[:3, 3]

    return mesh


# Генерируем 500 3D-моделей с разным углом скручивания и сохраняем их
def generate_and_save_models():
    np.random.seed(42)
    num_models = 140
    min_angle = -3
    max_angle = 3

    for i in range(num_models):
        angle = np.random.uniform(min_angle, max_angle)
        rounded_angle = round(angle, 4)

        # Загружаем эталонную модель (здесь подставьте путь к вашей 3D-модели)
        mesh = trimesh.load('./dataset/original/Эталон.stl')

        # Применяем скручивание к модели
        mesh_twisted = twist_model(mesh.copy(), angle)

        # Определяем папку для сохранения модели в зависимости от угла скручивания
        save_directory = './dataset2/augmented/non_defect' if abs(angle) <= 3 else './dataset2/augmented/defect'

        # Сохраняем модель в файл
        file_name = os.path.join(save_directory, f'{rounded_angle:.3f}.stl')
        mesh_twisted.export(file_name)

if __name__ == "__main__":
    generate_and_save_models()