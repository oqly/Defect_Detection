import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_sample_mesh(file_path, n_points=2048):
    # Загрузка меша
    mesh = trimesh.load(file_path, force='mesh')
    mesh.show()

    # Пример точек из меша
    points = mesh.sample(n_points)

    return points


def visualize_point_cloud(points):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title("Point Cloud Visualization")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def main(file_path, n_points):
    points = load_and_sample_mesh(file_path, n_points)
    visualize_point_cloud(points)


if __name__ == "__main__":
    file_path = './dataset/augmented/defect/9.859.stl'
    n_points = 15000
    main(file_path, n_points)

    file_path = './dataset/augmented/non_defect/0.030.stl'
    n_points = 15000
    main(file_path, n_points)
