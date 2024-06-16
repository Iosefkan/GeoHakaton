#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import json

def read_image_with_rasterio(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.moveaxis(image, 0, -1)  # Перемещаем каналы в конец для совместимости с OpenCV
        if image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] > 3:
            image_rgb = image[:, :, :3]  # Используем только первые 3 канала (R, G, B)
        else:
            image_rgb = image
        
        # Проверяем и конвертируем изображение в uint8
        if image_rgb.dtype != np.uint8:
            image_rgb = cv2.convertScaleAbs(image_rgb, alpha=(255.0/np.max(image_rgb)))
        return image_rgb, src

def detect_and_match_features(scene_image, layout_image):
    # Используем ORB для обнаружения и описания ключевых точек
    orb = cv2.ORB_create()

    # Обнаружение ключевых точек и вычисление дескрипторов
    keypoints_scene, descriptors_scene = orb.detectAndCompute(scene_image, None)
    keypoints_layout, descriptors_layout = orb.detectAndCompute(layout_image, None)

    # Сопоставление ключевых точек с помощью BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_scene, descriptors_layout)

    # Сортируем совпадения по расстоянию (чем меньше, тем лучше)
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints_scene, keypoints_layout, matches

def get_matched_keypoints(keypoints_scene, keypoints_layout, matches):
    points_scene = np.zeros((len(matches), 2), dtype=np.float32)
    points_layout = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_scene[i, :] = keypoints_scene[match.queryIdx].pt
        points_layout[i, :] = keypoints_layout[match.trainIdx].pt

    return points_scene, points_layout

def calculate_transformed_scene_corners(scene_rgb, transform_matrix, layout_transform):
    height, width, _ = scene_rgb.shape

    def normalize(point):
        return point[:2] / point[2]

    # Вычисляем координаты углов сцены после преобразования
    top_left_scene = normalize(np.dot(transform_matrix, np.array([0, 0, 1])))
    top_right_scene = normalize(np.dot(transform_matrix, np.array([width, 0, 1])))
    bottom_right_scene = normalize(np.dot(transform_matrix, np.array([width, height, 1])))
    bottom_left_scene = normalize(np.dot(transform_matrix, np.array([0, height, 1])))

    # Привязываем координаты сцены к координатной сетке подложки
    top_left_layout = layout_transform * (top_left_scene[0], top_left_scene[1])
    top_right_layout = layout_transform * (top_right_scene[0], top_right_scene[1])
    bottom_right_layout = layout_transform * (bottom_right_scene[0], bottom_right_scene[1])
    bottom_left_layout = layout_transform * (bottom_left_scene[0], bottom_left_scene[1])

    coords_text = []
    coords_text.append(f"top_left: ({top_left_layout[0]:.3f}, {top_left_layout[1]:.3f})")
    coords_text.append(f"top_right: ({top_right_layout[0]:.3f}, {top_right_layout[1]:.3f})")
    coords_text.append(f"bottom_right: ({bottom_right_layout[0]:.3f}, {bottom_right_layout[1]:.3f})")
    coords_text.append(f"bottom_left: ({bottom_left_layout[0]:.3f}, {bottom_left_layout[1]:.3f})")

    coords = {
        "type": "Polygon",
        "coordinates": [[
            [top_left_layout[0], top_left_layout[1]],
            [top_right_layout[0], top_right_layout[1]],
            [bottom_right_layout[0], bottom_right_layout[1]],
            [bottom_left_layout[0], bottom_left_layout[1]],
            [top_left_layout[0], top_left_layout[1]]
        ]]
    }

    return coords, coords_text, [top_left_layout, top_right_layout, bottom_right_layout, bottom_left_layout]

def save_geojson(coords, output_path):
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": coords,
                "properties": {}
            }
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)
        
def save_geotiff(image, layout_corners, crs, output_path):
    # Создаем трансформацию на основе углов изображения сцены
    top_left, top_right, bottom_right, bottom_left = layout_corners
    height, width, _ = image.shape

    # Определяем новые границы изображения
    x_min = min(top_left[0], bottom_left[0])
    x_max = max(top_right[0], bottom_right[0])
    y_min = min(top_left[1], top_right[1])
    y_max = max(bottom_left[1], bottom_right[1])

    new_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=image.shape[2],
        dtype=image.dtype,
        crs=crs,
        transform=new_transform
    ) as dst:
        for i in range(image.shape[2]):
            dst.write(image[:, :, i], i + 1)

def process_images(layers_folder, input_tiff_path, layout_name, scene_name, output_geojson_path, output_geotiff_path):
    # Получаем путь к подложке
    layout_path = os.path.join(layers_folder, layout_name)
    
    # Чтение изображений
    scene_rgb, scene_src = read_image_with_rasterio(os.path.join(input_tiff_path, scene_name))
    layout_rgb, layout_src = read_image_with_rasterio(layout_path)
    
    # Обнаружение и сопоставление признаков
    keypoints_scene, keypoints_layout, matches = detect_and_match_features(scene_rgb, layout_rgb)
    
    # Получение совпадающих ключевых точек
    points_scene, points_layout = get_matched_keypoints(keypoints_scene, keypoints_layout, matches)
    
    # Вычисление матрицы преобразования
    transform_matrix, mask = cv2.findHomography(points_scene, points_layout, cv2.RANSAC)
    
    # Получение аффинного преобразования для подложки
    layout_transform = layout_src.transform
    
    # Вычисление координат углов сцены после преобразования и вывод их в текстовый файл
    coords, scene_coords_text, layout_corners = calculate_transformed_scene_corners(scene_rgb, transform_matrix, layout_transform)
    
    # Сохранение координат в формате GeoJSON
    save_geojson(coords, output_geojson_path)
    
    # Сохранение сцены в формате GeoTiff с геопривязкой
    save_geotiff(scene_rgb, layout_corners, CRS.from_epsg(32637), output_geotiff_path)
    
    return coords

# Пример использования
# layers_folder = r"path_to_your_layout_folder"
# input_tiff_path = r"path_to_your_scene_folder"
# layout_name = "layout_image.tif"
# scene_name = "scene_image.tif"
# output_geojson_path = r"output.geojson"
# output_geotiff_path = r"output_geotiff.tif"

# process_images(layers_folder, input_tiff_path, layout_name, scene_name, output_geojson_path, output_geotiff_path)

