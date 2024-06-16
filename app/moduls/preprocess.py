#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import csv

def detect_dead_pixels(image, dark_threshold=0.2, overlit_threshold=3, window_size=5):
    height, width, channels = image.shape
    dead_pixels_mask = np.zeros((height, width, channels), dtype=bool)
    
    half_window = window_size // 2
    for channel in range(channels):
        for x in range(half_window, height - half_window):
            for y in range(half_window, width - half_window):
                local_patch = image[x - half_window:x + half_window + 1, y - half_window:y + half_window + 1, channel]
                local_mean = np.mean(local_patch)
                
                if image[x, y, channel] == 0:
                    dead_pixels_mask[x, y, channel] = True
                elif image[x, y, channel] < dark_threshold * local_mean:
                    dead_pixels_mask[x, y, channel] = True
                elif image[x, y, channel] > overlit_threshold * local_mean:
                    dead_pixels_mask[x, y, channel] = True
    
    return dead_pixels_mask

def restore_dead_pixels(image, dead_pixels_mask):
    restored_image = image.copy()
    
    for channel in range(image.shape[2]):
        mask = dead_pixels_mask[:, :, channel]
        median_filtered = median_filter(image[:, :, channel], size=3)
        
        for x, y in zip(*np.where(mask)):
            neighbors = []
            for i in range(max(0, x-1), min(image.shape[0], x+2)):
                for j in range(max(0, y-1), min(image.shape[1], y+2)):
                    if not mask[i, j]:
                        neighbors.append(image[i, j, channel])
            if neighbors:
                restored_image[x, y, channel] = np.mean(neighbors)
            else:
                restored_image[x, y, channel] = median_filtered[x, y]
    
    return restored_image

def generate_bug_report_csv(dead_pixels_mask, original_image, restored_image, report_path):
    height, width, channels = original_image.shape
    
    with open(report_path, 'w', newline='') as csvfile:
        report_writer = csv.writer(csvfile, delimiter=',')
        report_writer.writerow(['Row', 'Column', 'Channel', 'Dead Value', 'Corrected Value'])
        
        for channel in range(channels):
            for x in range(height):
                for y in range(width):
                    if dead_pixels_mask[x, y, channel]:
                        report_writer.writerow([x, y, channel+1, original_image[x, y, channel], restored_image[x, y, channel]])

def process_image(image_path, image_name):
    image = tifffile.imread(image_path+"/"+image_name)

    # Detect dead pixels
    dead_pixels_mask = detect_dead_pixels(image)

    # Restore dead pixels
    restored_image = restore_dead_pixels(image, dead_pixels_mask)

    # Generate bug report
    base_filename = os.path.basename(image_name)
    bug_report_path = os.path.join(image_path, os.path.splitext(base_filename)[0] + '_bug_report.csv')
    generate_bug_report_csv(dead_pixels_mask, image, restored_image, bug_report_path)

    # Save the restored image
    restored_image_path = os.path.join(image_path, os.path.splitext(base_filename)[0] + '_restored.tif')
    tifffile.imwrite(restored_image_path, restored_image.astype(np.uint16))

    return restored_image

# Example usage
# folder_path = "C:/Users/remso/OneDrive/Desktop/competition/18. Sitronics/1_20"
# process_image(folder_path, "crop_0_0_0000.tif")



