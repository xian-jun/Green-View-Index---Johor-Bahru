import os
import pandas as pd
import numpy as np
from PIL import Image


def load_and_resize_pred(image_loc, current_model):
    '''
    load street view images as array
    '''
    # current_model = [224, 224, 3]
    test_data = []
    test_gt = []
    with open(image_loc, 'r') as training:
        content = training.readlines()
    for line in content:
        line = line.replace('\n', '')
        test_data.append(line)
    imgdata = []
    for path1 in test_data:
        imgdata.append((Image.open(path1)).resize(current_model[0:2]))
    imgdata_array = np.zeros(
        [len(imgdata), current_model[0], current_model[1], current_model[2]], dtype=np.uint8)
    for i in range(len(imgdata)):
        imgdata_array[i, :, :, :] = imgdata[i]
    del (imgdata)
    return (imgdata_array)


def load_and_resize(image_loc, current_model):
    '''
    load masked_img and masked_2channel into array and resize them. 
    '''
    test_data = []
    test_gt = []
    with open(image_loc, 'r') as training:
        content = training.readlines()
    for line in content:
        paths = line.split()
        if len(paths) == 2:
            test_data.append(paths[0])
            test_gt.append(paths[1].replace("\n", ""))
        if len(paths) == 3:
            test_data.append(paths[0] + " "+paths[1])
            test_gt.append(paths[2].replace("\n", ""))
        if len(paths) == 1:
            test_data.append(paths[0])
    imgdata = []
    for path1 in test_data:
        imgdata.append(np.asarray(
            (Image.open(path1).resize(current_model[0:2]))))
    labeldata = []
    for path in test_gt:
        labeldata.append(np.asarray(
            (Image.open(path).resize(current_model[0:2]))))

    # convert list of arrays to arrays
    imgdata_array = np.zeros(
        [len(imgdata), current_model[0], current_model[1], current_model[2]], dtype=np.uint8)
    labeldata_array = np.zeros(
        [len(imgdata), current_model[0], current_model[1]], dtype=np.uint8)
    for i in range(len(imgdata)):
        imgdata_array[i, :, :, :] = imgdata[i]
        labeldata_array[i, :, :] = labeldata[i]
    del (imgdata, labeldata)
    return (imgdata_array, labeldata_array)


# extract metadata from file name
def save_file_path_in_list(path):
    data = []
    gt = []
    with open(path, 'r') as f:
        content = f.readlines()
    for line in content:
        paths = line.split()
        if len(paths) == 2:
            data.append(paths[0])
            gt.append(paths[1].replace("\n", ""))
        else:
            data.append(paths[0].replace("\n", ""))

    return data, gt


def retrieve_metadata(path):

    data, gt = save_file_path_in_list(path)

    label_metadata = []
    for file_name in data:
        # Retrieve the file name
        file_name = os.path.basename(file_name)

        # Remove the file extension
        file_name = os.path.splitext(file_name)[0]
        # Split the file name into its components using the '-' separator
        file_components = file_name.split('-')
        # Extract the latitude, longitude, and heading information from the file name
        lat = (file_components[0])
        lng = (file_components[1])
        heading = int(file_components[2].split('.')[0])

        label_metadata.append([lat, lng, heading])

    return pd.DataFrame(label_metadata, columns=['lat', 'lng', 'heading'])
