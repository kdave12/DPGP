#!usr/bin/env python3

import numpy as np
import matplotlib as plt
import os
import pandas as pd
from frame import Frame
import pickle

# Preprocess Argoverse dataset into frames
# Structure of Argoverse
# data_path
# -- log_name.txt
# -- -- [lidar_time_stamp frame_idx track_id label_class x y z vel_x vel_y]

# Unique object type
# 'VEHICLE' 'LARGE_VEHICLE' 'ON_ROAD_OBSTACLE' 'TRAILER' 'PEDESTRIAN'
#  'BICYCLE' 'BICYCLIST'


map_range = np.array([[2570, 2600, 1180, 1210], [2640, 2670, 1240, 1270]])
area_idx = 0

frames = []
for folder_idx in range(4):
    data_path = '/home/mengdi/Dropbox/Research/Mobility21/Mobility21/train' + str(folder_idx+1) +'/'
    file_names = os.listdir(data_path)
    for i in range(len(file_names)):
        # open a new file
        data = pd.read_csv(data_path + file_names[i], sep=" ")
        data['log'] = file_names[i]

        # filter by the type of object
        data_range = data.loc[(data['label_class'] == 'VEHICLE') | (data['label_class'] == 'LARGE_VEHICLE')]
        # data_range = data.loc[(data['label_class'] == 'PEDESTRIAN') | (data['label_class'] == 'BICYCLIST')]

        # filter by range
        data_range = data_range[data_range['x'] >= map_range[area_idx, 0]]
        data_range = data_range[data_range['x'] <= map_range[area_idx, 1]]
        data_range = data_range[data_range['y'] >= map_range[area_idx, 2]]
        data_range = data_range[data_range['y'] <= map_range[area_idx, 3]]

        # delete the element with 0 velocty
        data_range = data_range[data_range['vel_x'] != 0.0]
        data_range = data_range[data_range['vel_y'] != 0.0]

        # load data file into frames
        if data_range.shape[0] != 0:
            frame_range_idx = data_range['frame_idx'].unique()
            for j in range(len(frame_range_idx)):
                # get data at frame_idx j and append to frames
                data_temp = data_range[data_range['frame_idx'] == frame_range_idx[j]]
                frame_temp = Frame(data_temp.x.values, data_temp.y.values,
                                   data_temp.vel_x.values, data_temp.vel_y.values)
                frames.append(frame_temp)
                del frame_temp

# save frames
with open("data_sample/frame_map_range_0_argo_all", "wb") as fb:
    pickle.dump(frames, fb)

print(len(frames))
print(frames[0].x.shape)