import numpy as np

import torch
import torch.utils.data

import random
import logging

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, 150.0)/150.0

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)


def collect_data(env, max_size, mp_num, sub_num, epoch):
    train_data = []
    while len(train_data) < max_size:
        step_data_mp = env.step(env.action_space.sample())
        for i in range(mp_num): # one simulation multi-agent
            for j in range(sub_num): # one agent
                if step_data_mp[1][i][j] != 0: # reward
                    step_data = [step_data_mp[0][i][j], step_data_mp[1][i][j], step_data_mp[2], step_data_mp[3][i][j]]
                    step_data[0] = torch.from_numpy(step_data[0])
                    step_data[3] = [torch.from_numpy(np.expand_dims(s, 0).astype(np.float32)) for s in step_data[3][0]]
                    train_data.append(step_data)
                    logging.info('Collecting Data {:02d}/{} in epoch {}'.format(len(train_data), max_size, epoch))
                    if len(train_data) >= max_size:
                        return train_data
        env.reset()

def get_file_data(dataset_dir, max_size, batch_num):
    train_data = []
    cnt = 0
    while len(train_data) < max_size:
        filename_0 = dataset_dir + str(batch_num * 100 + cnt).zfill(4) + "_0.npy.npz" 
        filename_3 = dataset_dir + str(batch_num * 100 + cnt).zfill(4) + "_3.npy.npz" 
        print("load data:", filename_0, filename_3)
        with np.load(filename_0, allow_pickle=True) as data_0:
            with np.load(filename_3, allow_pickle=True) as data_3:
                train_data.append([torch.from_numpy(data_0['arr_0']), 0.,0.,[torch.from_numpy(np.expand_dims(s, 0).astype(np.float32)) for s in data_3["arr_0"][0]], data_3['arr_0'][1]])
        cnt += 1
    random.shuffle(train_data)
    return train_data

