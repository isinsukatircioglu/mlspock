import os
import h5py
import glob
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, None] - dst[None]) ** 2, dim=-1)

def sphere_normalization(pc):
    centroid = np.mean(pc, axis=0)
    pc_c = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_c ** 2, axis=1)))
    # pc_normalized = pc / m
    return m

class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1], **kwargs):
        self.angle = angle
    def __call__(self, data):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        data = np.dot(data, np.transpose(R))
        return data


class PointCloudDataset(Dataset):
    def __init__(self, opts, split='training', process_data=True):
        self.data_path = opts.path_data
        self.segment = opts.segment
        self.npoints = 882
        if self.segment == 'bottom' or self.segment == 'top':
            self.npoints = self.npoints//2
        self.norm = opts.norm
        self.scaling_mode = opts.scaling_mode
        self.lprotocol = opts.lprot
        self.lratio = opts.lratio
        self.split = split
        self.process_data = process_data
        self.file = os.path.join(self.data_path, self.split+'.txt')
        self.filenames = [line.rstrip() for line in open(self.file)]
        self.filenames_rot180 = []
        self.filenames_mirror = []
        self.num_neigh = 32
        self.delta = opts.delta
        self.delta3d = opts.delta3d
        
        for fn in range(len(self.filenames)):
            self.filenames_rot180.append(self.filenames[fn] + ',r')
            self.filenames_mirror.append(self.filenames[fn] + ',m')
            self.filenames[fn] = self.filenames[fn] + ',o' 
        if opts.rotate180:
            self.filenames.extend(self.filenames_rot180)
        if opts.mirror:
            self.filenames.extend(self.filenames_mirror)
        if split != 'test':
            if self.lratio is not None:
                self.filenames = [file_name for file_name in self.filenames if self.lratio in file_name]
            if self.lprotocol != 'all':
                if '+' in self.lprotocol:
                    self.filenames = [file_name for file_name in self.filenames if self.lprotocol.split('+')[0] in file_name or self.lprotocol.split('+')[1] in file_name]
                else:
                    self.filenames = [file_name for file_name in self.filenames if self.lprotocol in file_name]

        print(split, len(self.filenames))
        #############################################
        #Randomly remove some of the training columns
        drop=0.5
        sample=False
        if split =='training' and sample:
            grouped_dict = defaultdict(list)
            # Group based on the last three parts of the name
            for key in self.filenames:
                base_key = key.split(',')[0]
                group_key = '-'.join(base_key.split('-')[-3:])
                grouped_dict[group_key].append(key)

            grouped_dict = dict(grouped_dict)
            files_to_remove = set()
            for key in grouped_dict:
                sample_size = int(len(grouped_dict[key]) * drop)  # Keep 20% or 50%
                random_sample = random.sample(grouped_dict[key], sample_size)  # Sample 80% to remove
                files_to_remove.update(random_sample)  # Add to the set of files to remove
            # Remove the sampled files from self.filenames in one pass
            self.filenames = [filename for filename in self.filenames if filename not in files_to_remove]
        print('After random sampling: ', split, len(self.filenames))
        #############################################
        
        ###classification labels###
        self.lprot_names = {'Symmetric': 0, 'Monotonic': 1, 'Collapse_consistent': 2}
        self.lratio_names = {'0.1': 0, '0.2':1, '0.3':2, '0.4': 3, '0.5':4}
        self.boundary_names = {'Fixed':0, 'Flexible':1}
        self.direction_names = {'o': 0, 'm': 0, 'r': 1}
        self.prob_threshold = 0.5
        self.noise_std = 0.01
        self.noise_clip = 0.05
        self.cache = {}
        self.rotation_transform = RandomRotate()
        self.rotation180_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.knn_ids = {}
        # Open the HDF5 file in read mode ('r')
        with h5py.File(os.path.join(self.data_path, self.split+'.h5'), 'r') as hf:
            # Loop through top-level groups (scenarios)
            for scen_name, scenario_group in hf.items():
                # if (self.lratio is not None) and (self.lratio not in scen_name):
                #     continue
                self.cache[scen_name] = {}  # Create a dictionary for this scenario
                # Load processed_center data
                self.cache[scen_name]['processed_center'] = scenario_group['processed_center'][:]  # Access data using [:]
                # Load reserve_capacity data
                self.cache[scen_name]['reserve_capacity'] = scenario_group['reserve_capacity'][:]
                self.cache[scen_name]['column_size'] = scenario_group['column_size'][:]
                # Load scale data (nested structure)
                scale_group = scenario_group['scale']
                self.cache[scen_name]['scale'] = {}
                self.cache[scen_name]['scale']['buckling'] = scale_group['buckling'][:]
                self.cache[scen_name]['scale']['cube'] = scale_group['cube'][:]
                self.cache[scen_name]['scale']['multiple'] = scale_group['multiple'][:]
                self.cache[scen_name]['scale']['sphere'] = scale_group['sphere'][:]
                
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx].split(',')[0]
        data_id = int(self.filenames[idx].split(',')[1])
        direction_id = self.filenames[idx].split(',')[2]
        boundary = str(filename.split('-')[-1])
        ax_load = str(filename.split('-')[-2])
        lprot = str(filename.split('-')[-3])
        section = str(filename.split('-')[0])
        length = str(filename.split('-')[1])
        lprot_label = torch.zeros(1).long()
        lratio_label = torch.zeros(1).long()
        boundary_label = torch.zeros(1).long()
        direction_label = torch.zeros(1).long()
        point_set = self.cache[filename]
        pc_local = np.array(point_set['processed_center'][data_id])
        pc_local_init = np.array(point_set['processed_center'][0])
        column_size = np.array(point_set['column_size'])
        if self.norm is not None:
            if 'mnorm' in self.norm:
                scl = point_set['scale']['multiple'][data_id][:]
                scl_init = point_set['scale']['multiple'][0][:]
            elif 'snorm' in self.norm:
                scl = point_set['scale']['sphere'][data_id][:]
                scl_init = point_set['scale']['sphere'][0][:]
            elif 'cnorm' in self.norm:
                scl = point_set['scale']['cube'][data_id][:]
                scl_init = point_set['scale']['cube'][0][:]
            elif 'bnorm' in self.norm:
                scl = point_set['scale']['buckling'][data_id][:]
                scl_init = point_set['scale']['buckling'][0][:]
            if self.scaling_mode == 'same':
                pc_local = pc_local * scl_init
            else:
                pc_local = pc_local * scl
            pc_local_init = pc_local_init * scl_init
        if 'top' not in self.segment:
            pc_local = pc_local[0, :, :]
            pc_local_init = pc_local_init[0, :, :]
        if 'bottom' not in self.segment:
            pc_local = pc_local[1, :, :]
            pc_local_init = pc_local_init[1, :, :]
        if 'top_bottom' in self.segment:
            pc_local = pc_local.reshape(2*pc_local.shape[1], 3)
            pc_local_init = pc_local_init.reshape(2*pc_local_init.shape[1], 3)
        #apply the augmentation if enabled
        if direction_id == 'r':
            pc_local = np.matmul(self.rotation180_matrix, pc_local.transpose(1,0))
            pc_local = pc_local.transpose(1,0)

            pc_local_init = np.matmul(self.rotation180_matrix, pc_local_init.transpose(1,0))
            pc_local_init = pc_local_init.transpose(1,0)

        elif direction_id == 'm':
            pc_local = np.matmul(self.mirror_matrix, pc_local.transpose(1,0))
            pc_local = pc_local.transpose(1,0)

            pc_local_init = np.matmul(self.mirror_matrix, pc_local_init.transpose(1,0))
            pc_local_init = pc_local_init.transpose(1,0)

        direction_label[0] = self.direction_names[direction_id]        
        lprot_label[0] = self.lprot_names[lprot]
        lratio_label[0] = self.lratio_names[ax_load]
        boundary_label[0] = self.boundary_names[boundary]
        lprot_label_onehot = (torch.nn.functional.one_hot(lprot_label, num_classes=len(self.lprot_names)))[0]#initial shape is (1,3)
        lratio_label_onehot = (torch.nn.functional.one_hot(lratio_label, num_classes=len(self.lratio_names)))[0]#initial shape is (1,5)
        boundary_label_onehot = (torch.nn.functional.one_hot(boundary_label, num_classes=len(self.boundary_names)))[0]#initial shape is (1,2)
        direction_label_onehot = (torch.nn.functional.one_hot(direction_label, num_classes=2))[0]#initial shape is (1,2)
        if direction_id == 'r':    
            reserve_capacity = point_set['reserve_capacity'][data_id][[1,0]]
        else:
            reserve_capacity = point_set['reserve_capacity'][data_id]
        xyz = torch.from_numpy(pc_local).float()
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :self.num_neigh]  # b x n x k
        pc_local_delta = pc_local - pc_local_init
        if self.delta:
            data_item = {'pc_local': torch.cat((torch.from_numpy(pc_local).float(), torch.from_numpy(pc_local_delta).float()), dim=-1), 'rc': torch.from_numpy((reserve_capacity)).float(), 'lprot': lprot, 'ax_load': ax_load, 'boundary': boundary, 'column_size':torch.from_numpy(column_size).float(), 'lprot_label': lprot_label_onehot.float(), 'lratio_label': lratio_label_onehot.float(), 'boundary_label': boundary_label_onehot.float(), 'direction_id': direction_id, 'direction_label': direction_label_onehot.float(), 'knn_idx': knn_idx}
        else:
            if self.delta3d:
                data_item = {'pc_local': torch.from_numpy(pc_local_delta).float(), 'rc': torch.from_numpy((reserve_capacity)).float(), 'lprot': lprot, 'ax_load': ax_load, 'boundary': boundary, 'column_size':torch.from_numpy(column_size).float(), 'lprot_label': lprot_label_onehot.float(), 'lratio_label': lratio_label_onehot.float(), 'boundary_label': boundary_label_onehot.float(), 'direction_id': direction_id, 'direction_label': direction_label_onehot.float(), 'knn_idx': knn_idx}
            else:
                data_item = {'pc_local': torch.from_numpy(pc_local).float(), 'rc': torch.from_numpy((reserve_capacity)).float(), 'lprot': lprot, 'ax_load': ax_load, 'boundary': boundary, 'column_size':torch.from_numpy(column_size).float(), 'lprot_label': lprot_label_onehot.float(), 'lratio_label': lratio_label_onehot.float(), 'boundary_label': boundary_label_onehot.float(), 'direction_id': direction_id, 'direction_label': direction_label_onehot.float(), 'knn_idx': knn_idx}
        return data_item 
        
class PointCloudDatasetTest(Dataset):
    def __init__(self, data_path, segment, norm, lprotocol, gt, process_data=True):
        self.data_path = data_path
        self.segment = segment
        self.norm = norm
        self.lprotocol = lprotocol
        self.use_gt = gt
        self.process_data = process_data
        self.data_array = []
        self.gt_array = []
        self.gt_dict = {}
        self.boundary = []
        self.ax_load = []
        self.lprot = []
        self.lprot_names = {'Symmetric': 0, 'Monotonic': 1, 'Collapse_consistent': 2}
        self.lratio_names = {'0.1': 0, '0.2':1, '0.3':2, '0.4': 3, '0.5':4}
        self.boundary_names = {'Fixed':0, 'Flexible':1}
        self.num_neigh = 32
        if self.use_gt:
            with open(os.path.join(self.data_path, 'info.txt'), 'r') as f:
                for line in f:
                    # Remove leading/trailing whitespace and split by comma
                    info = line.strip().split(', ') #info[0] is the file name of the point cloud
                    self.gt_dict[info[0]] = {'pc_type': info[1], 'lprot': info[4], 'ax_load': info[5], 'boundary':info[6], 'rc+': info[7], 'rc-':info[8]}

        for pc_path in sorted(glob.glob(self.data_path+'/*.npy')):
            if self.use_gt:
                self.gt_array.append(np.array([float(self.gt_dict[pc_path.split('/')[-1]]['rc+']), float(self.gt_dict[pc_path.split('/')[-1]]['rc-'])]))
                self.boundary.append(self.gt_dict[pc_path.split('/')[-1]]['boundary'])
                self.ax_load.append(self.gt_dict[pc_path.split('/')[-1]]['ax_load'])
                self.lprot.append(self.gt_dict[pc_path.split('/')[-1]]['lprot'])
            pc_local = np.load(pc_path, allow_pickle=True)
            if 'norm' in self.norm:
                if 'snorm' in self.norm:
                    scl = sphere_normalization(pc_local)
                    pc_local = pc_local / scl
            self.data_array.append(pc_local)
        self.data_array = np.array(self.data_array)
        #Read ground truth data if the corresponding flag is enabled
        if self.use_gt:
            self.gt_array = np.array(self.gt_array)
        self.data_name = self.data_path.split('/')[-1]
        

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        lprot_label = torch.zeros(1).long()
        lratio_label = torch.zeros(1).long()
        boundary_label = torch.zeros(1).long()
        xyz = torch.from_numpy(self.data_array[idx]).float()
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :self.num_neigh]
        data_item = {'pc_local': torch.from_numpy(self.data_array[idx]).float()}
        direction_id = 'r'
        direction_label_onehot = (torch.nn.functional.one_hot(torch.tensor(1, dtype=torch.long), num_classes=2))[0]#initial shape is (1,2)
        column_size = np.array([884.0000, 404.0000,  24.4000,  43.9000]) #change this to the actual test time column size
        if self.use_gt:
            data_item['rc'] = torch.from_numpy(self.gt_array[idx]).float()
            data_item['boundary'] = self.boundary[idx]
            data_item['ax_load'] = self.ax_load[idx]
            data_item['lprot'] = self.lprot[idx]
            data_item['column_size']=torch.from_numpy(column_size).float()
            lprot_label[0] = self.lprot_names[self.lprot[idx]]
            lratio_label[0] = self.lratio_names[self.ax_load[idx]]
            boundary_label[0] = self.boundary_names[self.boundary[idx]]
            lprot_label_onehot = (torch.nn.functional.one_hot(lprot_label, num_classes=len(self.lprot_names)))[0]#initial shape is (1,3)
            lratio_label_onehot = (torch.nn.functional.one_hot(lratio_label, num_classes=len(self.lratio_names)))[0]#initial shape is (1,3)
            boundary_label_onehot = (torch.nn.functional.one_hot(boundary_label, num_classes=len(self.boundary_names)))[0]#initial shape is (1,3)            
            data_item['lprot_label'] = lprot_label_onehot.float()
            data_item['lratio_label'] = lratio_label_onehot.float()
            data_item['boundary_label'] = boundary_label_onehot.float()
            data_item['knn_idx']= knn_idx
            data_item['direction_id']= direction_id
            data_item['direction_label_onehot']=direction_label_onehot.float()
        return data_item