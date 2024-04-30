import os
import h5py
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def sphere_normalization(pc):
    centroid = np.mean(pc, axis=0)
    pc_c = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_c ** 2, axis=1)))
    # pc_normalized = pc / m
    return m

class PointCloudDataset(Dataset):
    def __init__(self, data_path, segment, norm, lprotocol, split='training', process_data=True):
        self.data_path = data_path
        self.segment = segment
        self.norm = norm
        self.lprotocol = lprotocol
        self.split = split
        self.process_data = process_data
        self.file = os.path.join(self.data_path, self.split+'.txt')
        self.filenames = [line.rstrip() for line in open(self.file)]
        self.cache = {}
        # Open the HDF5 file in read mode ('r')
        with h5py.File(os.path.join(self.data_path, self.split+'.h5'), 'r') as hf:
            
            # Loop through top-level groups (scenarios)
            for scen_name, scenario_group in hf.items():
                self.cache[scen_name] = {}  # Create a dictionary for this scenario
                # Load processed_center data
                self.cache[scen_name]['processed_center'] = scenario_group['processed_center'][:]  # Access data using [:]
                # Load reserve_capacity data
                self.cache[scen_name]['reserve_capacity'] = scenario_group['reserve_capacity'][:]
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
        boundary = str(filename.split('-')[-1])
        ax_load = str(filename.split('-')[-2])
        lprot = str(filename.split('-')[-3])
        point_set = self.cache[filename]
        pc_local = np.array(point_set['processed_center'][data_id])
        if self.norm is not None:
            if 'mnorm' in self.norm:
                scl = point_set['scale']['multiple'][data_id][:]
            elif 'snorm' in self.norm:
                scl = point_set['scale']['sphere'][data_id][:]
            elif 'cnorm' in self.norm:
                scl = point_set['scale']['cube'][data_id][:]
            elif 'bnorm' in self.norm:
                scl = point_set['scale']['buckling'][data_id][:]
            pc_local = pc_local * scl
        if 'top' not in self.segment:
            pc_local = pc_local[0, :, :]
        if 'bottom' not in self.segment:
            pc_local = pc_local[1, :, :]
        if 'top_bottom' in self.segment:
            pc_local = pc_local.reshape(2*pc_local.shape[1], 3)
        pc_local = pc_local.transpose(1,0)
        data_item = {'pc_local': torch.from_numpy(pc_local).float(), 'rc': torch.from_numpy((point_set['reserve_capacity'][data_id])).float(), 'lprot': lprot, 'ax_load': ax_load, 'boundary': boundary}
        return data_item 
        
class PointCloudDatasetTest(Dataset):
    def __init__(self, data_path, segment, norm, lprotocol, process_data=True):
        self.data_path = data_path
        self.segment = segment
        self.norm = norm
        self.lprotocol = lprotocol
        self.process_data = process_data
        self.data_array = []
        for pc_path in sorted(glob.glob(self.data_path+'/*.npy')):
            pc_local = np.load(pc_path, allow_pickle=True)
            if 'norm' in self.norm:
                if 'snorm' in self.norm:
                    scl = sphere_normalization(pc_local)
                    pc_local = pc_local / scl
            pc_local = (pc_local).transpose(1,0)
            self.data_array.append(pc_local)
        self.data_array = np.array(self.data_array)
        self.data_name = self.data_path.split('/')[-1]

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data_array[idx]).float()