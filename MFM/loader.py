import os
import torch
import numpy as np
from torch.utils.data import Dataset


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


class EEGDataLoader(Dataset):

    def __init__(self, config, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.config = config
        self.dataset = config['dataset']
        self.seq_len = config['seq_len']
        self.n_splits = config['n_splits']
        self.target_idx = config['target_idx']
        self.signal_type = config['signal_type']
        self.data_aug = config['data_aug']
        self.sampling_rate = config['sampling_rate']

        self.dataset_path = os.path.join('./', self.dataset)
        self.inputs, self.labels, self.epochs = self.split_dataset()

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sampling_rate * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx + seq_len]

        # inputs = inputs.reshape(1, n_sample)
        inputs = torch.from_numpy(inputs).float()

        labels = self.labels[file_idx][idx:idx + seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]

        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_aug_root = os.path.join(self.dataset_path, self.data_aug)

        data_fname_list = sorted(os.listdir(data_root))
        data_fname_aug_list = sorted(os.listdir(data_aug_root))

        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dataset)), allow_pickle=True)

        assert len(split_idx_list) == self.n_splits

        if self.dataset == 'sleep-78':
            for i in range(len(data_fname_list)):
            subject_idx = int(data_fname_list[i][3:5])
            if subject_idx == self.fold - 1:
                data_fname_dict['test'].append(data_fname_list[i])
            elif subject_idx in split_idx_list[self.fold - 1]:
                data_fname_dict['val'].append(data_fname_list[i])
            else:
                data_fname_dict['train'].append(data_fname_list[i])
            data_fname_dict['train'].append(data_fname_list[1])
            data_fname_dict['test'].append(data_fname_list[0])
            data_fname_dict['val'].append(data_fname_list[2])

        elif self.dataset == 'MASS' or self.dataset == 'SHHS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.mode]:
                    data_fname_dict[self.mode].append(data_fname_list[i])

        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))


        if self.mode != "test":
            for data_fname in data_fname_dict[self.mode]:
                npz_file = np.load(os.path.join(data_root, data_fname))
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                for i in range(len(npz_file['y']) - self.seq_len + 1):
                    epochs.append([file_idx, i, self.seq_len])
                file_idx += 1
            print('train : ', len(inputs))
            print('train : ', len(inputs[0]))
            print(len(epochs))

        if self.mode == "test":
            for data_fname in data_fname_dict[self.mode]:
                npz_file = np.load(os.path.join(data_root, data_fname))
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                for i in range(len(npz_file['y']) - self.seq_len + 1):
                    epochs.append([file_idx, i, self.seq_len])
                file_idx += 1
            print(len(inputs[0]))
            print('test : ', len(inputs))


        if self.mode=="train":
            npz_file = np.load(os.path.join(data_aug_root,data_fname_aug_list[0]))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            for i in range(len(npz_file['y']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len])
            file_idx += 1
            length=len(inputs)
            print(len(epochs))

            for j in range(length):
                data_augment=permutation(inputs[j])
                data_y=labels[j]
                inputs.append(data_augment)
                labels.append(data_y)
                for i in range(len(data_y) - self.seq_len + 1):
                    epochs.append([file_idx, i, self.seq_len])
                file_idx += 1

        return inputs, labels, epochs
