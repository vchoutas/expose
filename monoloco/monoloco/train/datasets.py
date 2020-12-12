
import json
import torch

from torch.utils.data import Dataset


class KeypointsDataset(Dataset):
    """
    Dataloader fro nuscenes or kitti datasets
    """

    def __init__(self, joints, phase):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        # Define input and output for normal training and inference
        self.inputs_all = torch.tensor(dic_jo[phase]['X'])
        self.outputs_all = torch.tensor(dic_jo[phase]['Y']).view(-1, 1)
        self.names_all = dic_jo[phase]['names']
        self.kps_all = torch.tensor(dic_jo[phase]['kps'])

        # Extract annotations divided in clusters
        self.dic_clst = dic_jo[phase]['clst']

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        kps = self.kps_all[idx, :]

        return inputs, outputs, names, kps

    def get_cluster_annotations(self, clst):
        """Return normalized annotations corresponding to a certain cluster
        """
        inputs = torch.tensor(self.dic_clst[clst]['X'])
        outputs = torch.tensor(self.dic_clst[clst]['Y']).float()
        count = len(self.dic_clst[clst]['Y'])

        return inputs, outputs, count
