import torch
import os
import scipy.io


class MallDataset(torch.utils.data.Dataset):

    def __init__(self, frames_folder, ground_truth):
        self.frames_folder = frames_folder
        self.ground_truth = scipy.io.loadmat(ground_truth)['frame'][0]

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        pass

    def visualize(self, index):
        pass


if __name__ == "__main__":
    frames_directory = os.path.join("mall_dataset", "frames")
    ground_truth = os.path.join("mall_dataset", "mall_gt.mat")
    mall_dataset = MallDataset(frames_directory, ground_truth)
