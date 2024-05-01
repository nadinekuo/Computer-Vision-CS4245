import torch
import os
import scipy.io
import cv2


class MallDataset(torch.utils.data.Dataset):

    def __init__(self, frames_folder, ground_truth):
        self.frames_folder = os.listdir(frames_folder)
        self.ground_truth = scipy.io.loadmat(ground_truth)['frame'][0]

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join("mall_dataset", "frames", self.frames_folder[index]))
        points = self.ground_truth[index][0][0][0]
        return image, points

    def visualize(self, index):
        image = cv2.imread(os.path.join("mall_dataset", "frames", self.frames_folder[index]))
        for point in self.ground_truth[index][0][0][0]:
            x, y = point
            cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

        cv2.imshow("im", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    frames_directory = os.path.join("mall_dataset", "frames")
    ground_truth = os.path.join("mall_dataset", "mall_gt.mat")
    mall_dataset = MallDataset(frames_directory, ground_truth)
    print(mall_dataset[3])
