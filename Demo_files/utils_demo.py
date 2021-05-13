import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt


def get_class(label):
    label = label.item()
    if label == 1:
        return "Alive"
    else:
        return "Dead"


# Function to visualize a sequence
def visualize_seq_prediction(sequence, label, prediction, mean_unorm, std_unorm):
    sequence = sequence.squeeze(0)
    cols = 3
    rows = 1
    fig = plt.figure()
    plt.title('Label: ' + get_class(label) + '  Prediction: ' + get_class(prediction), fontweight='bold')
    plt.axis('off')

    for pic_no, pic in enumerate(sequence):
        if pic_no % 3 == 0:

            pic = pic.cpu().numpy()
            pic = std_unorm * pic + mean_unorm
            pic = np.clip(pic, 0, 1)
            pic = pic * 255
            pic = pic.astype(np.uint8)

            if pic_no == 0:
                ind = 1
                plot_name = 'Day before'
            elif pic_no == 3:
                ind = 2
                plot_name = 'Current day'
            else:
                ind = 3
                plot_name = 'Day after'

            fig.add_subplot(rows, cols, ind)
            plt.title(plot_name, fontweight='bold')
            plt.axis('off')
            plt.imshow(pic, cmap='gray')

    plt.show()


class RNN_Dataset(torch.utils.data.Dataset):
    """RNN dataset."""

    def __init__(self, root_dir, seq_lenght, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            seq_lenght: number of images per sequence
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.plain_dataset = torchvision.datasets.ImageFolder(root_dir, transform)
        self.seq_lenght = seq_lenght
        self.classes = self.plain_dataset.classes

    def __len__(self):
        return int(len(self.plain_dataset) / self.seq_lenght)

    def __getitem__(self, idx):
        idx_plain = idx * self.seq_lenght
        image, label = self.plain_dataset.__getitem__(idx_plain)

        for i in range(self.seq_lenght-1):
            idx_plain = idx_plain + 1
            image_i, label_i = self.plain_dataset.__getitem__(idx_plain)
            image = np.concatenate((image, image_i), axis=0)

        composed_sample = [image, label]

        return composed_sample

