import torch
from torchvision import transforms
import utils_demo

# Using CPU or GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/home/lab/PycharmProjects/demo/data'  # path to data folder
input_shape = 80  # sub-image size
seq_lenght = 3
mean = [0.1716, 0.1716, 0.1716]
std = [0.0334, 0.0334, 0.0334]

data_transforms = transforms.Compose([
        transforms.CenterCrop(input_shape),
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

# Call dataset class and dataloader
data = utils_demo.RNN_Dataset(data_dir, seq_lenght, data_transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

# Load the model and set to eval mode
path_model = 'resnet18lstm.pth'
model = torch.load(path_model, map_location=device)
model.eval()

# Make and visualize predictions of the demo dataset
with torch.no_grad():
    for i, (seq, label) in enumerate(data_loader):
        seq = seq.to(device)
        label = label.to(device)
        output = model(seq)
        _, pred = torch.max(output, 1)
        utils_demo.visualize_seq_prediction(seq, label, pred, mean_unorm=0.1716, std_unorm=0.0334)
