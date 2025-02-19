import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class DRIVE_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Thư mục gốc chứa dữ liệu.
            split (str): 'train' hoặc 'test' để chỉ định phân vùng dữ liệu.
            transform (callable, optional): Hàm transform để áp dụng lên ảnh.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Xác định đường dẫn tới các thư mục chứa ảnh và nhãn
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, '1st_manual')

        # Liệt kê tất cả các file ảnh trong thư mục images
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]

    def __len__(self):
        """
        Trả về số lượng mẫu trong dataset
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Trả về 1 cặp ảnh và nhãn từ dataset
        """
        image_name = self.image_files[idx]

        # Tạo đường dẫn đầy đủ đến ảnh và nhãn
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, image_name)

        # Mở ảnh và nhãn
        image = Image.open(image_path).convert('RGB')  # Ảnh màu
        label = Image.open(label_path).convert('L')  # Nhãn grayscale (1 kênh màu)

        # Áp dụng transform cho ảnh, nhưng không cho nhãn
        if self.transform:
            image = self.transform(image)
            label = transforms.CenterCrop((576, 560))(label)
            label = self.transform(label)  # Chuyển nhãn thành tensor với 1 kênh

        return image, label


# Hàm huấn luyện với việc in tiến độ
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):  # Lặp qua số lượng epoch
        model.train()  # Chế độ huấn luyện
        running_loss = 0.0  # Biến lưu trữ tổng loss mỗi epoch

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Di chuyển dữ liệu lên GPU nếu có
            
            optimizer.zero_grad()  # Reset gradient của optimizer về 0 trước khi tính toán mới
            outputs = model(images)  # Forward pass: chạy ảnh qua mô hình để có output
            
            loss = criterion(outputs, labels)  # Tính loss giữa output và ground truth labels
            loss.backward()  # Backward pass: tính toán gradient của loss với các trọng số của mô hình
            optimizer.step()  # Cập nhật trọng số của mô hình bằng cách sử dụng gradient từ backward pass
            
            running_loss += loss.item()  # Cộng dồn loss của mỗi batch vào running_loss

            # In loss sau mỗi batch
            if batch_idx % 10 == 0:  # In mỗi 10 batch
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # In ra thông tin về epoch sau mỗi lần lặp
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Hàm đánh giá với việc in tiến độ
def evaluate_model(model, test_loader):
    model.eval()  # Chế độ đánh giá (tắt dropout, batch norm)
    all_preds = []  # Danh sách lưu trữ tất cả các dự đoán
    all_labels = []  # Danh sách lưu trữ tất cả các nhãn thực tế
    
    with torch.no_grad():  # Tắt tính toán gradient trong quá trình đánh giá
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)  # Di chuyển dữ liệu lên GPU nếu có
            
            outputs = model(images)  # Forward pass: chạy ảnh qua mô hình để có output
            preds = torch.sigmoid(outputs)  # Chuyển output từ sigmoid thành giá trị [0, 1]
            preds = preds > 0.5  # Áp dụng threshold 0.5 để phân loại thành 2 lớp (background/foreground)
            
            all_preds.append(preds.cpu().numpy())  # Lưu trữ dự đoán về CPU dưới dạng numpy
            all_labels.append(labels.cpu().numpy())  # Lưu trữ nhãn thực tế về CPU dưới dạng numpy
            
            # In tiến độ mỗi 10 batch trong quá trình đánh giá
            if batch_idx % 10 == 0:  # In mỗi 10 batch
                print(f"Evaluating Batch [{batch_idx+1}/{len(test_loader)}]")
        
    # Tính toán IoU (Intersection over Union) cho đánh giá
    all_preds = np.concatenate(all_preds, axis=0)  # Nối tất cả các dự đoán lại thành 1 mảng
    all_labels = np.concatenate(all_labels, axis=0)  # Nối tất cả các nhãn thực tế lại thành 1 mảng
    
    all_preds = all_preds.flatten()  # Chuyển các mảng thành dạng 1 chiều để tính toán
    all_labels = all_labels.flatten()  # Chuyển các nhãn thành dạng 1 chiều để tính toán
    
    # Tính toán IoU (Intersection over Union)
    iou_score = jaccard_score(all_labels, all_preds)  # Dùng chỉ số IoU để đánh giá mô hình
    print(f"Mean IoU: {iou_score:.4f}")


if __name__ == "__main__":
    transform = transforms.Compose([
        # transforms.Resize((576, 576)),             
        transforms.ToTensor(),
    ])

    # Tạo dataset cho train và test
    train_dataset = DRIVE_Dataset(root_dir='DRIVE', split='training', transform=transform)
    test_dataset = DRIVE_Dataset(root_dir='DRIVE', split='test', transform=transform)

    # Tạo DataLoader cho train và test
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Kiểm tra số lượng mẫu trong dataset train
    print(f"Dataset train có {len(train_dataset)} mẫu.")

    # Khởi tạo mô hình U-Net
    model = UNet(in_channels=3, n_classes=1, depth=5, padding=True, up_mode='upconv')  # Nếu ảnh đầu vào có 3 kênh (RGB) và ảnh phân đoạn 1 kênh (grayscale)
    print(device)
    model = model.to(device)  # Di chuyển mô hình lên GPU nếu có, nếu không sẽ sử dụng CPU

    # Định nghĩa loss function và optimizer
    criterion = torch.nn.BCEWithLogitsLoss()  # Đối với binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Huấn luyện mô hình
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Đánh giá mô hình trên tập test
    evaluate_model(model, test_loader)

