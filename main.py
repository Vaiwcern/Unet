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

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)
        
        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder path with skip connections
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
        
        # Output layer
        out = self.output(dec1)
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

        # Chuyển nhãn thành tensor (vì nhãn là ảnh grayscale chỉ cần 1 kênh)
        label = transforms.ToTensor()(label)  # Chuyển nhãn thành tensor với 1 kênh

        return image, label


# Hàm huấn luyện với việc in tiến độ
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):  # Lặp qua số lượng epoch
        print("HEHE")
        model.train()  # Chế độ huấn luyện
        print("HEHE")
        running_loss = 0.0  # Biến lưu trữ tổng loss mỗi epoch
        
        print("HEHE")

        for batch_idx, (images, labels) in enumerate(train_loader):
            print("HEHE")
            images, labels = images.cuda(), labels.cuda()  # Di chuyển dữ liệu lên GPU nếu có
            
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
            images, labels = images.cuda(), labels.cuda()  # Di chuyển dữ liệu lên GPU nếu có
            
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
        transforms.Resize(572, 572),             
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    model = UNet(in_channels=3, out_channels=1)  # Nếu ảnh đầu vào có 3 kênh (RGB) và ảnh phân đoạn 1 kênh (grayscale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)  # Di chuyển mô hình lên GPU nếu có, nếu không sẽ sử dụng CPU

    # Định nghĩa loss function và optimizer
    criterion = torch.nn.BCEWithLogitsLoss()  # Đối với binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    
    # Huấn luyện mô hình
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Đánh giá mô hình trên tập test
    evaluate_model(model, test_loader)

