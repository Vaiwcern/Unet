import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

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
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # Áp dụng transform nếu có
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
