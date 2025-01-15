import torch
import torch.nn as nn

class IterativeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, iterations=3):
        super(IterativeUNet, self).__init__()
        
        self.iterations = iterations
        
        # Define the standard U-Net architecture
        self.unet = UNet(in_channels, out_channels)
        
    def forward(self, x):
        outputs = []
        current_input = x

        for i in range(self.iterations):
            # Apply the standard U-Net to the current input
            segment_output = self.unet(current_input)

            # Add the segment output as input for the next iteration
            current_input = torch.cat([current_input, segment_output], dim=1)  # Concatenate with previous input

            # Store the segment output for the current iteration
            outputs.append(segment_output)

        # Return the final output after all iterations
        return outputs[-1]

# Define the basic U-Net (similar to the previous implementation)
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
