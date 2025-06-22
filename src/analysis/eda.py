import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
from collections import Counter
import shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import imagehash
from tqdm import tqdm

class DrowsinessDatasetEDA:
    def __init__(self, dataset_path='dataset'):
        """Initialize the EDA class with dataset path."""
        self.dataset_path = Path(dataset_path)
        self.plots_dir = Path('data/plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')  # Using a specific seaborn style
        sns.set_theme()  # Set seaborn defaults
    
    def get_class_distribution(self):
        """Get the distribution of images across classes."""
        class_counts = {'Drowsy': 0, 'Non Drowsy': 0}
        
        for class_name in class_counts.keys():
            class_path = self.dataset_path / class_name
            if class_path.exists():
                class_counts[class_name] = len(list(class_path.glob('*')))
        
        # Plot class distribution
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Class Distribution in Dataset')
        plt.ylabel('Number of Images')
        plt.savefig(self.plots_dir / 'class_distribution.png')
        plt.close()
        
        return class_counts
    
    def analyze_image_properties(self):
        """Analyze image properties like dimensions, channels, and formats."""
        image_properties = {
            'dimensions': [],
            'channels': [],
            'formats': [],
            'aspect_ratios': []
        }
        
        for class_name in ['Drowsy', 'Non Drowsy']:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue
                
            for img_path in tqdm(list(class_path.glob('*')), desc=f'Analyzing {class_name} images'):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w, c = img.shape
                        image_properties['dimensions'].append((w, h))
                        image_properties['channels'].append(c)
                        image_properties['formats'].append(img_path.suffix)
                        image_properties['aspect_ratios'].append(w/h)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Plot dimension distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        widths, heights = zip(*image_properties['dimensions'])
        plt.scatter(widths, heights, alpha=0.5)
        plt.title('Image Dimensions Distribution')
        plt.xlabel('Width')
        plt.ylabel('Height')
        
        plt.subplot(1, 2, 2)
        plt.hist(image_properties['aspect_ratios'], bins=50)
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Aspect Ratio')
        plt.savefig(self.plots_dir / 'image_properties.png')
        plt.close()
        
        return image_properties
    
    def analyze_pixel_intensities(self):
        """Analyze pixel intensity distributions."""
        intensities = {
            'Drowsy': {'r': [], 'g': [], 'b': []},
            'Non Drowsy': {'r': [], 'g': [], 'b': []}
        }
        
        for class_name in intensities.keys():
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue
                
            sample_images = list(class_path.glob('*'))[:100]  # Analyze first 100 images
            for img_path in tqdm(sample_images, desc=f'Analyzing {class_name} intensities'):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        b, g, r = cv2.split(img)
                        intensities[class_name]['r'].extend(r.flatten())
                        intensities[class_name]['g'].extend(g.flatten())
                        intensities[class_name]['b'].extend(b.flatten())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Plot intensity distributions
        plt.figure(figsize=(15, 5))
        for idx, channel in enumerate(['r', 'g', 'b']):
            plt.subplot(1, 3, idx+1)
            for class_name in intensities.keys():
                plt.hist(intensities[class_name][channel], bins=50, alpha=0.5, label=class_name)
            plt.title(f'{channel.upper()} Channel Distribution')
            plt.legend()
        plt.savefig(self.plots_dir / 'pixel_intensities.png')
        plt.close()
        
        return intensities
    
    def check_image_quality(self):
        """Check for corrupted images and duplicates."""
        quality_issues = {
            'corrupted': [],
            'duplicates': set()
        }
        
        image_hashes = {}
        
        for class_name in ['Drowsy', 'Non Drowsy']:
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue
                
            for img_path in tqdm(list(class_path.glob('*')), desc=f'Checking {class_name} quality'):
                try:
                    # Check if image can be opened
                    img = Image.open(str(img_path))
                    img.verify()
                    
                    # Check for duplicates using perceptual hash
                    img = Image.open(str(img_path))
                    img_hash = str(imagehash.average_hash(img))
                    
                    if img_hash in image_hashes:
                        quality_issues['duplicates'].add((str(img_path), image_hashes[img_hash]))
                    else:
                        image_hashes[img_hash] = str(img_path)
                        
                except Exception as e:
                    quality_issues['corrupted'].append(str(img_path))
        
        return quality_issues
    
    def visualize_samples(self, samples_per_class=5):
        """Visualize sample images from each class."""
        plt.figure(figsize=(15, 5))
        
        for idx, class_name in enumerate(['Drowsy', 'Non Drowsy']):
            class_path = self.dataset_path / class_name
            if not class_path.exists():
                continue
                
            sample_images = list(class_path.glob('*'))[:samples_per_class]
            
            for i, img_path in enumerate(sample_images):
                plt.subplot(2, samples_per_class, idx * samples_per_class + i + 1)
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.axis('off')
                plt.title(class_name)
        
        plt.savefig(self.plots_dir / 'sample_images.png')
        plt.close()
    
    def visualize_augmentations(self):
        """Visualize common augmentation techniques."""
        # Get one sample image
        sample_path = next((self.dataset_path / 'Drowsy').glob('*'))
        img = cv2.imread(str(sample_path))
        
        augmentations = {
            'Original': img,
            'Horizontal Flip': cv2.flip(img, 1),
            'Rotation': cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            'Brightness+': cv2.convertScaleAbs(img, alpha=1.2, beta=10),
            'Brightness-': cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        }
        
        plt.figure(figsize=(20, 4))
        for idx, (aug_name, aug_img) in enumerate(augmentations.items()):
            plt.subplot(1, 5, idx+1)
            plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            plt.title(aug_name)
            plt.axis('off')
        
        plt.savefig(self.plots_dir / 'augmentations.png')
        plt.close()
    
    def run_full_analysis(self):
        """Run all analysis steps and generate a summary report."""
        print("Starting full dataset analysis...")
        
        # 1. Class Distribution
        print("\nAnalyzing class distribution...")
        class_dist = self.get_class_distribution()
        
        # 2. Image Properties
        print("\nAnalyzing image properties...")
        img_props = self.analyze_image_properties()
        
        # 3. Pixel Intensities
        print("\nAnalyzing pixel intensities...")
        intensities = self.analyze_pixel_intensities()
        
        # 4. Quality Issues
        print("\nChecking image quality...")
        quality = self.check_image_quality()
        
        # 5. Visualizations
        print("\nGenerating visualizations...")
        self.visualize_samples()
        self.visualize_augmentations()
        
        # Generate summary report
        report = f"""
        Driver Drowsiness Detection Dataset Analysis Report
        ================================================

        1. Class Distribution:
           - Drowsy: {class_dist['Drowsy']} images
           - Non-drowsy: {class_dist['Non Drowsy']} images
           - Class balance ratio: {class_dist['Drowsy']/class_dist['Non Drowsy']:.2f}

        2. Image Properties:
           - Unique dimensions: {len(set(img_props['dimensions']))}
           - Color channels: {set(img_props['channels'])}
           - File formats: {set(img_props['formats'])}
           - Average aspect ratio: {np.mean(img_props['aspect_ratios']):.2f}

        3. Quality Issues:
           - Corrupted images: {len(quality['corrupted'])}
           - Duplicate pairs: {len(quality['duplicates'])}

        4. Generated Visualizations:
           - Class distribution plot
           - Image properties plots
           - Pixel intensity distributions
           - Sample images from each class
           - Augmentation examples

        Recommendations:
        1. {'Consider data augmentation to balance classes' if abs(1 - class_dist['Drowsy']/class_dist['Non Drowsy']) > 0.2 else 'Classes are relatively balanced'}
        2. {'Standardize image dimensions' if len(set(img_props['dimensions'])) > 1 else 'Image dimensions are consistent'}
        3. {'Handle corrupted images' if quality['corrupted'] else 'No corrupted images found'}
        4. {'Remove or handle duplicates' if quality['duplicates'] else 'No duplicates found'}
        """
        
        # Save report
        with open(self.plots_dir / 'eda_report.txt', 'w') as f:
            f.write(report)
        
        print("\nAnalysis complete! Check the 'data/plots' directory for visualizations and the full report.")
        return report

if __name__ == "__main__":
    eda = DrowsinessDatasetEDA()
    eda.run_full_analysis()