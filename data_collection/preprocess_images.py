from PIL import Image
import os
import csv


class PosePreprocessor:
    def __init__(self, data_dir, output_csv, label_mapping, img_size=(224, 224)):
        """
        Initializes the preprocessor with required parameters.

        :param data_dir: Directory containing the pose subdirectories.
        :param output_csv: Path to save the labeled dataset CSV.
        :param label_mapping: Dictionary mapping subdirectory names to labels.
        :param img_size: Tuple specifying the resize dimensions for images.
        """
        self.data_dir = data_dir
        self.output_csv = output_csv
        self.label_mapping = label_mapping
        self.img_size = img_size

    def resize_images(self):
        """Resizes all images in the pose directories to the specified size."""
        for pose_dir in self.label_mapping.keys():
            path = os.path.join(self.data_dir, pose_dir)
            if not os.path.exists(path):
                print(f"Directory {path} does not exist!")
                continue

            print(f"Resizing images in {path}...")
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                if str(img_name).endswith((".jpg", ".jpeg", ".png")):
                    img = Image.open(img_path)
                    img = img.resize(self.img_size)
                    img.save(img_path)
        print("Image resizing complete.")

    def generate_labels(self):
        """Generates a CSV file with image paths and corresponding labels."""
        print(f"Generating labels in {self.output_csv}...")
        with open(self.output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "label"])  # Header row

            for pose_dir, label in self.label_mapping.items():
                path = os.path.join(self.data_dir, pose_dir)
                if not os.path.exists(path):
                    print(f"Directory {path} does not exist!")
                    continue

                for img_name in os.listdir(path):
                    if str(img_name).endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(pose_dir, img_name)  # Relative path
                        writer.writerow([img_path, label])

        print(f"Labels generated and saved to {self.output_csv}.")

    def preprocess(self):
        """Runs the full preprocessing pipeline: resizing images and generating labels."""
        self.resize_images()
        self.generate_labels()


if __name__ == "__main__":
    # Configuration
    DATA_DIR = ""  # Replace with your data directory
    OUTPUT_CSV = "dataset_labels.csv"
    LABEL_MAPPING = {
        "frames": 0,
    }
    IMG_SIZE = (224, 224)

    # Preprocessing
    preprocessor = PosePreprocessor(DATA_DIR, OUTPUT_CSV, LABEL_MAPPING, IMG_SIZE)
    preprocessor.preprocess()
