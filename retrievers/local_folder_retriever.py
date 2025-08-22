import os
from PIL import Image
from typing import List
from .base import BaseRetriever

class SimpleRetriever(BaseRetriever):
    """
    A simple retriever that loads the first k images from a local folder.
    """

    def __init__(self, folder_path: str):
        """
        Initializes the retriever with the path to a local folder of images.

        Args:
            folder_path (str): Path to the folder containing image files.
        """
        self.folder_path = folder_path
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a valid directory.")
        self.image_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        if not self.image_files:
            raise ValueError(f"No image files found in '{folder_path}'.")

    def retrieve(self, prompt: str = "", k: int = 4) -> List[Image.Image]:
        """
        Retrieves the first k images from the local folder.

        Args:
            prompt (str): Ignored.
            k (int): Number of images to retrieve.

        Returns:
            List[Image.Image]: List of PIL Image objects.
        """
        selected_files = self.image_files[:k]
        images = []
        for filename in selected_files:
            path = os.path.join(self.folder_path, filename)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
        return images
