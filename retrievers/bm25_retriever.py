import os
from typing import List
from PIL import Image
from pyserini.search.lucene import LuceneSearcher
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    A retriever class that uses Pyserini and BM25 to retrieve image paths from a Lucene index
    given a textual prompt, and loads them as PIL.Image.Image objects.
    """

    def __init__(self, index_path: str, image_base_path: str = ""):
        """
        Initializes the Pyserini retriever.

        Args:
            index_path (str): Path to the Lucene index.
            image_base_path (str): Base path to locate images (if stored locally).
                                   If the Lucene doc contains relative paths, this
                                   should point to the root directory containing images.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Lucene index not found at: {index_path}")
        self.searcher = LuceneSearcher(index_path)
        self.image_base_path = image_base_path
        print("PyseriniRetriever initialized with index:", index_path)

    def retrieve(self, prompt: str, k: int = 4) -> List[Image.Image]:
        """
        Retrieve the top-k images using Pyserini search.

        Args:
            prompt (str): The natural language prompt/query.
            k (int): Number of top images to retrieve.

        Returns:
            List[PIL.Image.Image]: List of PIL Image objects.
        """
        hits = self.searcher.search(prompt.lower, k=k)
        pil_images = []

        for i, hit in enumerate(hits):
            try:
                image_path = hit.raw.strip().strip('"')
                full_path = os.path.join(self.image_base_path, image_path)
                img = Image.open(full_path).convert("RGB")
                pil_images.append(img)
                print(f"[{i + 1}] Loaded image from: {full_path}")
            except Exception as e:
                print(f"Failed to load image from {full_path}: {e}")

        return pil_images