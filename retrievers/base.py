from abc import ABC, abstractmethod
from PIL import Image
from typing import List


class BaseRetriever(ABC):
    """
    Abstract base class for image retrievers.
    """

    @abstractmethod
    def retrieve(self, prompt: str, k: int) -> List[Image.Image]:
        """
        Retrieve k images based on a text prompt.

        Args:
            prompt (str): Text prompt to guide retrieval.
            k (int): Number of images to retrieve.

        Returns:
            List[Image.Image]: A list of PIL Image objects.
        """
        pass
