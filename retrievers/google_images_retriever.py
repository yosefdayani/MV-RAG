import io
import requests
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .base import BaseRetriever


class GoogleRetriever(BaseRetriever):
    """
    A class to retrieve images from Google Images using the Custom Search JSON API.
    """

    def __init__(self, api_key: str, cx_id: str):
        """
        Initializes the Retriever with Google Images retrieval capabilities.

        Args:
            api_key (str): Your Google Cloud API key.
                           You can obtain this from the Google Cloud Console.
            cx_id (str): Your Custom Search Engine ID.
                         This is obtained after setting up a Programmable Search Engine
                         and enabling image search.
        """
        self.api_key = api_key
        self.cx_id = cx_id
        try:
            self.service = build("customsearch", "v1", developerKey=self.api_key)
            print("Retriever initialized successfully with Google Custom Search API.")
        except Exception as e:
            print(f"Error initializing Google Custom Search API service: {e}")
            self.service = None

    def retrieve(self, prompt: str, k: int = 4) -> list[Image.Image]:
        """
        Retrieves a list of PIL (Pillow) images from Google Images based on a prompt.

        Args:
            prompt (str): The search query for images (e.g., "BMW 319").
            k (int): The maximum number of images to retrieve.
                     Note: The Google Custom Search API returns a maximum of 10
                     results per request. This method handles pagination to
                     retrieve up to 'k' images.

        Returns:
            list[PIL.Image.Image]: A list of PIL Image objects. Returns an empty
                                   list if no images are found or an error occurs.
        """
        if not self.service:
            print("API service not initialized. Cannot retrieve images.")
            return []

        pil_images = []
        results_per_page = 10
        current_start_index = 1

        while len(pil_images) < k:
            try:
                num_to_fetch = min(k - len(pil_images), results_per_page)
                res = self.service.cse().list(
                    q=prompt,
                    cx=self.cx_id,
                    searchType="image",
                    num=num_to_fetch,
                    start=current_start_index
                ).execute()

                if 'items' not in res:
                    print(f"No more image results found for '{prompt}'.")
                    break

                for item in res['items']:
                    if 'link' in item and item.get('mime', '').startswith('image/'):
                        image_url = item['link']
                        try:
                            response = requests.get(image_url, stream=True, timeout=5)
                            response.raise_for_status()

                            image_data = io.BytesIO(response.content)
                            pil_image = Image.open(image_data)
                            pil_images.append(pil_image)
                            print(f"Successfully retrieved image from: {image_url}")

                        except requests.exceptions.RequestException as req_err:
                            print(f"Error downloading image from {image_url}: {req_err}")
                        except Image.UnidentifiedImageError:
                            print(f"Could not identify image format for {image_url}. Skipping.")
                        except Exception as img_err:
                            print(f"An unexpected error occurred while processing image {image_url}: {img_err}")
                        finally:
                            if 'response' in locals() and response:
                                response.close()

                    if len(pil_images) >= k:
                        break
                next_page_info = res.get('queries', {}).get('nextPage')
                if next_page_info and next_page_info[0].get('startIndex'):
                    current_start_index = next_page_info[0]['startIndex']
                else:
                    print(f"Reached end of available results for '{prompt}'.")
                    break

            except HttpError as http_err:
                print(f"Google Custom Search API error: {http_err}")
                print("Please check your API key, CX ID, and daily query limits.")
                break
            except Exception as e:
                print(f"An unexpected error occurred during image retrieval: {e}")
                break

        return pil_images