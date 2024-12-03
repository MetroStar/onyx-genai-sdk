import base64
from io import BytesIO

import requests


class EmbeddingClient:
    """
    A client for interacting with the Onyx Embedding Service.
    Args:
        svc_url (str): The URL of the Onyx Embedding Service
    """

    def __init__(
        self,
        svc_url,
    ) -> None:
        self.svc_url = svc_url

    def _onyx_embed(
        self, batch, media_type, model_name, model_version, num_workers, collection_name
    ):
        if media_type == "text":
            url = f"{self.svc_url}/embedding/text"
        elif media_type == "image":
            url = f"{self.svc_url}/embedding/image"
        else:
            print("Invalid media type")
            return None

        data = {
            "data": batch,
            "model_identifier": model_name,
            "model_version": model_version,
            "num_workers": num_workers,
            "collection_name": collection_name,
        }

        response = requests.post(url, json=data)
        if response.status_code == 200:
            response_value = response.json()["embeddings"]
            print("Embedding Successful:", response_value)
            return response_value
        else:
            print("Failed to get embedding:", response.status_code, response.text)
            return None

    def _onyx_vector_search(
        self, query: str, collection_name: str, limit: int, query_filter=None
    ):
        url = f"{self.svc_url}/vector-store/search"
        payload = {
            "query_vector": query,
            "collection_name": collection_name,
            "kwargs": {"limit": limit, "query_filter": query_filter},
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if "results" in response.json():
                response_value = response.json()["results"]
                print("Search Successful:", response_value)
                return response_value
            else:
                print("No search results found")
                return None
        else:
            print("Failed to get search results:", response.status_code, response.text)
            return None

    def _onyx_get_collections(self):
        url = f"{self.svc_url}/vector-store/collections"
        response = requests.get(url)
        if response.status_code == 200:
            response_value = response.json()
            print("Collections:", response_value)
            return response_value
        else:
            print("Failed to get collections:", response.status_code, response.text)
            return None

    def _onyx_delete_collection(self, collection_name):
        url = f"{self.svc_url}/vector-store/collections/{collection_name}"
        response = requests.delete(url)
        if response.status_code == 200:
            response_value = response.json()
            print("Collection deleted:", response_value)
            return response_value
        else:
            print("Failed to delete collection:", response.status_code, response.text)
            return None

    def batch(self, iterable, batch_size=1):
        """
        Batch an iterable into chunks of size batch_size
        Args:
            iterable (iterable): The iterable to batch
            batch_size (int): The size of the batches
        Returns:
            generator: A generator that yields batches of the iterable
        """

        batch_length = len(iterable)
        for ndx in range(0, batch_length, batch_size):
            yield iterable[ndx : min(ndx + batch_size, batch_length)]

    def embed_text(
        self,
        data: list,
        model_name,
        model_version=1,
        num_workers=1,
        collection_name=None,
        batch_size=None,
        return_results=True,
    ):
        """
        Get the embeddings for the input text
        Args:
            data (list): The input text
            model_name (str): The name of the model
            model_version (int): The version of the model
            num_workers (int): The number of workers
            collection_name (str): The name of the collection
            batch_size (int): The size of the batches
            return_results (bool): Whether to return the results
        Returns:
            list: The embeddings for the input text
        """

        if batch_size is None:
            batch_size = len(data)

        results = []
        for b in self.batch(data, batch_size):
            result = self._onyx_embed(
                b, "text", model_name, model_version, num_workers, collection_name
            )
            if return_results:
                results.extend(result)

        return results

    def embed_images(
        self,
        data: list,
        model_name,
        model_version=1,
        num_workers=1,
        collection_name=None,
        batch_size=None,
        return_results=True,
    ):
        """
        Get the embeddings for the input images
        Args:
            data (list): The input images
            batch_size (int): The size of the batches
            return_results (bool): Whether to return the results
        Returns:
            list: The embeddings for the input images
        """

        if batch_size is None:
            batch_size = len(data)

        encoded = []
        for d in data:
            if d is str:  # we assume this a filepath
                with open(data, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                    encoded.append(encoded_image)
            else:  # assume that it is a PIL image
                buffered = BytesIO()
                encoded_image = base64.b64encode(buffered.getvalue())
                encoded.append(encoded_image)

        results = []
        for b in self.batch(encoded, batch_size):
            result = self._onyx_embed(
                b, "image", model_name, model_version, num_workers, collection_name
            )
            if return_results:
                results.extend(result)

        return results

    def vector_search(self, query, collection_name, limit=3, query_filter=None):
        """
        Search for vectors in the collection
        Args:
            query (str): The query vector
            collection_name (str): The name of the collection
            limit (int): The number of results to return (default 3)
            query_filter (dict): The query filter (default None)
        Returns:
            list: The vector search results
        """
        return self._onyx_vector_search(query, collection_name, limit, query_filter)

    def get_collections(self):
        """
        Get the list of collections available in the service
        Returns:
            list: The list of collections
        """
        return self._onyx_get_collections()

    def delete_collection(self, collection_name):
        """
        Delete a collection
        Args:
            collection_name (str): The name of the collection
        Returns:
            dict: The response from the server
        """
        return self._onyx_delete_collection(collection_name)
