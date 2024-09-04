import base64
from io import BytesIO

import requests


class EmbeddingClient:

    def __init__(
        self, svc_url, model_name, model_version=1, num_workers=1, collection_name=None
    ) -> None:
        self.svc_url = svc_url
        self.model_name = model_name
        self.model_version = model_version
        self.num_workers = num_workers
        self.collection_name = collection_name

    def _onyx_embed(self, batch, media_type):
        if media_type == "text":
            url = f"{self.svc_url}/embedding/text"
        elif media_type == "image":
            url = f"{self.svc_url}/embedding/image"
        else:
            print("Invalid media type")
            return None

        data = {
            "data": batch,
            "model_identifier": self.model_name,
            "model_version": self.model_version,
            "num_workers": self.num_workers,
            "collection_name": self.collection_name,
        }

        response = requests.post(url, json=data)
        if response.status_code == 200:
            response_value = response.json()["embeddings"]
            print("Embedding Successful:", response_value)
            return response_value
        else:
            print("Failed to get embedding:", response.status_code, response.text)
            return None

    def _onyx_vector_search(self, query: str, collection_name: str):
        url = f"{self.svc_url}/vector-store/search"
        payload = {
            "query_vector": query,
            "collection_name": collection_name,
            "kwargs": {"limit": 3},
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

    def batch(self, iterable, batch_size=1):
        batch_length = len(iterable)
        for ndx in range(0, batch_length, batch_size):
            yield iterable[ndx : min(ndx + batch_size, batch_length)]

    def embed_text(self, data: list, batch_size=None, return_results=True):
        if batch_size is None:
            batch_size = len(data)

        results = []
        for b in self.batch(data, batch_size):
            result = self._onyx_embed(b, "text")
            if return_results:
                results.extend(result)

        return results

    def embed_images(self, data: list, batch_size=None, return_results=True):
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
                d.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue())
                encoded.append(encoded_image)

        results = []
        for b in self.batch(encoded, batch_size):
            result = self._onyx_embed(b, "image")
            if return_results:
                results.extend(result)

        return results

    def vector_search(self, query, collection_name):
        return self._onyx_vector_search(query, collection_name)

    def get_collections(self):
        return self._onyx_get_collections()
