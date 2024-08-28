import base64
from io import BytesIO

import requests

from onyxgenai.utils import get_endpoint_url

"""
TODO how to utilize the service ID to route embeddings to a specific instance
and bypass if not present

"""


class Embedder:

    def __init__(
        self, svc_url, model, model_version=1, num_workers=1, collection_name=None
    ) -> None:
        self.svc_url = svc_url
        self.model = model
        self.model_version = model_version
        self.num_workers = num_workers
        self.collection_name = collection_name

    def _onyx_embed(self, batch, media_type, metadata=None):
        print(f"embedding {len(batch)} items")
        print(f"metadata: {metadata}")

        url = get_endpoint_url(self.svc_url, "embedding", media_type)
        data = {
            "data": batch,
            "model_identifier": self.model,
            "model_version": self.model_version,
            "num_workers": self.num_workers,
            "collection_name": self.collection_name,
        }

        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print("Failed to get embedding:", response.status_code, response.text)
            return None

    def _onyx_vector_search(self, query: str, collection_name: str):
        payload = {
            "query_vector": query,
            "collection_name": collection_name,
            "kwargs": {"limit": 3},
        }

        url = get_endpoint_url(self.svc_url, "search")

        response = requests.post(url, json=payload)
        return response.json()

    """
    iterable = list of items
    n = number of items to batch
    """

    def batch(self, iterable, batch_size=1):
        batch_length = len(iterable)
        for ndx in range(0, batch_length, batch_size):
            yield iterable[ndx : min(ndx + batch_size, batch_length)]

    """
    n_workers = number of workers to start
    """

    def start_service(self, n_workers=1):
        pass  # return uid

    """
    uid = service id
    """

    def stop_service(self, uid=None):
        pass

    """
    data = list of strings
    metadata = list of metadata
    batch_size = number of items to embed at once
    return_results = if True then return the results
    """

    def embed_text(
        self, data: list, metadata: list = None, batch_size=None, return_results=True
    ):
        if batch_size is None:
            batch_size = len(data)

        results = []
        if metadata is not None:
            for b, m in self.batch(zip(data, metadata), batch_size):
                result = self._onyx_embed(b, "text", metadata=m)
                if return_results:
                    results.extend(result)
        else:
            for b in self.batch(data, batch_size):
                result = self._onyx_embed(b, "text")
                if return_results:
                    results.extend(result)

        return results

    """
    data = list of base64 encoded images
    metadata = list of metadata
    batch_size = number of items to embed at once
    return_results = if True then return the results
    """

    def embed_images(
        self, data: list, metadata: list = None, batch_size=None, return_results=True
    ):
        if batch_size is None:
            batch_size = len(data)

        encoded = []
        for d in data:
            if isinstance(d) == str:  # we assume this a filepath
                with open(data, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                    encoded.append(encoded_image)
            else:  # assume that it is a PIL image
                buffered = BytesIO()
                d.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue())
                encoded.append(encoded_image)

        results = []
        if metadata is not None:
            for b, m in self.batch(zip(encoded, metadata), batch_size):
                result = self._onyx_embed(b, "image", metadata=m)
                if return_results:
                    results.extend(result)
        else:
            for b in self.batch(encoded, batch_size):
                result = self._onyx_embed(b, "image")
                if return_results:
                    results.extend(result)

        return results

    def vector_search(self, query, collection_name):
        return self._onyx_vector_search(query, collection_name)

    def query_text(self, data, n=5):
        # TODO embed the data
        # todo this will connect to qdrant
        return [f"This is result {i}" for i in range(n)]

    def query_image(self, data, n=5):
        if isinstance(data) == str:  # we assume this a filepath
            with open(data, "rb") as f:
                encoded_image = base64.b64encode(f.read())
        else:  # assume that it is a PIL image
            buffered = BytesIO()
            data.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue())
        # TODO embed the data

        # query qdrant
        pass
