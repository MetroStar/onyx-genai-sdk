import requests

from onyxgenai.utils import get_endpoint_url


class ModelStore:
    def __init__(self, svc_url) -> None:
        self.svc_url = svc_url

    def _onyx_model_info(self):
        url = get_endpoint_url(self.svc_url, "info", "model_info")
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            print("Failed to get model info:", response.status_code, response.text)
            return None

    def get_models(self):
        result = self._onyx_model_info()
        return result
