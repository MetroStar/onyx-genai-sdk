import requests

from onyxgenai.utils import get_endpoint_url


class ModelStore:
    def __init__(self, svc_url) -> None:
        self.svc_url = svc_url

    def _onyx_model_info(self):
        url = get_endpoint_url(self.svc_url, "info/model_info")
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            print("Failed to get model info:", response.status_code, response.text)
            return None

    def _onyx_model_serve(self, model, model_version, app_name):
        url = get_endpoint_url(self.svc_url, "serve/deploy", model)
        payload = {
            "app_name": app_name,
            "model_version": str(model_version),
            "num_replicas": 2,
            "ray_actor_options": {"num_cpus": 2, "memory": 8000 * 1024 * 1024},
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Deployment Successful:", response.json())
            return app_name
        else:
            print("Deployment Failed:", response.status_code, response.text)
            return None

    def _onyx_model_cleanup(self, app_name):
        url = get_endpoint_url(self.svc_url, "serve/cleanup")
        payload = {
            "app_name": app_name,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Cleanup Successful:", response.json())
            return response.json()
        else:
            print("Cleanup Failed:", response.status_code, response.text)
            return None

    def _onyx_model_predict(self, app_name, data):
        url = get_endpoint_url(self.svc_url, "serve/predict", "text")
        payload = {
            "app_name": app_name,
            "data": data,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Prediction Successful:", response.json())
            return response.json()["embeddings"]
        else:
            print("Prediction Failed:", response.status_code, response.text)
            return None

    def get_models(self):
        result = self._onyx_model_info()
        return result

    def serve_model(self, model, model_version, app_name):
        result = self._onyx_model_serve(model, model_version, app_name)
        return result

    def stop_model(self, app_name):
        result = self._onyx_model_cleanup(app_name)
        return result

    def predict_text(self, app_name, data):
        result = self._onyx_model_predict(app_name, data)
        return result
