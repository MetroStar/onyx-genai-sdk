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

    def _onyx_get_deployments(self):
        url = get_endpoint_url(self.svc_url, "serve/deployments")
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            print("Failed to get deployment info:", response.status_code, response.text)
            return None

    def _onyx_model_serve(self, model, model_version, replicas, options):
        url = get_endpoint_url(self.svc_url, "serve/deploy", model)
        payload = {
            "app_name": model,
            "model_version": str(model_version),
            "num_replicas": replicas,
            "ray_actor_options": options,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Deployment Successful:", response.json())
            return model
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
            return response.json()["embeddings"][0]
        else:
            print("Prediction Failed:", response.status_code, response.text)
            return None

    def _onyx_model_generate(self, app_name, prompt):
        url = get_endpoint_url(self.svc_url, "serve/generate", "text")
        payload = {
            "app_name": app_name,
            "messages": [{"role": "user", "content": prompt}],
            "kwargs": {"max_new_tokens": 10000, "temperature": 0.4, "top_p": 0.9},
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Generate Successful:", response.json())
            result = response.json()["generated_text"][-1]["content"]
            return result
        else:
            print("Generate Failed:", response.status_code, response.text)
            return None

    def get_models(self):
        result = self._onyx_model_info()
        return result

    def get_deployments(self):
        result = self._onyx_get_deployments()
        return result

    def serve_model(self, model, model_version, replicas, options):
        result = self._onyx_model_serve(model, model_version, replicas, options)
        return result

    def stop_model(self, app_name):
        result = self._onyx_model_cleanup(app_name)
        return result

    def predict_text(self, app_name, data):
        result = self._onyx_model_predict(app_name, data)
        return result

    def generate_text(self, app_name, data):
        result = self._onyx_model_generate(app_name, data)
        return result
