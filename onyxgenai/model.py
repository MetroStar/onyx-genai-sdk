import requests


class ModelStore:
    def __init__(self, svc_url) -> None:
        self.svc_url = svc_url

    def _onyx_model_info(self):
        url = f"{self.svc_url}/info/model_info"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            print("Failed to get model info:", response.status_code, response.text)
            return None

    def _onyx_get_deployments(self):
        url = f"{self.svc_url}/serve/deployments"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to get deployment info:", response.status_code, response.text)
            return None

    def _onyx_model_predict(self, deployment_name, data):
        url = f"{self.svc_url}/serve/predict/text"
        payload = {
            "app_name": deployment_name,
            "data": data,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Prediction Successful:", response.json())
            return response.json()["embeddings"][0]
        else:
            print("Prediction Failed:", response.status_code, response.text)
            return None

    def _onyx_model_generate(self, deployment_name, data):
        url = f"{self.svc_url}/serve/generate/text"
        payload = {
            "app_name": deployment_name,
            "messages": [{"role": "user", "content": data}],
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

    def _onyx_model_serve(
        self, model_name, model_version, replicas, options, deployment_name=None
    ):
        url = f"{self.svc_url}/serve/deploy/{model_name}"
        payload = {
            "app_name": deployment_name or model_name,
            "model_version": str(model_version),
            "num_replicas": replicas,
            "ray_actor_options": options,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Deployment Successful:", response.json())
            return model_name
        else:
            print("Deployment Failed:", response.status_code, response.text)
            return None

    def _onyx_model_cleanup(self, deployment_name):
        url = f"{self.svc_url}/serve/cleanup"
        payload = {
            "app_name": deployment_name,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            print("Cleanup Successful:", response.json())
            return response.json()
        else:
            print("Cleanup Failed:", response.status_code, response.text)
            return None

    def get_models(self):
        result = self._onyx_model_info()
        return result

    def get_deployments(self):
        result = self._onyx_get_deployments()
        return result

    def embed_text(self, deployment_name, data):
        result = self._onyx_model_predict(deployment_name, data)
        return result

    def generate_completion(self, deployment_name, data):
        result = self._onyx_model_generate(deployment_name, data)
        return result

    def deploy_model(
        self, model_name, model_version, replicas, options, deployment_name=None
    ):
        result = self._onyx_model_serve(
            model_name, model_version, replicas, options, deployment_name
        )
        return result

    def delete_deployment(self, deployment_name):
        result = self._onyx_model_cleanup(deployment_name)
        return result
