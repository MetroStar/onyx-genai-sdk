import requests


class ModelClient:
    """Client for interacting with the Onyx Model Store Service
    Args:
        svc_url (str): The URL of the Onyx Model Store Service
        model_name (str): The name of the model to deploy
        model_version (int): The version of the model to deploy
        replicas (int): The number of replicas to deploy
        deployment_name (str): The name of the deployment
        options (dict): The options for deploying the model
    """

    def __init__(
        self,
        svc_url,
        model_name=None,
        model_version=1,
        replicas=1,
        deployment_name=None,
        options=None,
    ) -> None:
        self.svc_url = svc_url
        self.model_name = model_name
        self.model_version = model_version
        self.replicas = replicas
        self.deployment_name = deployment_name
        self.options = options

    def _get_deployment_name(self):
        if self.deployment_name:
            return self.deployment_name
        else:
            return self.model_name

    def _onyx_model_info(self):
        url = f"{self.svc_url}/info/model_info"

        response = requests.get(url)
        if response.status_code == 200:
            response_value = response.json()["data"]
            print("Model Info:", response_value)
            return response_value
        else:
            print("Failed to get model info:", response.status_code, response.text)
            return None

    def _onyx_get_deployments(self):
        url = f"{self.svc_url}/serve/deployments"

        response = requests.get(url)
        if response.status_code == 200:
            response_value = response.json()
            print("Deployments:", response_value)
            return response_value
        else:
            print("Failed to get deployment info:", response.status_code, response.text)
            return None

    def _onyx_model_predict(self, data):
        url = f"{self.svc_url}/serve/predict/text"
        payload = {
            "app_name": self._get_deployment_name(),
            "data": data,
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if "embeddings" in response.json():
                response_value = response.json()["embeddings"][0]
                print("Prediction Successful:", response_value)
                return response_value
            else:
                print("Prediction Failed:", response.status_code, response.text)
                return None
        else:
            print("Prediction Failed:", response.status_code, response.text)
            return None

    def _onyx_model_generate(
        self, prompt, system_prompt, max_new_tokens, temperature, top_p
    ):
        url = f"{self.svc_url}/serve/generate/text"
        payload = {
            "app_name": self._get_deployment_name(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "kwargs": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if "generated_text" in response.json():
                response_value = response.json()["generated_text"][-1]["content"]
                print("Generate Successful:", response_value)
                return response_value
            else:
                print("Generate Failed:", response.status_code, response.text)
                return None
        else:
            print("Generate Failed:", response.status_code, response.text)
            return None

    def _onyx_model_serve(self):
        url = f"{self.svc_url}/serve/deploy/{self.model_name}"
        payload = {
            "app_name": self._get_deployment_name(),
            "model_version": str(self.model_version),
            "num_replicas": self.replicas,
            "ray_actor_options": self.options,
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            response_value = response.json()
            print("Deployment Successful:", response_value)
            return response_value
        else:
            print("Deployment Failed:", response.status_code, response.text)
            return None

    def _onyx_model_cleanup(self):
        url = f"{self.svc_url}/serve/cleanup"
        payload = {
            "app_name": self._get_deployment_name(),
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            response_value = response.json()
            print("Cleanup Successful:", response_value)
            return response_value
        else:
            print("Cleanup Failed:", response.status_code, response.text)
            return None

    def get_models(self):
        """Get the list of models available in the service
        Returns:
            list: The list of models
        """

        result = self._onyx_model_info()
        return result

    def get_deployments(self):
        """Get the list of deployments available in the service
        Returns:
            list: The list of deployments
        """

        result = self._onyx_get_deployments()
        return result

    def embed_text(self, data):
        """Get the embeddings for the input text
        Args:
            data (str): The input text
        Returns:
            list: The embeddings for the input text
        """

        result = self._onyx_model_predict(data)
        return result

    def generate_completion(
        self,
        prompt="""""",
        system_prompt=None,
        max_new_tokens=10000,
        temperature=0.4,
        top_p=0.9,
    ):
        """Generate completion for the prompt
        Args:
            prompt (str): The prompt for completion
            system_prompt (str): The system prompt for completion
            max_new_tokens (int): The maximum number of tokens to generate
            temperature (float): The temperature for sampling
            top_p (float): The top_p value for sampling
        Returns:
            str: The generated completion text
        """

        result = self._onyx_model_generate(
            prompt, system_prompt, max_new_tokens, temperature, top_p
        )
        return result

    def deploy_model(self):
        """Deploy the model to the service"""

        result = self._onyx_model_serve()
        return result

    def delete_deployment(self):
        """Delete the deployment from the service"""

        result = self._onyx_model_cleanup()
        return result
