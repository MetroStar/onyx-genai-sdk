import requests


class ModelClient:
    """Client for interacting with the Onyx Model Store Service
    Args:
        svc_url (str): The URL of the Onyx Model Store Service
    """

    def __init__(
        self,
        svc_url,
    ) -> None:
        self.svc_url = svc_url

    def _onyx_model_info(self):
        url = f"{self.svc_url}/info/model_info"

        response = requests.get(url)
        if response.status_code == 200:
            response_value = response.json()["data"]["models"]
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
            deployment_list = []
            for model, details in response_value.items():
                flattened_deployment = {
                    "model": model,
                    "status": details["status"],
                    "message": details["message"],
                    "last_deployed_time_s": details["last_deployed_time_s"],
                    "deployment_status": details["deployments"][model]["status"],
                    "deployment_status_trigger": details["deployments"][model][
                        "status_trigger"
                    ],
                    "replica_states": details["deployments"][model]["replica_states"][
                        "RUNNING"
                    ],
                    "deployment_message": details["deployments"][model]["message"],
                }
                deployment_list.append(flattened_deployment)

            print("Deployments:", deployment_list)
            return deployment_list
        else:
            print("Failed to get deployment info:", response.status_code, response.text)
            return None

    def _onyx_model_predict(self, data, model_name):
        # Endpoint expects a list of strings
        if isinstance(data, str):
            data = [data]

        url = f"{self.svc_url}/serve/predict/text"
        payload = {
            "app_name": model_name,
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
        self, prompt, system_prompt, model_name, max_new_tokens, temperature, top_p
    ):
        url = f"{self.svc_url}/serve/generate/text"
        payload = {
            "app_name": model_name,
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

    def _onyx_model_generate_text(
        self, messages, model_name, max_new_tokens, temperature, top_p
    ):
        url = f"{self.svc_url}/serve/generate/text"
        payload = {
            "app_name": model_name,
            "messages": messages,
            "kwargs": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if "generated_text" in response.json():
                response_value = response.json()["generated_text"]
                print("Generate Successful:", response_value)
                return response_value
            else:
                print("Generate Failed:", response.status_code, response.text)
                return None
        else:
            print("Generate Failed:", response.status_code, response.text)
            return None

    def _onyx_model_serve(self, model_name, model_version, replicas, options):
        url = f"{self.svc_url}/serve/deploy/{model_name}"
        payload = {
            "app_name": model_name,
            "model_version": str(model_version),
            "num_replicas": replicas,
            "ray_actor_options": options,
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            response_value = response.json()
            print("Deployment Successful:", response_value)
            return response_value
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

    def embed_text(self, data, model_name):
        """Get the embeddings for the input text
        Args:
            data (str): The input text
            model_name (str): The name of the model
        Returns:
            list: The embeddings for the input text
        """

        result = self._onyx_model_predict(data, model_name)
        return result

    def generate_completion(
        self,
        prompt,
        system_prompt="",
        model_name=None,
        max_new_tokens=10000,
        temperature=0.4,
        top_p=0.9,
    ):
        """Generate completion for the prompt
        Args:
            prompt (str): The prompt for completion
            system_prompt (str): The system prompt for completion
            model_name (str): The name of the model
            max_new_tokens (int): The maximum number of tokens to generate
            temperature (float): The temperature for sampling
            top_p (float): The top_p value for sampling
        Returns:
            str: The generated completion text
        """

        result = self._onyx_model_generate(
            prompt, system_prompt, model_name, max_new_tokens, temperature, top_p
        )
        return result

    def generate_text(
        self,
        messages,
        model_name=None,
        max_new_tokens=10000,
        temperature=0.4,
        top_p=0.9,
    ):
        """Generate text based on the input messages
        Args:
            messages (list): The list of messages for the model
            model_name (str): The name of the model
            max_new_tokens (int): The maximum number of tokens to generate
            temperature (float): The temperature for sampling
            top_p (float): The top_p value for sampling
        Returns:
            str: The generated completion text
        """

        result = self._onyx_model_generate_text(
            messages, model_name, max_new_tokens, temperature, top_p
        )
        return result

    def deploy_model(self, model_name, model_version=1, replicas=1, options=None):
        """Deploy the model to the service"""

        result = self._onyx_model_serve(model_name, model_version, replicas, options)
        return result

    def delete_deployment(self, deployment_name):
        """Delete the deployment from the service"""

        result = self._onyx_model_cleanup(deployment_name)
        return result
