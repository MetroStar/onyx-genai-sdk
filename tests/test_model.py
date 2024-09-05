from onyxgenai.model import ModelClient


def test_base_model_client():
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    assert client.svc_url == svc_url
    assert client.model_name is None
    assert client.model_version == 1
    assert client.replicas == 1
    assert client.deployment_name is None
    assert client.options is None


def test_full_model_client():
    svc_url = "http://localhost:8000"
    model_name = "test_model"
    model_version = 2
    replicas = 4
    deployment_name = "test_deployment"
    options = {"option1": "value1"}
    client = ModelClient(
        svc_url, model_name, model_version, replicas, deployment_name, options
    )

    assert client.svc_url == svc_url
    assert client.model_name == model_name
    assert client.model_version == 2
    assert client.replicas == 4
    assert client.deployment_name == deployment_name
    assert client.options == options
