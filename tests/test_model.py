from onyxgenai.model import ModelClient


def test_base_model_client():
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    assert client.svc_url == svc_url
