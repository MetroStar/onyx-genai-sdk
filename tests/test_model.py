from unittest.mock import Mock, patch

from onyxgenai.model import ModelClient


def test_base_model_client():
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    assert client.svc_url == svc_url


@patch("onyxgenai.model.requests.get")
def test_get_models(mock_get):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": {"models": ["model1", "model2"]}}
    mock_get.return_value = mock_response

    models = client.get_models()
    assert models == ["model1", "model2"]


@patch("onyxgenai.model.requests.get")
def test_get_deployments(mock_get):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model1": {
            "status": "running",
            "message": "All good",
            "last_deployed_time_s": 1234567890,
            "deployments": {
                "model1": {
                    "status": "running",
                    "status_trigger": "manual",
                    "replica_states": {"RUNNING": 2},
                    "message": "Deployment successful",
                }
            },
        }
    }
    mock_get.return_value = mock_response

    deployments = client.get_deployments()
    assert len(deployments) == 1
    assert deployments[0]["model"] == "model1"
    assert deployments[0]["status"] == "running"


@patch("onyxgenai.model.requests.post")
def test_embed_text(mock_post):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_post.return_value = mock_response

    embeddings = client.embed_text("sample text", "model1")
    assert embeddings == [0.1, 0.2, 0.3]


@patch("onyxgenai.model.requests.post")
def test_generate_completion(mock_post):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "generated_text": [{"content": "Generated text"}]
    }
    mock_post.return_value = mock_response

    generated_text = client.generate_completion(
        "sample prompt", "system prompt", "model1"
    )
    assert generated_text == "Generated text"


@patch("onyxgenai.model.requests.post")
def test_deploy_model(mock_post):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_post.return_value = mock_response

    response = client.deploy_model("model1", 1, 1, {})
    assert response["status"] == "success"


@patch("onyxgenai.model.requests.post")
def test_delete_deployment(mock_post):
    svc_url = "http://localhost:8000"
    client = ModelClient(svc_url)

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "deleted"}
    mock_post.return_value = mock_response

    response = client.delete_deployment("deployment1")
    assert response["status"] == "deleted"
