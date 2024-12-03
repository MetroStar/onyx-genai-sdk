from unittest.mock import patch

from onyxgenai.embed import EmbeddingClient


def test_base_embedding_client():
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)

    assert client.svc_url == svc_url


@patch("requests.post")
def test_onyx_embed_text(mock_post):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_response

    data = ["sample text"]
    model_name = "test_model"
    result = client.embed_text(data, model_name)

    assert result == mock_response["embeddings"]


@patch("requests.post")
def test_onyx_embed_image(mock_post):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_response

    data = ["path/to/image.jpg"]
    model_name = "test_model"
    result = client.embed_images(data, model_name)

    assert result == mock_response["embeddings"]


@patch("requests.post")
def test_onyx_vector_search(mock_post):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = {"results": ["result1", "result2", "result3"]}
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_response

    query = "sample query"
    collection_name = "test_collection"
    result = client.vector_search(query, collection_name)

    assert result == mock_response["results"]


@patch("requests.post")
def test_onyx_vector_search_with_kwargs(mock_post):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = {"results": ["result1", "result2", "result3"]}
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_response

    query = "sample query"
    collection_name = "test_collection"
    limit = 10
    query_filter = {"field": "value"}
    result = client.vector_search(query, collection_name, limit, query_filter)

    assert result == mock_response["results"]


@patch("requests.get")
def test_onyx_get_collections(mock_get):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = ["collection1", "collection2"]
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_response

    result = client.get_collections()

    assert result == mock_response


@patch("requests.delete")
def test_onyx_delete_collection(mock_delete):
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)
    mock_response = {"status": "success"}
    mock_delete.return_value.status_code = 200
    mock_delete.return_value.json.return_value = mock_response

    collection_name = "test_collection"
    result = client.delete_collection(collection_name)

    assert result == mock_response
