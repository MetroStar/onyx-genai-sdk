from onyxgenai.embed import EmbeddingClient


def test_base_embedding_client():
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)

    assert client.svc_url == svc_url
    assert client.model_name is None
    assert client.model_version == 1
    assert client.num_workers == 1
    assert client.collection_name is None


def test_full_embedding_client():
    svc_url = "http://localhost:8000"
    model_name = "test_model"
    model_version = 2
    num_workers = 4
    collection_name = "test_collection"
    client = EmbeddingClient(
        svc_url, model_name, model_version, num_workers, collection_name
    )

    assert client.svc_url == svc_url
    assert client.model_name == model_name
    assert client.model_version == 2
    assert client.num_workers == 4
    assert client.collection_name == collection_name
