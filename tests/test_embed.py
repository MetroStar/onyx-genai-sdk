from onyxgenai.embed import EmbeddingClient


def test_base_embedding_client():
    svc_url = "http://localhost:8000"
    client = EmbeddingClient(svc_url)

    assert client.svc_url == svc_url
