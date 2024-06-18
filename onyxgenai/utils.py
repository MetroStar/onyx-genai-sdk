def get_endpoint_url(svc_url, media_type):
    if svc_url[-1] == "/":
        svc_url = svc_url[:-1]

    return f"{svc_url}/embedding/{media_type}"
