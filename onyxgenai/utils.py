def get_endpoint_url(svc_url, path, endpoint):
    if svc_url[-1] == "/":
        svc_url = svc_url[:-1]

    return f"{svc_url}/{path}/{endpoint}"
