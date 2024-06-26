def get_endpoint_url(svc_url, path, item_id=None):
    if svc_url[-1] == "/":
        svc_url = svc_url[:-1]

    if item_id:
        return f"{svc_url}/{path}/{item_id}"
    else:
        return f"{svc_url}/{path}"
