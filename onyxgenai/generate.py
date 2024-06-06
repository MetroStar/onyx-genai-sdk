'''
TODO how to utilize the service ID to route embeddings to a specific instance
and bypass if not present

Once you start a service that is what is used
'''

class TextGenerator():
    def __init__(self, url, model) -> None:
        self.url = url
        self.model = model


    def generate(self, prompt, context:list = None):
        return f"This is sample response from {self.model}, with prompt of '{prompt}' with context of {context}"

    def start_service(self, n_workers=1):
        pass # return uid # use 

    def stop_service(self, uid=None):
        pass
