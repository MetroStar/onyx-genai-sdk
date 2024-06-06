import base64
from io import BytesIO
'''
TODO how to utilize the service ID to route embeddings to a specific instance
and bypass if not present

'''
class Embedder():

    def __init__(self, url, model, vector_store) -> None:
        self.url = url
        self.model = model
        self.vector_store = vector_store

    def batch(self,iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def _onyx_embed(self, batch, model, vector_store=None):
        print(f"embedding {len(batch)} items")
     
        return [[0.0,-1.0,1.0]] * len(batch) # TODO connect to the service


    def start_service(self, n_workers=1):
        pass # return uid

    def stop_service(self, uid=None):
        pass

    '''
    model = model ref
    data = list of data - either file paths or data to be base64 encoded
    vector_store = if not none then persist the embeddigns
    '''
    def embed_text(self, data: list, metadata:list=None, batch_size=None, return_results=True):
        if batch_size == None:
            batch_size = len(data)
        
        results = []
        if metadata is not None:
            for (b, m) in self.batch(zip(data, metadata), batch_size):
                result = self._onyx_embed(b, self.model, self.vector_store, metadata= m)
                if return_results: results.extend(result)
        else:
            for b in self.batch(data, batch_size):
                result = self._onyx_embed(b, self.model, self.vector_store)
                if return_results: results.extend(result)

        return results

    '''
    model = model ref
    data = list of data - either file paths or data to be base64 encoded
    vector_store = if not none then persist the embeddigns
    '''
    def embed_images(self, data: list,  metadata:list=None, batch_size=None, return_results=True):
        if batch_size == None:
            batch_size = len(data)
        encoded =[]
        for d in data:
            if type(d) == str: # we assume this a filepath
                with open(data, "rb") as f:
                    encoded_image = base64.b64encode(f.read())
                    encoded.append(encoded_image)
            else: # assume that it is a PIL image
                buffered = BytesIO()
                d.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue())
                encoded.append(encoded_image)
      

        for b in self.batch(encoded, batch_size):
            self._onyx_embed(b, self.model, self.vector_store)


    # d
    def query_text(self, data, n=5):
        #TODO embed the data
        # todo this will connect to qdrant
        return [f"This is result {i}" for i in range(n)]
    
    def query_image(self, data, n=5):
        if type(data) == str: # we assume this a filepath
            with open(data, "rb") as f:
                encoded_image = base64.b64encode(f.read())
        else: # assume that it is a PIL image
            buffered = BytesIO()
            data.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue())
        #TODO embed the data

        
        #query qdrant
        pass
    

    def list_models(self):
        #todo call to service
        return [
            {"model_id":"abcdefg",
             "model_description":"this model does images",
             "embed_dim":512}
        ]


