from engine import Engine

class EngineFactory:
        def __init__(self):
                self.engines={}
        def register(self,key:str,engine:Engine):
                self.engines[key]=engine
        def get_engine(self,key):
                return self.engines[key]
        def registered_keys(self):
                return self.engines.keys()
                
        