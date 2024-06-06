
from abc import ABC

class Engine(ABC):
    def process(self,filename:str)->str:
        ...