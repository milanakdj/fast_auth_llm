from engine_factory import EngineFactory
from engines import EasyOCR,DocTrSyntaxEngine,DocTrJsonEngine


def get_factory():
    factory = EngineFactory()
    factory.register("Easy Ocr",EasyOCR())
    factory.register("Doctr",DocTrSyntaxEngine())
    factory.register("Doctr(Json)",DocTrJsonEngine())
    
    return factory
    