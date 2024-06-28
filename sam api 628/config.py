import json


def import_config():
    config = None
    with open("config.json") as user_file:
        file_contents = user_file.read()
        config = json.loads(file_contents)
    return config
