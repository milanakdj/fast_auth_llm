import json

class ContextHandler:
    def __init__(self, context_file: str):
        self.context_file = context_file

    def get_context_from_key(self, key=''):

        with open(self.context_file) as f:
            objs = json.load(f)
            if key in objs:
                return objs[key]['Context']
            context_list = [objs[t]['Context'] for t in objs if objs[t].get('is_default')]
            return context_list[0] if context_list else ''
