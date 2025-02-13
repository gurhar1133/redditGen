import json

_SCHEMA_PATH = "src/redditGen/training_config.json"

class ConfLoader:
    def __init__(self):
        self.schema_path = _SCHEMA_PATH
        # read config into dict
        with open(self.schema_path, "r") as file:
            self.conf = json.load(file)
    
    