import json

_TEMPLATE = """
prompt: 
   subreddit: {subreddit}
   topic: {topic}
"""


class TemplateGen:
    def __init__(self, examples_path, subreddit, topic):
        self.template = _TEMPLATE
        self.subreddit = subreddit
        self.topic = topic
        with open(examples_path, "r") as file:
            d = json.load(file)
        self.fewshot_examples = d["examples"]
        # read from /fewshot_examples and insert into template

    def get_template(self):
        res = """
Here are some examples of generated reddit posts and their corresponding prompts.
Each example reflects the tone, depth, and engagement style commonly found in that subreddit.
        """
        for e in self.fewshot_examples:
            res += f"""
prompt: 
   subreddit: {self.subreddit}
   topic: {self.topic}

reddit post result:
    Title: {e["title"]}

    Body: {e["body"]}
📌 *Why this is a good example: {e["rationale"]}.*
        """
        return res + self.template