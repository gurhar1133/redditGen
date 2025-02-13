
_TEMPLATE = "Subreddit: {subreddit}\nTopic: {topic}\nGenerate a Reddit post title and body."

class TemplateGen:
    def __init__(self):
        self.template = _TEMPLATE
        # read from /fewshot_examples and insert into template

    def get_template(self):
        return self.template