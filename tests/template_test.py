import pytest
from redditGen.template import TemplateGen

def test_template_gen():
    t = TemplateGen().get_template()
    print(t)
