import pytest
from redditGen.template import TemplateGen
from redditGen.conf_loader import ConfLoader

_TEST_DATA_PATH = "tests/test_data/"
_EXAMPLES_PATH = _TEST_DATA_PATH + "examples.json"
_EXPECT_TEMPLATE_PATH = _TEST_DATA_PATH + "expected_template.txt"


def test_conf():
    config = ConfLoader().conf
    # print(config["model"], config["fewshot_examples_path"])
    assert config["model"] == "teknium/OpenHermes-2.5-Mistral-7B"
    assert config["fewshot_examples_path"] == "fewshot_examples"


def test_template_gen():
    actual = TemplateGen(
        _EXAMPLES_PATH,
        "r/relationship_advice",
        "My relationship story",
    ).get_template()
    with open(_EXPECT_TEMPLATE_PATH, "r") as file:
        expect = file.read()
    assert expect == actual
