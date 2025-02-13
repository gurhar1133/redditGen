import torch
import pandas as pd
import callbacks
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from redditGen.template import TemplateGen
# TODO: remove unnecessary imports

FINE_TUNED_MODEL_PATH = ""


class RedditLLMChain:
    def __init__(self):
        self.model_path = FINE_TUNED_MODEL_PATH
        self.tokenier = AutoTokenizer.from_pretrained(model_path)
        # Check and set pad_token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"Model is loaded on device: {model.device}")

        # Create text generation pipeline with model and tokenizer
        self.text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=800,
            temperature=0.8,
            do_sample=True,
            top_p=0.5,
            top_k=500,
        )
        # Wrap the pipeline in LangChain using HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=self.text_generator)
        self.full_prompt = PromptTemplate(
            template=TemplateGen().get_template(),
            input_variables=["subreddit", "topic"]
        )
        self.llm_chain = LLMChain(prompt=self.full_prompt, llm=llm)

    def generate_post(self, subreddit, topic):
        return self.llm_chain.invoke({"subreddit": subreddit, "topic": topic})["text"]


def main():
    # TODO: reads text files that have prompts and
    # generate an array of {prompt: result}
    subreddit = "r/relationship_advice"
    topic = "I am seeking relationship advice, here is my funny predicament"
    r_chain = RedditLLMChain()
    result = r_chain.generate_post(subreddit, topic)
    print(result)


if __name__ == "__main__":
    main()
