from importlib import import_module

from huggingface_hub import HfApi
from transformers import AutoModelForQuestionAnswering

from models.custom_models import electra


def get_model(model_name: str, pretrained):
    api = HfApi()

    if model_name == "electra":
        return electra(pretrained)
    elif model_name in api.list_models():
        model_module = getattr(import_module("transformers"), model_name)
        return model_module.from_pretrained(pretrained)
    else:
        try:
            return AutoModelForQuestionAnswering.from_pretrained(pretrained)
        except:
            raise ValueError(
                f"Model name {model_name} not defined and pretrained path {pretrained} can't be found"
            )


if __name__ == "__main__":
    pass
