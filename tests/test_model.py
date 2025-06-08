# tests/test_model.py

import pytest
import torch
from cine_bert.model import build_model_and_tokenizer
from cine_bert.api.main import load_config


def test_model_forward():
    """
    Test that build_model_and_tokenizer returns a working tokenizer and model
    and that the model produces logits of correct shape.
    """
    # Load configuration
    config = load_config()
    tokenizer, model = build_model_and_tokenizer(
        tokenizer_name=config["tokenizer_name"],
        model_name=config["model_name"],
        num_labels=config["num_labels"]
    )

    # Construct a minimal batch of input text
    inputs = tokenizer(
        "Test sentence",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["train"]["max_length"]
    )

    # Run a forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # logits shape should be (batch_size, num_labels)
    assert outputs.logits.shape == (1, config["num_labels"]), \
        f"Expected logits shape (1, {config['num_labels']}), got {outputs.logits.shape}"
