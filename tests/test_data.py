# tests/test_data.py

import pytest
from cine_bert.data import load_and_split


def test_load_and_split_default():
    """
    Test that load_and_split reads default YAML and returns correct splits
    """
    train_ds, val_ds, test_ds = load_and_split()
    # Default YAML: test_size=100, val_size=2500
    assert len(test_ds) == 100, f"Expected 100 test samples, got {len(test_ds)}"
    assert len(val_ds) == 2500, f"Expected 2500 validation samples, got {len(val_ds)}"
    # Train split should cover full training set of IMDB (25000)
    assert len(train_ds) == 25000, f"Expected 25000 train samples, got {len(train_ds)}"

    # Check required tokenization fields
    for ds in (train_ds, val_ds, test_ds):
        cols = ds.column_names
        assert "input_ids" in cols, "input_ids column missing"
        assert "attention_mask" in cols, "attention_mask column missing"


def test_split_consistency(tmp_path, monkeypatch):
    """
    Override config values to test custom split sizes.
    """
    # Create temporary YAML config
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
        tokenizer_name: "distilbert-base-uncased"
        model_name: "distilbert-base-uncased"
        num_labels: 2
        train:
          test_size: 10
          val_size: 20
          max_length: 64
        training:
          num_train_epochs: 1
          per_device_train_batch_size: 1
          per_device_eval_batch_size: 1
          logging_steps: 10
          fp16: false
        paths:
          output_dir: "./tmp"
          logs_dir: "./logs"
          models_dir: "./models"
        """
    )
    # Monkeypatch default config path
    monkeypatch.setenv("CONFIG_PATH", str(custom_yaml))

    # Call load_and_split with custom config
    train_ds, val_ds, test_ds = load_and_split(config_path=str(custom_yaml))
    assert len(test_ds) == 10
    assert len(val_ds) == 20
    assert len(train_ds) == 25000  # full train remains
    # Confirm columns
    for ds in (train_ds, val_ds, test_ds):
        assert "input_ids" in ds.column_names
        assert "attention_mask" in ds.column_names
