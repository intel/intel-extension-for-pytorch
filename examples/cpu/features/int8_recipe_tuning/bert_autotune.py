import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex
from sklearn.metrics import accuracy_score


def custom_collate_fn(batch):
    input_keys = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = {k: torch.stack([item[k] for item in batch]) for k in input_keys}
    labels = torch.stack([item["label"] for item in batch])
    return inputs, labels


def load_imdb_dataset(batch_size=16, max_samples=5000):
    # Load the IMDb dataset
    dataset = load_dataset("imdb", split=f"train[:{max_samples}]")

    # Slice the dataset to only use the first `max_samples` samples (optional)
    dataset = load_dataset("imdb", split=f"test[:{max_samples}]")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_token_type_ids=True,
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Convert to PyTorch format
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"]
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=custom_collate_fn
    )
    return dataloader


def quantize_model(model, test_loader):

    ################################ QUANTIZE ##############################  # noqa F401

    model.eval()

    def evaluate_accuracy(model_q):
        test_preds, test_labels = [], []
        for inputs, labels in test_loader:
            with torch.no_grad():
                outputs = model_q(**inputs)
                test_preds.extend(outputs.logits.argmax(-1))
                test_labels.extend(labels)

        return accuracy_score(test_labels, test_preds)

    ######################## recipe tuning with INC ########################  # noqa F401

    # Define quantization configuration using Intel Extension for PyTorch
    qconfig = ipex.quantization.default_dynamic_qconfig
    # Prepare model for quantization
    example_data, _ = next(iter(test_loader))
    prepared_model = ipex.quantization.prepare(
        model, qconfig, example_inputs=example_data, inplace=False
    )

    # # Auto-tune the prepared model using the provided test_loader and the accuracy evaluation function
    tuned_model = ipex.quantization.autotune(
        prepared_model,
        test_loader,
        evaluate_accuracy,
        sampling_sizes=[100],
        accuracy_criterion={"relative": 0.01},
        tuning_time=0,
    )

    ########################################################################  # noqa F401

    # Convert and trace the model with TorchScript
    convert_model = ipex.quantization.convert(tuned_model)

    example_data = (
        example_data["input_ids"],
        example_data["attention_mask"],
        example_data["token_type_ids"],
    )
    traced_model = torch.jit.trace(convert_model, example_data, strict=False)
    traced_model = torch.jit.freeze(traced_model)

    return tuned_model, traced_model


# Main execution
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

test_loader = load_imdb_dataset()
tuned_model, traced_model = quantize_model(model, test_loader)

# Optionally, save the tuned model configuration
tuned_model.save_qconf_summary("tuned_conf.json")

print("Execution finished")
