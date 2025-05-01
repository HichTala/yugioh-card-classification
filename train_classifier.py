import evaluate
import numpy as np
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, ToTensor
from transformers import (
    AutoModelForImageClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    AutoImageProcessor
)

from datasets import load_dataset


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def transforms(examples, _transforms):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


def main():
    dataset_draw = load_dataset("datasets/ddraw", split="train[:100]")
    dataset_draw = dataset_draw.train_test_split(test_size=0.2)

    labels = dataset_draw["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[i] = label

    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )

    dataset_draw = dataset_draw.with_transform(
        lambda x: transforms(
            x,
            _transforms
        )
    )
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="runs/draw",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        fp16=True,
        learning_rate=5e-5,
        auto_find_batch_size=True,
        # per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
        push_to_hub=True,
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_draw["train"],
        eval_dataset=dataset_draw["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()
