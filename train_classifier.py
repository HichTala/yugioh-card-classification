import evaluate
import numpy as np
from timm.data import create_transform
from torch import nn
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification, DefaultDataCollator, TrainingArguments, \
    Trainer, CLIPImageProcessor

from datasets import load_dataset


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions[3]
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def transforms(examples, _transforms):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


def main():
    dataset_draw = load_dataset("datasets/ddraw", split="train")
    dataset_draw = dataset_draw.train_test_split(test_size=0.2)

    labels = dataset_draw["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[i] = label

    checkpoint = "OpenGVLab/internimage_l_22k_384"
    image_processor = CLIPImageProcessor.from_pretrained(checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (120, 89)
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])



    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_classes=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )
    nn.init.constant_(model.model.head.bias, 0)

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
