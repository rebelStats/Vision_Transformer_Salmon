# Based on https://github.com/huggingface/notebooks/blob/master/sagemaker/09_image_classification_vision_transformer/scripts/train.py
# by Philipp Schmid

# Updates:
#    _ Rename 'epochs' hyperparameter to match the one use in the PyTorch Lightning script
#    - Rename 'output-dir' hyperparameter to 'model-dir', and use SageMaker environment variable
#    - Update default learning to 5e-5 for improved accuracy
#    - Add support for test dataset

from transformers import ViTForImageClassification, Trainer, TrainingArguments,default_data_collator,ViTFeatureExtractor
from datasets import load_from_disk,load_metric
import random
import logging
import sys
import argparse
import os
import numpy as np
import subprocess

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    parser.add_argument("--model_dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    valid_dataset = load_from_disk(args.valid_dir)
    test_dataset = load_from_disk(args.test_dir)
    num_classes = train_dataset.features["label"].num_classes

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded valid_dataset length is: {len(valid_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    metric_name = "accuracy"
    # compute metrics function for binary classification

    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # download model from model hub
    model = ViTForImageClassification.from_pretrained(args.model_name,num_labels=num_classes)
    
    # change labels
    id2label =  {key:train_dataset.features["label"].names[index] for index,key in enumerate(model.config.id2label.keys())}
    label2id =  {train_dataset.features["label"].names[index]:value for index,value in enumerate(model.config.label2id.values())}
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.model_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.model_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)