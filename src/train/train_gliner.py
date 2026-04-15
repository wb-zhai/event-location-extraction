import argparse
import json
import logging

from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from src.data.utils import collect_schema
from src.train.eval import EventArgumentExtractionEvaluatorGliNER2

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Train GLiNER2 on a custom NER dataset"
    )
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="fastino/gliner2-base-v1",
        help="Pretrained model name or path (default: fastino/gliner2-base-v1)",
    )
    arg_parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to the training dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="Path to the validation dataset file (JSONL format)",
    )
    arg_parser.add_argument(
        "--schema_file",
        type=str,
        help="Path to the schema file (JSON format)",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the trained model and checkpoints",
    )
    arg_parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the training experiment (used for logging and checkpointing)",
    )
    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    arg_parser.add_argument(
        "--eval_batch_size",
        type=int,
        help="Batch size for evaluation (default: same as --batch_size)",
    )
    arg_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model parameters (default: 1)",
    )
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)",
    )
    arg_parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-5,
        help="Learning rate for the encoder (default: 1e-5)",
    )
    arg_parser.add_argument(
        "--task_lr",
        type=float,
        default=5e-4,
        help="Learning rate for the task-specific layers (default: 5e-4)",
    )
    arg_parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduling (default: 0.1)",
    )
    arg_parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts"],
        help="Type of learning rate scheduler (default: cosine)",
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Precision for training (default: bf16)",
    )
    arg_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement after which training will be stopped (default: 3)",
    )
    arg_parser.add_argument(
        "--wandb_project",
        type=str,
        help="Weights & Biases project name for logging (optional)",
    )

    args = arg_parser.parse_args()

    fp16_enabled = False
    bf16_enabled = False
    if args.precision == "fp16":
        fp16_enabled = True
    elif args.precision == "bf16":
        bf16_enabled = True

    train_dataset = TrainingDataset.load(args.train_file)
    val_dataset = TrainingDataset.load(args.val_file)

    if args.schema_file:
        logger.info(f"Loading schema from {args.schema_file}")
        with open(args.schema_file, "r") as f:
            schema = json.load(f)
        event_types = list(schema["macro_trigger_types"].keys())
        argument_types = list(schema["roles"].keys())
    else:
        logger.info("Collecting schema from training and validation datasets")
        event_types, argument_types = collect_schema([train_dataset, val_dataset])

    evaluator = EventArgumentExtractionEvaluatorGliNER2(
        event_types=event_types,
        argument_types=argument_types,
        batch_size=args.eval_batch_size or args.batch_size,
        num_workers=args.num_workers,
    )

    # Configure training
    model = GLiNER2.from_pretrained(args.model_name)
    config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        warmup_ratio=args.warmup_ratio,
        scheduler_type=args.scheduler_type,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        eval_strategy="steps",  # Evaluates and saves at end of each epoch
        eval_steps=100,
        metric_for_best="overall_event_arg_f1",
        save_best=True,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        report_to_wandb=bool(args.wandb_project),
        wandb_project=args.wandb_project,
    )

    # Step 6: Train
    trainer = GLiNER2Trainer(model, config, compute_metrics=evaluator)
    results = trainer.train(train_data=train_dataset, eval_data=val_dataset)

    print("Training completed!")
    print(f"Best validation loss: {results['best_metric']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Training time: {results['total_time_seconds']/60:.1f} minutes")

    # Step 7: Load best model for inference
    # best_model = GLiNER2.from_pretrained(args.output_dir + "/best")
