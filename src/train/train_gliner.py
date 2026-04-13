import argparse

from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

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

    # Configure training
    model = GLiNER2.from_pretrained(args.model_name)
    config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        warmup_ratio=args.warmup_ratio,
        scheduler_type=args.scheduler_type,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        eval_strategy="epoch",  # Evaluates and saves at end of each epoch
        save_best=True,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        report_to_wandb=bool(args.wandb_project),
        wandb_project=args.wandb_project,
    )

    # Step 6: Train
    trainer = GLiNER2Trainer(model, config)
    results = trainer.train(train_data=args.train_file, eval_data=args.val_file)

    print(f"Training completed!")
    print(f"Best validation loss: {results['best_metric']:.4f}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Training time: {results['total_time_seconds']/60:.1f} minutes")

    # Step 7: Load best model for inference
    # best_model = GLiNER2.from_pretrained(args.output_dir + "/best")
