from transformers import TrainerCallback

# TODO: remove comments
class EarlyStoppingOnTrainLoss(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience  # Number of steps without improvement before stopping
        self.min_delta = min_delta  # Minimum improvement to count as progress
        self.best_loss = float("inf")  # Track best training loss
        self.counter = 0  # Counter for bad epochs

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens during training."""
        if logs is None or "loss" not in logs:
            return

        train_loss = logs["loss"]  # Get current training loss
        if train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss  # Update best loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1  # Increment counter if no improvement

        # Stop training if patience is exceeded
        if self.counter >= self.patience:
            control.should_training_stop = True
            print(f"🛑 Early stopping triggered at step {state.global_step} due to no improvement in training loss.")


class SaveBestTrainingLossCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")

    def on_step_end(self, args, state, control, **kwargs):
        # Get the latest training loss
        if state.log_history and "loss" in state.log_history[-1]:
            current_loss = state.log_history[-1]["loss"]

            # If it's the best loss so far, save the model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                print(f"🔥 New Best Training Loss: {current_loss:.4f} — Saving Model!")
                kwargs["model"].save_pretrained(args.output_dir)
