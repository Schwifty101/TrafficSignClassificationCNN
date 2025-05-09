# trainer.py
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from model_builder import Parameters, model_pass, get_time_hhmmss, print_progress, TrafficSignModel
from data_loader import load_pickled_data
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Use TF2.x compatibility
print("Using TensorFlow version:", tf.__version__)

# Configure TensorFlow to use both CPU and GPU efficiently
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

# Set TensorFlow to use mixed precision for better GPU performance
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to mixed_float16")
except Exception as e:
    print(f"Could not set mixed precision policy: {e}")

# Configure inter/intra parallelism for better CPU utilization
cpu_count = os.cpu_count() or 4
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)  # Number of CPU cores for parallel operations
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)  # Number of CPU threads per operation
print(f"CPU threading configured with {cpu_count} threads")

# Configure GPU memory growth to avoid OOM errors
if gpus:
    try:
        # Enable memory growth on all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Configure memory limits for GPUs if needed
        # In TF 2.x, we use tf.config instead of the old session approach
        for i, gpu in enumerate(gpus):
            memory_limit = int(0.85 * 1024 * 1024 * 1024)  # 85% of total memory in bytes
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            except Exception as e:
                print(f"Could not set memory limit for GPU {i}: {e}")
        
        print(f"Found {len(gpus)} GPU(s), memory growth enabled, using 85% memory fraction")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs found, running on CPU only")

class EarlyStopping:
    def __init__(self, checkpoint, patience=100, minimize=True):
        self.minimize = minimize
        self.patience = patience
        self.checkpoint = checkpoint
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0

    def __call__(self, value, epoch):
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.checkpoint.save()
        elif self.best_monitored_epoch + self.patience < epoch:
            self.checkpoint.restore()
            return True
        return False

class ModelCloudLog:
    def __init__(self, log_dir, email_config=None):
        self.log_dir = log_dir
        self.email_config = email_config
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, "training.log"), "a")

    def __call__(self, message):
        self.log_file.write(message + "\n")
        self.log_file.flush()
        print(message)

    def log_parameters(self, params, train_size, valid_size, test_size):
        self("=============================================")
        self("=============== TRAINING DATA ===============")
        self("=============================================")
        
        self("=================== DATA ====================")
        self(f" Training set: {train_size} examples")
        self(f" Validation set: {valid_size} examples")
        self(f" Testing set: {test_size} examples")
        self(f" Batch size: {params.batch_size}")
        
        self("=================== MODEL ===================")
        self("--------------- ARCHITECTURE ----------------")
        self(" %-10s %-10s %-8s %-15s" % ("", "Type", "Size", "Dropout (keep p)"))
        self(" %-10s %-10s %-8s %-15s" % ("Layer 1", f"{params.conv1_k}x{params.conv1_k} Conv", str(params.conv1_d), str(params.conv1_p)))
        self(" %-10s %-10s %-8s %-15s" % ("Layer 2", f"{params.conv2_k}x{params.conv2_k} Conv", str(params.conv2_d), str(params.conv2_p)))
        self(" %-10s %-10s %-8s %-15s" % ("Layer 3", f"{params.conv3_k}x{params.conv3_k} Conv", str(params.conv3_d), str(params.conv3_p)))
        self(" %-10s %-10s %-8s %-15s" % ("Layer 4", "FC", str(params.fc4_size), str(params.fc4_p)))
        self("---------------- PARAMETERS -----------------")
        self(f" Learning rate decay: {'Enabled' if params.learning_rate_decay else f'Disabled (rate = {params.learning_rate})'}")
        self(f" L2 Regularization: {'Enabled (lambda = {params.l2_lambda})' if params.l2_reg_enabled else 'Disabled'}")
        self(f" Early stopping: {'Enabled (patience = {params.early_stopping_patience})' if params.early_stopping_enabled else 'Disabled'}")
        self(f" Keep training old model: {'Enabled' if params.resume_training else 'Disabled'}")
        
    def sync(self, notify=False, message=None):
        """Synchronize logs and optionally send email notification"""
        if notify and self.email_config and message:
            self.send_email_notification(message)
            
    def send_email_notification(self, message, image_path=None):
        """Send email notification with optional plot image attachment"""
        if not self.email_config:
            print("Email configuration not provided, skipping notification")
            return
            
        try:
            smtp_server = self.email_config.get("smtp_server")
            smtp_port = self.email_config.get("smtp_port", 587)
            sender_email = self.email_config.get("sender_email")
            receiver_email = self.email_config.get("receiver_email")
            password = self.email_config.get("password")
            
            if not all([smtp_server, sender_email, receiver_email, password]):
                print("Missing email configuration parameters, skipping notification")
                return
                
            msg = MIMEMultipart()
            msg['Subject'] = 'Traffic Sign Classification CNN Training Update'
            msg['From'] = sender_email
            msg['To'] = receiver_email
            
            # Add message body
            msg.attach(MIMEText(message))
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(img)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
                print(f"Email notification sent to {receiver_email}")
                
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            
    def add_plot(self, notify=False, caption=None):
        """Add plot to the log and optionally send as an email"""
        learning_curves_path = os.path.join(self.log_dir, "learning_curves.png")
        plt.savefig(learning_curves_path)
        
        if notify and caption and self.email_config:
            self.send_email_notification(f"Training update: {caption}", learning_curves_path)

def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test, logger_config):
    # Record start time for overall timing
    training_start_time = time.time()
    
    # Convert input data to float32 if needed
    if X_train.dtype != np.float32:
        print(f"Converting input data from {X_train.dtype} to float32")
        X_train = X_train.astype(np.float32)
        X_valid = X_valid.astype(np.float32)
        X_test = X_test.astype(np.float32)
    
    # Normalize the data to [0, 1] range if not already
    if X_train.max() > 1.0:
        print("Normalizing data to [0, 1] range")
        X_train /= 255.0
        X_valid /= 255.0
        X_test /= 255.0
    
    # Create efficient data pipelines using tf.data
    print("Creating optimized data pipelines for CPU/GPU coordination...")
    
    # Get the number of CPU cores for parallel processing
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if detection fails
    
    # Calculate buffer sizes based on available memory and dataset size
    shuffle_buffer = min(len(X_train), 10000)  # Limit buffer size for large datasets
    prefetch_buffer = tf.data.AUTOTUNE  # Let TensorFlow optimize prefetch size
    
    # Create training dataset with prefetching and parallel processing
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()  # Cache the dataset in memory
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer)
    train_dataset = train_dataset.batch(params.batch_size)
    train_dataset = train_dataset.prefetch(prefetch_buffer)  # Prefetch next batch while GPU processes current batch
    
    # Create validation dataset
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    valid_dataset = valid_dataset.batch(params.batch_size)
    valid_dataset = valid_dataset.prefetch(prefetch_buffer)
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(params.batch_size)
    test_dataset = test_dataset.prefetch(prefetch_buffer)
    
    print(f"Data pipeline created with {cpu_count} parallel workers and automatic prefetching")
    
    paths = Paths(params)
    log = ModelCloudLog(
        os.path.join(paths.root_path, "logs"),
        email_config=logger_config.get("email_config")
    )
    start = time.time()
    
    log.log_parameters(params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])
    
    # Create model directly from model_builder
    model = TrafficSignModel(params)
    
    # Learning rate schedule if enabled
    if params.learning_rate_decay:
        # For stage 1 (pre-training with lr=0.001), use moderate decay
        # For stage 2 (fine-tuning with lr=0.0001), use slower decay
        is_pretraining = params.learning_rate >= 0.001
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params.learning_rate,
            decay_steps=params.max_epochs // (2 if is_pretraining else 4),  # Faster decay for pre-training
            decay_rate=0.1 if is_pretraining else 0.5,  # Moderate decay for pre-training, slower for fine-tuning
            staircase=False  # Smooth decay instead of steps
        )
        learning_rate = lr_schedule
    else:
        learning_rate = params.learning_rate
    
    # Create optimizer optimized for mixed precision training
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Check if mixed precision is available and being used
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        # When using mixed precision, use loss scaling to improve numerical stability
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("Using mixed precision with LossScaleOptimizer")
    
    # Compile the model with optimized settings
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        # Enable XLA compilation for faster execution
        jit_compile=True
    )
    
    # Create a sample input to build the model
    sample_input = X_train[:1]
    _ = model(sample_input, training=False)
    model.summary()
    
    # Function to evaluate model (moved earlier to make it available for checkpoint restoration)
    def get_accuracy_and_loss_in_batches(X, y, subset_percentage=1.0):
        # Use a subset of data for faster evaluation during training
        if subset_percentage < 1.0 and len(X) > 1000:
            subset_size = int(len(X) * subset_percentage)
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = y[indices]
            
        predictions = []
        losses = []
        batch_size = 256  # Adjust batch size as needed
        
        # Add progress tracking for large datasets
        num_batches = (len(X) + batch_size - 1) // batch_size
        if num_batches > 10:  # Only show progress for large datasets
            print(f"\rEvaluating on {len(X)} samples: ", end="")
        
        for i in range(0, len(X), batch_size):
            if num_batches > 10:
                progress = (i // batch_size) / num_batches * 100
                progress_bar = '█' * int(progress / 2) + '-' * (50 - int(progress / 2))
                print(f"\r[{progress_bar}] {progress:.1f}% - Evaluating on {len(X)} samples ", end="")
                
            x_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            logits = model(x_batch, training=False)
            loss_value = tf.keras.losses.categorical_crossentropy(y_batch, logits, from_logits=True)
            loss_value = tf.reduce_mean(loss_value)
            pred = tf.nn.softmax(logits)
            
            predictions.append(pred)
            losses.append(loss_value)
        
        if num_batches > 10:
            print()  # New line after progress bar
            
        predictions = tf.concat(predictions, axis=0)
        accuracy = 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(y, 1)) / predictions.shape[0]
        loss = tf.reduce_mean(losses)
        
        return accuracy, loss.numpy()
    
    # Checkpoints for saving model
    checkpoint_dir = os.path.dirname(paths.model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=3
    )
    
    class CustomCheckpoint:
        def __init__(self, checkpoint_manager):
            self.manager = checkpoint_manager
            
        def save(self):
            self.manager.save()
            
        def restore(self):
            if self.manager.latest_checkpoint:
                checkpoint.restore(self.manager.latest_checkpoint)
                return True
            return False
    
    # Create checkpoint wrapper for early stopping
    custom_checkpoint = CustomCheckpoint(checkpoint_manager)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(custom_checkpoint, patience=params.early_stopping_patience, minimize=True)
    
    # Try to restore previous model if requested
    if params.resume_training:
        try:
            # Try direct restore of TF checkpoint first
            if os.path.exists(checkpoint_dir + "/checkpoint"):
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    # Direct checkpoint restore
                    status = checkpoint.restore(latest_checkpoint)
                    status.expect_partial()  # Allow partial restore without warnings
                    log(f"Directly restored previously trained model from {latest_checkpoint}")
                    
                    # Extract epoch number from checkpoint name if possible
                    try:
                        checkpoint_name = os.path.basename(latest_checkpoint)
                        if '-' in checkpoint_name:
                            checkpoint_epoch = int(checkpoint_name.split('-')[1])
                            early_stopping.best_monitored_epoch = checkpoint_epoch
                            log(f"Set best monitored epoch to {checkpoint_epoch}")
                    except:
                        pass
                        
                    # Ensure optimizer state is properly warmed up with multiple steps
                    log("Warming up optimizer state...")
                    for i in range(3):  # Run a few steps to ensure proper optimizer state
                        dummy_data = X_train[i*params.batch_size:(i+1)*params.batch_size]
                        dummy_labels = y_train[i*params.batch_size:(i+1)*params.batch_size]
                        
                        # Run optimization step
                        with tf.GradientTape() as tape:
                            logits = model(dummy_data, training=True)
                            loss_value = tf.keras.losses.categorical_crossentropy(dummy_labels, logits, from_logits=True)
                            loss_value = tf.reduce_mean(loss_value)
                            
                        gradients = tape.gradient(loss_value, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    # Evaluate to set early stopping variables appropriately
                    val_acc, val_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid, subset_percentage=0.2)
                    early_stopping.best_monitored_value = val_loss
                    log(f"Set best monitored value to {val_loss:.8f}")
                    log("Optimizer state reactivated and early stopping initialized")
                else:
                    # Fallback to using our custom checkpoint manager
                    if custom_checkpoint.restore():
                        log("Restored previously trained model via checkpoint manager.")
                        # Rest of optimization warm-up
                        dummy_data = X_train[:params.batch_size]
                        dummy_labels = y_train[:params.batch_size]
                        
                        with tf.GradientTape() as tape:
                            logits = model(dummy_data, training=True)
                            loss_value = tf.keras.losses.categorical_crossentropy(dummy_labels, logits, from_logits=True)
                            loss_value = tf.reduce_mean(loss_value)
                            
                        gradients = tape.gradient(loss_value, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
                        log("Optimizer state reactivated.")
                    else:
                        log("No checkpoint found, starting from scratch.")
            else:
                # Fallback to using our custom checkpoint manager
                if custom_checkpoint.restore():
                    log("Restored previously trained model via checkpoint manager.")
                    # Rest of optimization warm-up
                    dummy_data = X_train[:params.batch_size]
                    dummy_labels = y_train[:params.batch_size]
                    
                    with tf.GradientTape() as tape:
                        logits = model(dummy_data, training=True)
                        loss_value = tf.keras.losses.categorical_crossentropy(dummy_labels, logits, from_logits=True)
                        loss_value = tf.reduce_mean(loss_value)
                        
                    gradients = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    log("Optimizer state reactivated.")
                else:
                    log("No checkpoint found, starting from scratch.")
        except Exception as e:
            log(f"Failed restoring previously trained model: {e}")
    
    # Training history
    train_loss_history = np.empty([0], dtype=np.float32)
    train_accuracy_history = np.empty([0], dtype=np.float32)
    valid_loss_history = np.empty([0], dtype=np.float32)
    valid_accuracy_history = np.empty([0], dtype=np.float32)
    
    # Define the training step with XLA compilation for better GPU performance
    @tf.function(jit_compile=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss_value = tf.reduce_mean(loss_value)
            
            # Add L2 regularization if enabled
            if params.l2_reg_enabled:
                l2_losses = []
                for var in model.trainable_variables:
                    if 'kernel' in var.name or 'weights' in var.name:
                        l2_losses.append(tf.nn.l2_loss(var))
                if l2_losses:
                    l2_loss = tf.add_n(l2_losses)
                    loss_value += params.l2_lambda * l2_loss
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_value, tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), tf.float32))
    
    if params.max_epochs > 0:
        log("================= TRAINING ==================")
    else:
        log("================== TESTING ==================")
        log(f" Timestamp: {get_time_hhmmss()}")
        log.sync()
    
    # Main training loop using tf.data pipeline for better performance
    for epoch in range(params.max_epochs):
        # Reset metrics for this epoch
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # Display epoch progress
        epoch_progress = epoch / params.max_epochs * 100
        progress_bar = '█' * int(epoch_progress / 2) + '-' * (50 - int(epoch_progress / 2))
        print(f"\r[{progress_bar}] {epoch_progress:.1f}% - Epoch {epoch}/{params.max_epochs} ", end='', flush=True)
        
        # Track time per epoch
        epoch_start_time = time.time()
        
        # Train on batches from tf.data pipeline
        step = 0
        total_steps = len(X_train) // params.batch_size
        
        for x_batch, y_batch in train_dataset:
            # Update progress within epoch
            if step % max(1, total_steps // 20) == 0:  # Show ~20 updates during an epoch
                batch_progress = step / total_steps * 100
                print(f"\r[{progress_bar}] {epoch_progress:.1f}% - Epoch {epoch}/{params.max_epochs} - Batch progress: {batch_progress:.1f}% ", end='', flush=True)
            
            # Perform training step
            loss_value, accuracy = train_step(x_batch, y_batch)
            
            # Update metrics
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_batch, model(x_batch, training=False))
            
            step += 1
        
        # If another significant epoch ended, we log our losses.
        if epoch % params.log_epoch == 0:
            # Get validation metrics using our optimized dataset
            valid_loss_avg = tf.keras.metrics.Mean()
            valid_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
            
            # Use the validation dataset for evaluation
            for x_valid_batch, y_valid_batch in valid_dataset:
                # Get model predictions
                valid_predictions = model(x_valid_batch, training=False)
                
                # Calculate validation loss
                v_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_valid_batch, valid_predictions, from_logits=True)
                )
                
                # Update metrics
                valid_loss_avg.update_state(v_loss)
                valid_accuracy_metric.update_state(y_valid_batch, valid_predictions)
            
            # Get the current metrics
            train_loss = epoch_loss_avg.result().numpy()
            train_accuracy = epoch_accuracy.result().numpy() * 100
            valid_loss = valid_loss_avg.result().numpy()
            valid_accuracy = valid_accuracy_metric.result().numpy() * 100
            
            # Update histories
            train_loss_history = np.append(train_loss_history, train_loss)
            train_accuracy_history = np.append(train_accuracy_history, train_accuracy)
            valid_loss_history = np.append(valid_loss_history, valid_loss)
            valid_accuracy_history = np.append(valid_accuracy_history, valid_accuracy)
            
            # Check early stopping condition
            if params.early_stopping_enabled:
                # Here we pass our validation loss to early stopping which
                # will save a checkpoint if improved or terminate if stalled for too long
                if early_stopping(valid_loss, epoch):
                    log(f"Early stopping triggered at epoch {epoch}.")
                    log(f"Best monitored loss was {early_stopping.best_monitored_value:.8f} at epoch {early_stopping.best_monitored_epoch}.")
                    break
            
            # Log results
            if epoch % params.print_epoch == 0:
                epoch_time = time.time() - epoch_start_time
                log(f"-------------- EPOCH {epoch:4d}/{params.max_epochs} --------------")
                log(f" Train loss: {train_loss:.8f}, accuracy: {train_accuracy:.2f}%")
                log(f" Validation loss: {valid_loss:.8f}, accuracy: {valid_accuracy:.2f}%")
                log(f" Best loss: {early_stopping.best_monitored_value:.8f} at epoch {early_stopping.best_monitored_epoch}")
                log(f" Epoch time: {epoch_time:.2f}s, Total time: {get_time_hhmmss(start)}")
                
                # Create a plot of learning curves
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(train_loss_history, label='Train')
                plt.plot(valid_loss_history, label='Validation')
                plt.title('Loss')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(train_accuracy_history, label='Train')
                plt.plot(valid_accuracy_history, label='Validation')
                plt.title('Accuracy')
                plt.legend()
                
                log.add_plot(
                    notify=(epoch > 0 and epoch % 10 == 0), 
                    caption=f"Epoch {epoch}/{params.max_epochs}: Train acc={train_accuracy:.2f}%, Valid acc={valid_accuracy:.2f}%"
                )
                log(f" Timestamp: {get_time_hhmmss()}")
                log.sync()
    
    # Evaluate on test dataset.
    print("\nEvaluating on test dataset...")
    test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
    valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
    log("=============================================")
    log(f" Valid loss: {valid_loss:.8f}, accuracy = {valid_accuracy:.2f}%")
    log(f" Test loss: {test_loss:.8f}, accuracy = {test_accuracy:.2f}%")
    log(f" Total time: {get_time_hhmmss(start)}")
    log(f" Timestamp: {get_time_hhmmss()}")
    
    # Save model weights for future use.
    saved_model_path = checkpoint_manager.save()
    log(f"Model file: {saved_model_path}")
    
    # Also save as a complete Keras model
    model_save_path = os.path.join(paths.root_path, "keras_model")
    model.save(model_save_path)
    log(f"Keras model saved to: {model_save_path}")
    
    np.savez(paths.train_history_path, 
            train_loss_history=train_loss_history,
            train_accuracy_history=train_accuracy_history,
            valid_loss_history=valid_loss_history,
            valid_accuracy_history=valid_accuracy_history)
    log(f"Train history file: {paths.train_history_path}")
    log.sync(notify=True, message=f"Finished training with {test_accuracy:.2f}% accuracy on the testing set (loss = {test_loss:.6f}).")
    
    plot_learning_curves(params)
    log.add_plot(notify=True, caption="Learning curves")
    plt.show()

class Paths:
    def __init__(self, params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        self.train_history_path = self.get_train_history_path()
        self.learning_curves_path = self.get_learning_curves_path()
        os.makedirs(self.root_path, exist_ok=True)
    
    def get_model_name(self, params):
        model_name = f"k{params.conv1_k}d{params.conv1_d}p{params.conv1_p}_k{params.conv2_k}d{params.conv2_d}p{params.conv2_p}_k{params.conv3_k}d{params.conv3_d}p{params.conv3_p}_fc{params.fc4_size}p{params.fc4_p}"
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name
    
    def get_variables_scope(self, params):
        var_scope = f"k{params.conv1_k}d{params.conv1_d}_k{params.conv2_k}d{params.conv2_d}_k{params.conv3_k}d{params.conv3_d}_fc{params.fc4_size}_fc0"
        return var_scope
    
    def get_model_path(self):
        return self.root_path + "model.ckpt"
    
    def get_train_history_path(self):
        return self.root_path + "train_history"
    
    def get_learning_curves_path(self):
        return self.root_path + "learning_curves.png"

def plot_learning_curves(params):
    curves_figure = plt.figure(figsize=(10, 4))
    axis = curves_figure.add_subplot(1, 2, 1)
    epochs_plotted = plot_curve(axis, params, train_column="train_accuracy_history", valid_column="valid_accuracy_history")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(50., 115.)
    plt.xlim(0, epochs_plotted)
    
    axis = curves_figure.add_subplot(1, 2, 2)
    epochs_plotted = plot_curve(axis, params, train_column="train_loss_history", valid_column="valid_loss_history")
    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0.0001, 10.)
    plt.xlim(0, epochs_plotted)
    plt.yscale("log")
    
def plot_curve(axis, params, train_column, valid_column, linewidth=2, train_linestyle="b-", valid_linestyle="g-"):
    model_history = np.load(Paths(params).train_history_path + ".npz")
    train_values = model_history[train_column]
    valid_values = model_history[valid_column]
    epochs = train_values.shape[0]
    x_axis = np.arange(epochs)
    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth, label="train")
    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth, label="valid")
    return epochs