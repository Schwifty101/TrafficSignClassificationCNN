# trainer.py
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from model_builder import Parameters, model_pass, get_time_hhmmss, print_progress
from data_loader import load_pickled_data
import os
import sys

# TF2.x compatible EarlyStopping class (for models not using Keras)
class EarlyStopping:
    """
    Early stopping handler for TensorFlow 2.x. 
    In most cases, you should use tf.keras.callbacks.EarlyStopping instead.
    This class is kept for compatibility with non-Keras TF2.x code.
    
    Args:
        model: The TensorFlow model to checkpoint/restore
        checkpoint_path: Path to save model checkpoints
        patience: Number of epochs to wait for improvement before stopping
        minimize: Whether to minimize (True) or maximize (False) the monitored value
    """
    def __init__(self, model=None, checkpoint_path=None, patience=100, minimize=True):
        self.minimize = minimize
        self.patience = patience
        self.model = model  # TF2.x model
        self.checkpoint_path = checkpoint_path or os.path.join(os.getcwd(), "early_stopping_checkpoint.h5")
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.best_weights = None
    
    def __call__(self, value, epoch):
        improved = False
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            improved = True
            
            # Save best weights
            if self.model is not None:
                self.best_weights = self.model.get_weights()
                if self.checkpoint_path:
                    self.model.save_weights(self.checkpoint_path)
                    print(f"Saved checkpoint at epoch {epoch+1}")
                    
        if self.best_monitored_epoch + self.patience < epoch:
            # Restore best weights if possible
            if self.model is not None and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print(f"Early stopping triggered, restored weights from epoch {self.best_monitored_epoch+1}")
            elif self.model is not None and self.checkpoint_path and os.path.exists(self.checkpoint_path):
                self.model.load_weights(self.checkpoint_path)
                print(f"Early stopping triggered, restored weights from checkpoint")
            else:
                print("WARNING: Early stopping triggered but couldn't restore best weights")
            return True
        return False

class ModelLogger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'a')
        
    def __call__(self, message):
        self.log_file.write(message + "\n")
        self.log_file.flush()
        print(message)
    
    def sync(self, notify=False, message=None):
        self.log_file.flush()
        if notify and message:
            print(f"Notification: {message}")
    
    def add_plot(self, notify=False, caption=None):
        if notify and caption:
            print(f"Added plot: {caption}")

    def log_parameters(self, params, train_size, valid_size, test_size):
        self("=============================================")
        self("============= RESUMING TRAINING =============")
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
        self(f" L2 Regularization: {'Enabled (lambda = ' + str(params.l2_lambda) + ')' if params.l2_reg_enabled else 'Disabled'}")
        self(f" Early stopping: {'Enabled (patience = ' + str(params.early_stopping_patience) + ')' if params.early_stopping_enabled else 'Disabled'}")
        self(f" Keep training old model: {'Enabled' if params.resume_training else 'Disabled'}")

def progress_bar(current, total, bar_length=50, prefix='', suffix=''):
    """Display a progress bar in the terminal"""
    filled_length = int(round(bar_length * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test, logger_config=None):
    # DEBUG: Start of training process
    print(f"DEBUG: Starting training with {X_train.shape[0]} training samples")
    
    paths = Paths(params)
    log = ModelLogger(os.path.join(paths.root_path, "training_log.txt"))
    start = time.time()
    
    log.log_parameters(params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])
    
    # DEBUG: Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"DEBUG: Found {len(gpus)} GPUs: {gpus}")
    
    # Set up TF2 strategy for potential GPU usage
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 0 else tf.distribute.get_strategy()
    print(f"DEBUG: Using {'multi-GPU' if len(gpus) > 0 else 'CPU'} strategy")
    
    with strategy.scope():
        # Define model architecture using tf.keras
        print("DEBUG: Building model architecture...")
        # This is the standard model architecture used throughout the project
        # Model Architecture:
        # - 3 Convolutional layers with ReLU activation
        # - 3 Max pooling layers with dropout
        # - Flatten layer
        # - 1 Fully connected layer with ReLU activation and dropout
        # - Output layer with softmax activation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(params.image_size[0], params.image_size[1], 1)))
        
        # Convert existing model_pass function to explicit layer definitions
        # Layer 1: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv1_d, (params.conv1_k, params.conv1_k), activation='relu', padding='same', 
                                         kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                         name='conv1'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(tf.keras.layers.Dropout(1 - params.conv1_p, name='dropout1'))
        
        # Layer 2: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv2_d, (params.conv2_k, params.conv2_k), activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                         name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
        model.add(tf.keras.layers.Dropout(1 - params.conv2_p, name='dropout2'))
        
        # Layer 3: Convolutional + MaxPooling
        model.add(tf.keras.layers.Conv2D(params.conv3_d, (params.conv3_k, params.conv3_k), activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                         name='conv3'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
        model.add(tf.keras.layers.Dropout(1 - params.conv3_p, name='dropout3'))
        
        # Flatten and fully connected layers
        model.add(tf.keras.layers.Flatten(name='flatten'))
        model.add(tf.keras.layers.Dense(params.fc4_size, activation='relu', 
                                        kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                        name='fc4'))
        model.add(tf.keras.layers.Dropout(1 - params.fc4_p, name='dropout4'))
        model.add(tf.keras.layers.Dense(params.num_classes, activation='softmax', 
                                         kernel_regularizer=tf.keras.regularizers.l2(params.l2_lambda) if params.l2_reg_enabled else None,
                                         name='output'))
        
        # Print model summary for debugging
        model.summary()
        print("DEBUG: Model architecture built successfully")
        
        # Learning rate schedule with proper decay rates
        if params.learning_rate_decay:
            print(f"DEBUG: Using learning rate decay from {params.learning_rate}")
            # Configure more effective decay settings based on initial learning rate
            if params.learning_rate >= 0.001:
                # For training mode: faster decay early, then more gradual
                decay_steps = 500  # Decay more frequently during training
                decay_rate = 0.95  # More gradual decay (95% of previous value)
                print(f"DEBUG: Training mode decay: rate={decay_rate}, steps={decay_steps}")
            else:
                # For evaluation/prediction mode: very slow decay to fine-tune
                decay_steps = 2000  # Decay less frequently during evaluation
                decay_rate = 0.98  # Very slow decay (98% of previous value)
                print(f"DEBUG: Evaluation mode decay: rate={decay_rate}, steps={decay_steps}")
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=params.learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            print(f"DEBUG: Using constant learning rate {params.learning_rate}")
            optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
        
        # Compile the model
        print("DEBUG: Compiling model...")
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Try to load weights if resuming training
    if params.resume_training:
        print(f"DEBUG: Attempting to load weights from {paths.model_path}")
        try:
            # Check if the file exists before trying to load it
            if os.path.exists(paths.model_path):
                model.load_weights(paths.model_path)
                log("Successfully loaded previously trained model weights")
                print("DEBUG: Model weights loaded successfully")
            else:
                log("No previous model weights found, starting with fresh weights")
                print("DEBUG: No previous model weights found")
        except Exception as e:
            log(f"Failed restoring previously trained model: {e}")
            print(f"DEBUG: Failed to load weights: {e}")
            log("Continuing with fresh model weights")
            print("DEBUG: Continuing with fresh model weights")
    
    # Create callbacks
    print("DEBUG: Setting up callbacks...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=paths.model_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1  # Print when model is saved
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(paths.root_path, 'training_log.csv')),
    ]
    
    # Add early stopping if enabled
    if params.early_stopping_enabled:
        print(f"DEBUG: Adding early stopping with patience {params.early_stopping_patience}")
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.early_stopping_patience,
            mode='min',
            restore_best_weights=True,
            verbose=1  # Print when early stopping triggers
        ))
    
    # Custom callback for logging and progress tracking
    class ProgressLoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, log_function, model_params, train_data_info):
            super(ProgressLoggingCallback, self).__init__()
            self.log = log_function
            self.model_params = model_params  # Renamed to avoid conflict with Keras internal params
            self.train_data_info = train_data_info
            self.start_time = time.time()
            self.epoch_start_time = None
            self.train_batches = None
            self.best_val_loss = float('inf')
            self.best_val_acc = 0
        
        def on_train_begin(self, logs=None):
            print("DEBUG: Training started")
            self.train_batches = int(np.ceil(self.train_data_info.train_data_size / self.model_params.batch_size))
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.model_params.max_epochs}")
            
        def on_train_batch_end(self, batch, logs=None):
            # Update progress bar after each batch
            progress_bar(
                batch + 1, 
                self.train_batches, 
                prefix=f'Training batch {batch+1}/{self.train_batches}', 
                suffix=f'Loss: {logs["loss"]:.4f}, Accuracy: {logs["accuracy"]:.4f}'
            )
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            
            # Track best metrics
            if logs.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = logs.get('val_loss')
                improved_loss = "✓"
            else:
                improved_loss = " "
                
            if logs.get('val_accuracy', 0) > self.best_val_acc:
                self.best_val_acc = logs.get('val_accuracy')
                improved_acc = "✓"
            else:
                improved_acc = " "
            
            # Log detailed metrics at specified intervals
            if epoch % self.model_params.print_epoch == 0:
                self.log(f"-------------- EPOCH {epoch+1:4d}/{self.model_params.max_epochs} --------------")
                self.log(f" Train loss: {logs['loss']:.8f}, accuracy: {logs['accuracy'] * 100:.2f}%")
                self.log(f" Validation loss: {logs['val_loss']:.8f}, accuracy: {logs['val_accuracy'] * 100:.2f}%")
                self.log(f" Best val loss: {self.best_val_loss:.8f}, Best val accuracy: {self.best_val_acc * 100:.2f}%")
                self.log(f" Epoch time: {epoch_time:.2f}s, Total time: {get_time_hhmmss(self.start_time)}")
                self.log(f" Timestamp: {get_time_hhmmss()}")
                self.log.sync()
            
            # Always print a summary to console
            # Access learning rate in a way that's compatible with TF 2.x
            try:
                # More robust way to get learning rate that works with different TensorFlow versions
                if hasattr(self.model.optimizer, 'learning_rate'):
                    lr = self.model.optimizer.learning_rate
                    if hasattr(lr, 'numpy'):
                        current_lr = lr.numpy()
                    elif callable(lr):
                        current_lr = float(lr(self.model.optimizer.iterations))
                    else:
                        current_lr = float(lr)
                elif hasattr(self.model.optimizer, '_decayed_lr'):
                    # For older TensorFlow versions
                    current_lr = float(self.model.optimizer._decayed_lr(tf.float32))
                elif hasattr(self.model.optimizer, 'lr'):
                    # For very old TensorFlow versions
                    current_lr = float(self.model.optimizer.lr)
                else:
                    # Last resort
                    current_lr = 0.0
            except Exception as e:
                print(f"DEBUG: Unable to get learning rate: {e}")
                current_lr = 0.0
                
            print(f"\nEpoch {epoch+1}/{self.model_params.max_epochs} completed in {epoch_time:.2f}s")
            print(f"Loss: {logs['loss']:.4f} [train], {logs['val_loss']:.4f} [val] {improved_loss}")
            print(f"Accuracy: {logs['accuracy']*100:.2f}% [train], {logs['val_accuracy']*100:.2f}% [val] {improved_acc}")
            print(f"Learning rate: {current_lr:.8f}")
            
    # Create a custom object to track train data size
    # Since params is a namedtuple and can't be modified directly
    train_data_info = type('TrainDataInfo', (), {})()
    train_data_info.train_data_size = len(X_train)
    
    # Pass both the params and the train data info to the callback
    callbacks.append(ProgressLoggingCallback(log, params, train_data_info))
    
    if params.max_epochs > 0:
        log("================= TRAINING ==================")
        print("DEBUG: Starting training loop...")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=params.max_epochs,
            batch_size=params.batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks,
            verbose=0  # We'll use our own progress reporting
        )
        
        print("DEBUG: Training completed, saving history...")
        # Save training history
        history_dict = {
            'train_loss_history': history.history['loss'],
            'train_accuracy_history': [acc * 100 for acc in history.history['accuracy']],
            'valid_loss_history': history.history['val_loss'],
            'valid_accuracy_history': [acc * 100 for acc in history.history['val_accuracy']]
        }
        np.savez(paths.train_history_path, **history_dict)
    else:
        log("================== TESTING ==================")
        log(f" Timestamp: {get_time_hhmmss()}")
        log.sync()
    
    # Evaluate on test dataset
    print("DEBUG: Evaluating on test set...")
    print("Evaluating test set:")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    print("Evaluating validation set:")
    valid_results = model.evaluate(X_valid, y_valid, verbose=1)
    
    log("=============================================")
    log(f" Valid loss: {valid_results[0]:.8f}, accuracy = {valid_results[1] * 100:.2f}%")
    log(f" Test loss: {test_results[0]:.8f}, accuracy = {test_results[1] * 100:.2f}%")
    log(f" Total time: {get_time_hhmmss(start)}")
    log(f" Timestamp: {get_time_hhmmss()}")
    
    # Save model weights
    print("DEBUG: Saving final model weights...")
    model.save_weights(paths.model_path)
    log(f"Model file: {paths.model_path}")
    
    log(f"Train history file: {paths.train_history_path}")
    log.sync(notify=True, message=f"Finished training with {test_results[1] * 100:.2f}% accuracy on the testing set (loss = {test_results[0]:.6f}).")
    
    # Generate and save learning curves
    print("DEBUG: Generating learning curves...")
    plot_learning_curves(params)
    plt.savefig(paths.learning_curves_path)
    log.add_plot(notify=True, caption="Learning curves")
    plt.show()
    
    print("DEBUG: Training process completed successfully")
    return model

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
        # Use .weights.h5 extension for compatibility with Keras 3
        return self.root_path + "model.weights.h5"
    
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