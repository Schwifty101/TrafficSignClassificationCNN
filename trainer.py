# trainer.py
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from model_builder import Parameters, model_pass, get_time_hhmmss, print_progress
from data_loader import load_pickled_data
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

class EarlyStopping:
    def __init__(self, saver, session, patience=100, minimize=True):
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        if (self.minimize and value < self.best_monitored_value) or (not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session, os.getcwd() + "/early_stopping_checkpoint")
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
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
    paths = Paths(params)
    log = ModelCloudLog(
        os.path.join(paths.root_path, "logs"),
        email_config=logger_config.get("email_config")
    )
    start = time.time()
    
    log.log_parameters(params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])
    
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        tf_x_batch = tf.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.placeholder(tf.float32, shape=(None, params.num_classes))
        is_training = tf.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)
        
        if params.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(params.learning_rate, current_epoch, decay_steps=params.max_epochs, decay_rate=0.01)
        else:
            learning_rate = params.learning_rate
            
        with tf.variable_scope(params.var_scope):
            logits = model_pass(tf_x_batch, params, is_training)
            
        predictions = tf.nn.softmax(logits)
        
        if params.l2_reg_enabled:
            with tf.variable_scope('fc4', reuse=True):
                l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
        else:
            l2_loss = 0
            
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y_batch)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            
            def get_accuracy_and_loss_in_batches(X, y):
                p = []
                sce = []
                batch_size = 128
                for i in range(0, len(X), batch_size):
                    x_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict={tf_x_batch: x_batch, tf_y_batch: y_batch, is_training: False})
                    p.extend(p_batch)
                    sce.extend(sce_batch)
                p = np.array(p)
                sce = np.array(sce)
                accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
                loss = np.mean(sce)
                return (accuracy, loss)
            
            if params.resume_training:
                try:
                    tf.train.Saver().restore(session, paths.model_path)
                except Exception as e:
                    log(f"Failed restoring previously trained model: {e}")
                    pass
            
            saver = tf.train.Saver()
            early_stopping = EarlyStopping(tf.train.Saver(), session, patience=params.early_stopping_patience, minimize=True)
            
            train_loss_history = np.empty([0], dtype=np.float32)
            train_accuracy_history = np.empty([0], dtype=np.float32)
            valid_loss_history = np.empty([0], dtype=np.float32)
            valid_accuracy_history = np.empty([0], dtype=np.float32)
            
            if params.max_epochs > 0:
                log("================= TRAINING ==================")
            else:
                log("================== TESTING ==================")
                log(f" Timestamp: {get_time_hhmmss()}")
                log.sync()
            
            for epoch in range(params.max_epochs):
                current_epoch = epoch
                
                # Train on whole randomised dataset in batches
                batch_size = params.batch_size
                for i in range(0, len(X_train), batch_size):
                    x_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    session.run([optimizer], feed_dict={tf_x_batch: x_batch, tf_y_batch: y_batch, is_training: True})
                
                # If another significant epoch ended, we log our losses.
                if epoch % params.log_epoch == 0:
                    # Get validation data predictions and log validation loss:
                    valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                    # Get training data predictions and log training loss:
                    train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)
                    if epoch % params.print_epoch == 0:
                        log(f"-------------- EPOCH {epoch:4d}/{params.max_epochs} --------------")
                        log(f" Train loss: {train_loss:.8f}, accuracy: {train_accuracy:.2f}%")
                        log(f" Validation loss: {valid_loss:.8f}, accuracy: {valid_accuracy:.2f}%")
                        log(f" Best loss: {early_stopping.best_monitored_value:.8f} at epoch {early_stopping.best_monitored_epoch}")
                        log(f" Elapsed time: {get_time_hhmmss(start)}")
                        log(f" Timestamp: {get_time_hhmmss()}")
                        log.sync()
                    else:
                        valid_loss = 0.
                        valid_accuracy = 0.
                        train_loss = 0.
                        train_accuracy = 0.
                    
                    valid_loss_history = np.append(valid_loss_history, [valid_loss])
                    valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
                    train_loss_history = np.append(train_loss_history, [train_loss])
                    train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])
                
                if params.early_stopping_enabled:
                    # Get validation data predictions and log validation loss:
                    if valid_loss == 0:
                        _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                    if early_stopping(valid_loss, epoch):
                        log(f"Early stopping.\nBest monitored loss was {early_stopping.best_monitored_value:.8f} at epoch {early_stopping.best_monitored_epoch}.")
                        break
            
            # Evaluate on test dataset.
            test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
            valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
            log("=============================================")
            log(f" Valid loss: {valid_loss:.8f}, accuracy = {valid_accuracy:.2f}%")
            log(f" Test loss: {test_loss:.8f}, accuracy = {test_loss:.2f}%")
            log(f" Total time: {get_time_hhmmss(start)}")
            log(f" Timestamp: {get_time_hhmmss()}")
            
            # Save model weights for future use.
            saved_model_path = saver.save(session, paths.model_path)
            log(f"Model file: {saved_model_path}")
            
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