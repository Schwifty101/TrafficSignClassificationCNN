# predictor.py
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_builder import Parameters, model_pass
from data_loader import load_sign_names
from data_processor import preprocess_dataset
from skimage import io

def predict_on_new_images(params, image_paths):
    sign_names = load_sign_names()
    X_custom = np.empty([0, 32, 32, 3], dtype=np.int32)
    
    for i in range(len(image_paths)):
        image = io.imread(image_paths[i])
        X_custom = np.append(X_custom, [image[:, :, :3]], axis=0)
    
    X_custom, _ = preprocess_dataset(X_custom)
    predictions = get_top_k_predictions(params, X_custom)
    
    for i in range(len(image_paths)):
        print(f"Actual class: {sign_names[i]}")
        plot_image_statistics(predictions, i, image_paths[i], sign_names, X_custom)
        print("---------------------------------------------------------------------------------------------------\n")
    
    # The lines below have an issue as y_custom is not defined
    # Commenting out until the variable is properly defined
    # X_custom = X_custom[y_custom < 99]
    # y_custom = y_custom[y_custom < 99]
    # y_custom = np.eye(43)[y_custom]
    # predictions = get_top_k_predictions(params, X_custom)[1][:, 0]
    # accuracy = 100.0 * np.sum(predictions == np.argmax(y_custom, 1)) / predictions.shape[0]
    # print(f"Accuracy on captured images: {accuracy:.2f}%")

def plot_image_statistics(predictions, index, image_path, sign_names, X_custom):
    original = io.imread(image_path)
    preprocessed = X_custom[index].reshape(32, 32)
    
    plt.figure(figsize=(6, 2))
    plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=2)
    plt.barh(np.arange(5)+.5, predictions[0][index], align='center')
    plt.yticks(np.arange(5)+.5, sign_names[predictions[1][index].astype(int)])
    plt.tick_params(axis='both', which='both', labelleft='off', labelright='on', labeltop='off', labelbottom='off')
    plt.show()

def get_top_k_predictions(params, X, k=5):
    paths = Paths(params)
    graph = tf.Graph()
    with graph.as_default():
        tf_x = tf.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
        is_training = tf.constant(False)
        with tf.variable_scope(paths.var_scope):
            predictions = tf.nn.softmax(model_pass(tf_x, params, is_training))
        top_k_predictions = tf.nn.top_k(predictions, k)
        
        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            tf.train.Saver().restore(session, paths.model_path)
            [p] = session.run([top_k_predictions], feed_dict={tf_x: X})
            return np.array(p)

class Paths:
    def __init__(self, params):
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        
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