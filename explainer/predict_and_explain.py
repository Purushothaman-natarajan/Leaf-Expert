import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from lime.lime_image import LimeImageExplainer, SegmentationAlgorithm
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def load_model_details(model_path):
    # Detect the format and load the model accordingly
    if model_path.endswith('.keras'):
        # Assuming V3 `.keras` format
        print("Loading .keras format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    elif model_path.endswith('.h5'):
        # Legacy H5 format
        print("Loading .h5 format model...")
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        # Assuming it's a SavedModel format; use TFSMLayer
        print("Loading SavedModel using TFSMLayer...")
        model = tf.keras.Sequential([
            tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        ])

    # Get the target size dynamically
    input_shape = model.input_shape[1:3]

    # Find the last convolutional layer name
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    return model, last_conv_layer_name, input_shape
    
def load_label_encoder(train_directory):
    labels = sorted(os.listdir(train_directory))
    label_encoder = {i: label for i, label in enumerate(labels)}
    return label_encoder

def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        preds = tf.convert_to_tensor(preds)
        class_channel = preds[:, pred_index]
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])  # Default to the class with the highest probability
        # pred_index = tf.squeeze(pred_index)  # Ensure pred_index is a scalar tensor
        # if tf.executing_eagerly():
        #     pred_index = pred_index.numpy()  # Convert to a NumPy array
        # pred_index = int(pred_index)  # Convert to a Python integer
        # class_channel = preds[0][pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(array, heatmap, alpha=0.8):
    img = array
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.jet
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

def generate_splime_mask(img_array, model, num_features, num_samples, segmentation_alg, kernel_size, max_dist, ratio):
    explainer = LimeImageExplainer()
    if segmentation_alg == 'quickshift':
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    else:
        segmentation_fn = SegmentationAlgorithm('slic', n_segments=kernel_size, compactness=max_dist, sigma=ratio)
    
    explanation = explainer.explain_instance(
        img_array, model.predict, top_labels=1, hide_color=0,
        num_samples=num_samples, num_features=num_features, segmentation_fn=segmentation_fn
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=num_features, hide_rest=True
    )
    
    # Convert grayscale mask to RGB
    if len(mask.shape) == 2:  # Mask is grayscale
        mask = np.stack([mask] * 3, axis=-1)
    
    return mask

def save_and_display_splime(array, mask, alpha=0.6):
    """
    Overlap the LIME mask with the original image and return the superimposed image.
    """
    img = array
    mask = np.uint8(255 * mask)
    mask = array_to_img(mask)
    mask = mask.resize((img.shape[1], img.shape[0]))
    mask = img_to_array(mask)
    superimposed_img = mask * alpha + img
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

image_counter = 0

def classify_image_and_explain(image_path, model_path, train_directory, num_samples, num_features, segmentation_alg, kernel_size, max_dist, ratio):
    global image_counter
    image_counter += 1
    model, last_conv_layer_name, target_size = load_model_details(model_path)
    
    image = Image.open(image_path).resize(target_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    array = img_to_array(image)
    img_array = array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    label_encoder = load_label_encoder(train_directory)
    
    preds = model.predict(img_array)
    top_prediction = np.argmax(preds[0])
    top_label = label_encoder[top_prediction]
    top_prob = preds[0][top_prediction]

    model.layers[-1].activation = None
    grad_cam_heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    grad_cam_explanation = save_and_display_gradcam(array, grad_cam_heatmap)
    
    splime_mask = generate_splime_mask(img_array[0], model, num_features, num_samples, segmentation_alg, kernel_size, max_dist, ratio)
    splime_explanation = save_and_display_splime(array, splime_mask)

    # Ensure the explanation directory exists
    if not os.path.exists("explanation"):
        os.makedirs("explanation")

    grad_cam_explanation.save(f"explanation/gradcam_explanation_{image_counter}.jpg")
    splime_explanation.save(f"explanation/splime_explanation_{image_counter}.jpg")

    print(f"Predicted Label: {top_label}")
    print(f"Probability: {top_prob:.4f}")

    return splime_explanation, grad_cam_explanation, top_label, top_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image and generate explanations using LIME and Grad-CAM.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (e.g., /path/to/model.h5 or /path/to/model.keras)")
    parser.add_argument("--train_directory", type=str, required=True, help="Path to the training directory (e.g., /path/to/train)")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples for LIME")
    parser.add_argument("--num_features", type=int, default=50, help="Number of features for LIME")
    parser.add_argument("--segmentation_alg", type=str, choices=['quickshift', 'slic'], default='quickshift', help="Segmentation algorithm for LIME")
    parser.add_argument("--kernel_size", type=int, default=2, help="Kernel size for segmentation algorithm")
    parser.add_argument("--max_dist", type=int, default=100, help="Max distance for segmentation algorithm")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio for segmentation algorithm")

    args = parser.parse_args()

    classify_image_and_explain(
        args.image_path,
        args.model_path,
        args.train_directory,
        args.num_samples,
        args.num_features,
        args.segmentation_alg,
        args.kernel_size,
        args.max_dist,
        args.ratio
    )
