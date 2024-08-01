import gradio as gr
from predict_and_explain import classify_image_and_explain

# Utility function to strip quotes from paths
def strip_quotes(path):
    if isinstance(path, str):
        return path.strip('\'\"')

def classify_image_interface(image_path, model_path, train_directory, num_samples, num_features, segmentation_alg, kernel_size, max_dist, ratio):
    # Ensure inputs are correct
    print(f"Image: {image_path}")
    print(f"Model Path: {model_path}")
    print(f"Train Directory: {train_directory}")
    print(f"Number of Samples: {num_samples}")
    print(f"Number of Features: {num_features}")
    print(f"Segmentation Algorithm: {segmentation_alg}")
    print(f"Kernel Size: {kernel_size}")
    print(f"Max Distance: {max_dist}")
    print(f"Ratio: {ratio}")

    train_directory = strip_quotes(train_directory)

    # Run classification and explanation
    lime_mask, gradcam_image, predicted_label, probability = classify_image_and_explain(
        image_path,
        model_path,
        train_directory,
        num_samples,
        num_features,
        segmentation_alg,
        kernel_size,
        max_dist,
        ratio
    )

    return lime_mask, gradcam_image, predicted_label, f"{probability:.4f}"

explainer_interface = gr.Interface(
    fn=classify_image_interface,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),  # Change type to "file"
        gr.File(type="filepath", label="Model (e.g., /path/to/model.keras)"),
        gr.Textbox(label="Train dataset path"),
        gr.Slider(10, 1000, value=300, step=10, label="Number of Samples"),
        gr.Slider(10, 100, value=50, step=5, label="Number of Features"),
        gr.Dropdown(['quickshift', 'slic'], value='quickshift', label="Segmentation Algorithm"),
        gr.Slider(1, 10, value=2, step=1, label="Kernel Size"),
        gr.Slider(10, 200, value=100, step=10, label="Max Distance"),
        gr.Slider(0.1, 10.0, value=0.1, step=0.1, label="Ratio")
    ],
    outputs=[
        gr.Image(label="LIME Explanation"),
        gr.Image(label="GradCAM Explanation"),
        gr.Textbox(label="Predicted Label"),
        gr.Textbox(label="Probability Score")
    ],
    title="Image Classification with LIME & GRAD-CAM Explanation",
    description="Upload an image, provide the model path and training directory, and get the classification along with LIME and Grad-CAM explanations."
)

if __name__ == "__main__":
    explainer_interface.launch(debug=True)
