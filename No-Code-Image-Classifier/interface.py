__author__ = "Purushothaman Natarajan"
__copyright__ = "Copyright 2024, Purushothaman Natarajan"
__credits__ = ["Purushothaman Natarajan"]
__license__ = "MIT"
__version__ = "V1.0"
__maintainer__ = "Purushothaman Natarajan"
__email__ = "purushothamanprt@gmail.com"
__status__ = "pushed"


import gradio as gr
import subprocess

# Utility function to strip quotes from paths
def strip_quotes(path):
    if isinstance(path, str):
        return path.strip('\'\"')

# Utility function to run a command and handle errors
def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# Define wrapper functions for each script
def run_data_loader(path, target_folder, dim, batch_size, num_workers, augment_data):
    # Strip quotes from paths
    path = strip_quotes(path)
    target_folder = strip_quotes(target_folder)
    
    command = [
        "python", "data_loader.py",
        "--path", path,
        "--target_folder", target_folder,
        "--dim", str(dim),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers)
    ]
    if augment_data:
        command.append("--augment_data")
    
    return run_command(command)

def run_train(base_models, shape, data_path, log_dir, model_dir, epochs, optimizer, learning_rate, batch_size):
    if not base_models:
        return "Error: You must select at least one base model for training."

    if not shape or not data_path or not log_dir or not model_dir:
        return "Error: Shape, data path, log directory, and model directory are required."

    try:
        shape_values = list(map(int, shape.split()))
        if len(shape_values) != 3:
            return "Error: Shape must contain exactly three integers (Height Width Channels)"
    except ValueError:
        return "Error: Shape must be a space-separated string of integers"

    # Strip quotes from paths
    data_path = strip_quotes(data_path)
    log_dir = strip_quotes(log_dir)
    model_dir = strip_quotes(model_dir)
    
    command = [
        "python", "train.py",
        "--base_models", ','.join(base_models),
        "--shape", str(shape_values[0]), str(shape_values[1]), str(shape_values[2]),
        "--data_path", data_path,
        "--log_dir", log_dir,
        "--model_dir", model_dir,
        "--epochs", str(epochs),
        "--optimizer", optimizer,
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size)
    ]
    
    return run_command(command)

def run_test(model_path, model_dir, img_path, log_dir, test_dir, train_dir, class_names):
    # Strip quotes from paths
    model_path = strip_quotes(model_path)
    model_dir = strip_quotes(model_dir)
    img_path = strip_quotes(img_path)
    log_dir = strip_quotes(log_dir)
    test_dir = strip_quotes(test_dir)
    train_dir = strip_quotes(train_dir)
    
    command = [
        "python", "test.py",
        "--log_dir", log_dir
    ]
    if model_path:
        command.extend(["--model_path", model_path])
    if model_dir:
        command.extend(["--model_dir", model_dir])
    if img_path:
        command.extend(["--img_path", img_path])
    if test_dir:
        command.extend(["--test_dir", test_dir])
    if train_dir:
        command.extend(["--train_dir", train_dir])
    if class_names:
        command.extend(["--class_names"] + class_names.split(","))
    
    return run_command(command)

def run_predict(model_path, img_path, train_dir):
    # Strip quotes from paths
    model_path = strip_quotes(model_path)
    img_path = strip_quotes(img_path)
    train_dir = strip_quotes(train_dir)
    
    command = [
        "python", "predict.py",
        "--model_path", model_path,
        "--img_path", img_path,
        "--train_dir", train_dir
    ]
    
    return run_command(command)

# Create Gradio interfaces
data_loader_interface = gr.Interface(
    fn=run_data_loader,
    inputs=[
        gr.Textbox(label="Raw Dataset Path (Path to file)"),
        gr.Textbox(label="Target Folder (Path to directory)"),
        gr.Slider(minimum=1, maximum=512, value=224, label="Dimension"),
        gr.Slider(minimum=1, maximum=128, value=32, label="Batch Size"),
        gr.Slider(minimum=1, maximum=16, value=4, label="Number of Workers"),
        gr.Checkbox(label="Augment Data")
    ],
    outputs="text",
    title="Data Loader"
)

train_interface = gr.Interface(
    fn=run_train,
    inputs=[
        gr.CheckboxGroup(["VGG16", "VGG19", "ResNet50", "ResNet101", "InceptionV3", "DenseNet121", "MobileNetV2", "Xception", "InceptionResNetV2", "EfficientNetB0"], label="Base Models"),
        gr.Textbox(value="224 224 3", label="Shape (Height Width Channels)"),
        gr.Textbox(label="Processed Dataset Path (Path to file)"),
        gr.Textbox(label="Log Directory (Path to directory)"),
        gr.Textbox(label="Model Directory (Path to directory)"),
        gr.Slider(minimum=1, maximum=1000, value=100, label="Epochs"),
        gr.Dropdown(["adam", "sgd"], label="Optimizer"),
        gr.Number(value=0.0001, label="Learning Rate"),
        gr.Slider(minimum=1, maximum=128, value=32, label="Batch Size")
    ],
    outputs="text",
    title="Training"
)

test_interface = gr.Interface(
    fn=run_test,
    inputs=[
        gr.Textbox(label="Model Path (Path to directory)"),
        gr.Textbox(label="Model Directory (Path to directory, optional)"),
        gr.Image(type="filepath", label="Image Path (optional, choose image)"),
        gr.Textbox(label="Log Directory (Path to directory)"),
        gr.Textbox(label="Test Directory (optional, Path to directory)"),
        gr.Textbox(label="Train Directory (optional, Path to directory)"),
        gr.Textbox(label="Class Names (optional, comma separated)")
    ],
    outputs="text",
    title="Testing"
)

predict_interface = gr.Interface(
    fn=run_predict,
    inputs=[
        gr.File(label="Model Path (choose model)", type="filepath"),
        gr.Image(type="filepath", label="Image Path (choose image)"),
        gr.Textbox(label="Train Directory (Path to directory)")
    ],
    outputs="text",
    title="Prediction"
)

gr.TabbedInterface([data_loader_interface, train_interface, test_interface, predict_interface], ["Data Loader", "Training", "Testing", "Prediction"]).launch(debug=True)
