[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# No-Code Image Classifier

This project provides a no-code interface for developing image classification models using the TensorFlow framework. Use the `setup_and_run` script to set up and open the Gradio-based interface, which simplifies the process of developing and testing image classification models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started (Demo Video)](#getting-started-demo-video)
- [Project Structure](#project-structure)
- [Dataset Structure](#dataset-structure)
- [Setup and Run (no code)](#setup-and-run-no-code)
- [Setup and Run (by Command Line)](#setup-and-run-by-command-line)
- [Supported Base Models](#supported-base-models)
- [Example](#example)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.6 or higher

## Getting Started (Demo Video)

<p align="center">
  <img src="data/demo snap.png" alt="Preview">
</p>

<p style="text-align: center;">
  <a href="https://www.youtube.com/watch?v=znRVrnVDgD8" target="_blank">Watch the demo video to see how to use the no-code image classifier interface.</a>
</p>

## Project Structure

- `interface.py`: The Gradio interface script that provides a web interface for data loading, training, testing, and prediction.
- `data_loader.py`: Processes and splits data into training, testing, and validation sets. It also performs data augmentation if enabled by the user.
- `test.py`: Contains functions for evaluating the trained model on test data and generating performance metrics.
- `train.py`: Includes the logic for training the image classification model, including data preprocessing, model training, and saving the trained model.
- `predict.py`: Handles the prediction process, allowing the model to make predictions on new, unseen images.
- `requirements.txt`: Lists all the required Python packages for the project.

## Dataset Structure

```sh
├── Dataset (Raw)
   ├── class_name_1
   │   └── *.jpg
   ├── class_name_2
   │   └── *.jpg
   ├── class_name_3
   │   └── *.jpg
   └── class_name_4
       └── *.jpg
```

## Setup and Run (no code)

1. **Clone the Repository**

   If you haven’t already, clone the repository to your local machine:

   ```sh
   git clone https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier
   cd No-Code-Image-Classifier
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   It’s a good practice to use a virtual environment to manage your project's dependencies:

   ```sh 
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Run the Interface Script**

   Method-1:
   Click `setup_and_run.bat`, and use default setup to start the Gradio interface.

   Method-2:
   Install the dependencies:

   ```sh 
   pip install -r requirements.txt 
   ```
      
   Execute the `interface.py` script to start the Gradio interface:

   ```sh 
   python interface.py
   ```
   This script will:
   - Run the `interface.py` script to start the Gradio interface.

5. **Access the Gradio Interface**

   Once the Gradio interface is running, you will see a URL in the terminal. Open this URL in your web browser to access the no-code image classifier interface.

## Setup and Run (by Command Line)

1. **Clone the Repository**

   If you haven’t already, clone the repository to your local machine:

   ```sh
   git clone https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier
   cd No-Code-Image-Classifier
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   It’s a good practice to use a virtual environment to manage your project's dependencies:

   ```sh 
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the Dependencies**

   ```sh 
   pip install -r requirements.txt 
   ```

4. **Run the Scripts**

   - **Data Loading**:
     ```sh
     python data_loader.py --path "path/to/raw/dataset" --target_folder "path/to/target/folder" --dim 224 --batch_size 32 --num_workers 4 --augment_data
     ```

   - **Model Training**:
     ```sh
     python train.py --base_models VGG16, ResNet50 --shape 224 224 3 --data_path "path/to/processed/dataset" --log_dir "path/to/log/dir" --model_dir "path/to/model/dir" --epochs 100 --optimizer adam --learning_rate 0.0001 --batch_size 32
     ```

   - **Model Testing**:
     ```sh
     python test.py --model_path "path/to/trained/model" --model_dir "path/to/model/dir" --img_path "path/to/test/image" --log_dir "path/to/log/dir" --test_dir "path/to/test/dir" --train_dir "path/to/train/dir" --class_names "class1,class2,class3"
     ```

   - **Prediction**:
     ```sh
     python predict.py --model_path "path/to/trained/model" --img_path "path/to/image" --train_dir "path/to/train/dir"
     ```

## Supported Base Models

The following base models are supported for training:
- VGG16
- VGG19
- ResNet50
- ResNet101
- InceptionV3
- DenseNet121
- MobileNetV2
- Xception
- InceptionResNetV2
- EfficientNetB0

## Example

Here's an example workflow that demonstrates how to use the scripts for data loading, model training, testing, and prediction.

1. **Data Loading:**
   - With Data Augmentation: 
     ```sh
      python data_loader.py --path "path/to/raw/dataset" --target_folder "path/to/target/folder" --dim 224 --batch_size 32 --num_workers 4 --augment_data
      ```
    - Without Data Augmentation: 
      ```sh
      python data_loader.py --path "path/to/raw/dataset" --target_folder "path/to/target/folder" --dim 224 --batch_size 32 --num_workers 4
      ```

3. **Model Training:**
    ```sh
    python train.py --base_models VGG16,ResNet50 --shape 224 224 3 --data_path "path/to/processed/dataset" --log_dir "path/to/log/dir" --model_dir "path/to/model/dir" --epochs 100 --optimizer adam --learning_rate 0.0001 --batch_size 32
    ```

4. **Model Testing:**
    ```sh
    python test.py --model_path "path/to/trained/model" --model_dir "path/to/model/dir" --img_path "path/to/test/image" --log_dir "path/to/log/dir" --test_dir "path/to/test/dir" --train_dir "path/to/train/dir"
    ```

5. **Prediction:**
    ```sh
    python predict.py --model_path "path/to/trained/model" --img_path "path/to/image" --train_dir "path/to/train/dir"
    ```

## Troubleshooting
- **Dependencies Issues**: If you encounter issues with installing dependencies, ensure you have the correct version of Python and try running the `run.py` script again.
- **Script Errors**: If you encounter errors while running `interface.py`, check the script for any missing or misconfigured paths.

## Contributing
Feel free to fork the repository and submit pull requests. For any issues or feature requests, please open an issue on the GitHub repository.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

----

[contributors-shield]: https://img.shields.io/github/contributors/Purushothaman-natarajan/No-Code-Image-Classifier.svg?style=flat-square
[contributors-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Purushothaman-natarajan/No-Code-Image-Classifier.svg?style=flat-square
[forks-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier/network/members
[stars-shield]: https://img.shields.io/github/stars/Purushothaman-natarajan/No-Code-Image-Classifier.svg?style=flat-square
[stars-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/Purushothaman-natarajan/No-Code-Image-Classifier.svg?style=flat-square
[issues-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier/issues
[license-shield]: https://img.shields.io/github/license/Purushothaman-natarajan/No-Code-Image-Classifier.svg?style=flat-square
[license-url]: https://github.com/Purushothaman-natarajan/No-Code-Image-Classifier/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/purushothamann/
