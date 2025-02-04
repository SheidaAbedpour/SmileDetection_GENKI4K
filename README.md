# Smile Detection using DeepFace and Transfer Learning
This project is designed to detect smiles in facial images using deep learning techniques. It leverages the `DeepFace` library for facial feature extraction and a custom neural network for smile classification. The project is divided into two main parts:

### 1. Face Detection and Alignment:
For face detection, the `RetinaFace` model was used due to its high accuracy and robustness. This model performs:
- Face detection: Identifies faces in images.
- Face alignment: Adjusts facial landmarks to standardize input images.

### 2. Smile Detection:
The aligned faces are processed to extract facial embeddings using the `DeepFace` library. These embeddings are then used to train a neural network classifier to distinguish between smiling and non-smiling faces.

To classify smiles, two `DeepFace` models were tested, including:
- `FaceNet`
- `VGG-Face`


#### Model Performance

| **Model**      | **Training Accuracy** | **Test Accuracy** |
|----------------|-----------------------|-------------------|
| **FaceNet512** | 82-85%                | 76-80%            |
| **VGG-Face**   | ~98%                  | 90-91%            |


Given the high performance of **VGG-Face**, it was chosen as the feature extractor for the final smile classification model.



## Project Structure
### `Face Detection`:

- Input images are stored in a directory.

- Faces are detected and aligned using RetinaFace.

- Aligned faces are saved in a separate directory.

### `Smile Detection`:

- Aligned faces are processed to extract facial embeddings using `DeepFace`.

- A neural network classifier is trained on these embeddings to detect smiles.

- The trained model is saved for future use.



## Transfer Learning
Transfer learning is a machine learning technique where a pre-trained model is used as a feature extractor instead of training a model from scratch. These extracted features were then passed to a classifier (e.g., a fully connected neural network) to make predictions. This project uses `VGG-Face`, a deep learning model trained on a large dataset of faces, to extract facial embeddings.

### Why Use Transfer Learning?
- #### Reduces training time:
  Instead of training a deep network from scratch, we use embeddings from a pre-trained model.
- #### Improves accuracy:
   VGG-Face embeddings already capture essential facial features, making classification easier.
- #### Works with smaller datasets:
   Since the base model is already trained on a large dataset, we need fewer training samples.

### Overfitting Prevention
Overfitting was a concern, as the dataset was relatively small. To mitigate this:
- `Data augmentation` was applied (random rotations, flips, and brightness changes).
- `Dropout layers` were added to prevent over-reliance on specific features.

## Requirements
To run this project, install the necessary dependencies:
```bash
pip install deepface tensorflow opencv-python numpy scikit-learn keras
```
