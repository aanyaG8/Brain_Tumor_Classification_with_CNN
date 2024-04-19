Brain Tumor Classification with CNN


Overview
This project develops a Convolutional Neural Network (CNN) model to classify brain tumor images from MRI scans. The model predicts the presence of glioma tumor, meningioma tumor, pituitary tumor, or detects cases with no tumor.

Model Architecture
The CNN architecture consists of convolutional and pooling layers followed by fully connected layers. The model predicts multiple probabilities for each tumor type, and the highest probability determines the predicted tumor class.

Results
Accuracy: The model achieves 92% accuracy in classifying brain tumor images.
Prediction Process: The model assigns probabilities to each tumor class and selects the class with the highest probability as the predicted tumor type.

Usage
To use the model for inference:
1. Clone the repository.
2. Run the inference script and provide the path to the MRI image.
3. View the predicted tumor class and associated probabilities.

Future Work
Explore transfer learning techniques for improved performance.
Enhance robustness for noisy or low-quality MRI images.
Deploy the model as a web or mobile application for real-time classification.

Contributors
Aannya Gupta (GitHub)

License
This project is licensed under the MIT License.
