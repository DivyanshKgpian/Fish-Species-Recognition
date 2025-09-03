# Fish-Species-Recognition
This project demonstrates a deep learning approach to classify fish species from images. It leverages transfer learning with a pre-trained ResNet50 model to achieve high accuracy on a dataset of 23 different fish species. The project includes data augmentation, model training, evaluation, and prediction on new images.
## Features
Model: Utilizes a ResNet50 model pre-trained on ImageNet for powerful feature extraction.  
Technique: Employs transfer learning by freezing the convolutional base and training a custom classifier head.  
Data Augmentation: Increases the dataset size and model robustness through transformations like rotation, shifting, zooming, and flipping.  
Performance: Achieves a high validation accuracy of approximately 96.3%.  

## Workflow Pipeline
The project follows a standard deep learning pipeline:  
Data Extraction: The initial dataset is extracted from a .tar archive.  
Data Augmentation: The ImageDataGenerator from Keras is used to create a larger, more diverse dataset from the original images.  
Train/Test Split: The augmented dataset is split into an 80% training set and a 20% testing set.  
Model Building: A sequential Keras model is built on top of the frozen ResNet50 base, with custom Dense layers for classification.   
Training: The model is trained using the Adam optimizer and categorical_crossentropy loss function.   
Evaluation: The model's performance is evaluated on the test set, and accuracy/loss curves are plotted.  
Prediction: The trained model is used to predict the species of a new, unseen fish image.  

## Technologies Used  
TensorFlow & Keras: For building and training the deep learning model.  
NumPy: For numerical operations, particularly for handling image arrays.
Matplotlib: For plotting the training history (accuracy and loss).
split-folders: A utility to split image datasets into training and validation sets.
OpenCV: For image processing tasks like reading and resizing.
Tarfile: For extracting the initial dataset.

## Results
The model was trained for 100 epochs, achieving a final validation accuracy of 96.34%.   
The training history shows that the model learned effectively and generalized well to the validation data.
<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/560e20d5-5aec-4a55-8a64-2a562bf0698e" />  
<img width="547" height="417" alt="image" src="https://github.com/user-attachments/assets/0ccde540-4bd4-455f-824f-993544ed130a" />
