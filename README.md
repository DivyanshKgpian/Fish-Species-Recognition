# Fish-Species Recognition and Tracking
This project implements a dual-system for fish analysis: a deep learning model for species classification from static images and a real-time object detection and tracking system for video.   
The classification component uses transfer learning with a ResNet50 model to identify 23 different fish species with high accuracy.   
The detection and tracking component uses a custom-trained YOLOv8 model with a BoT-SORT tracker to locate and follow fish in real-time.  
## Features
### Image Classification (ResNet50)  
Model: Utilizes a ResNet50 model pre-trained on ImageNet for powerful feature extraction.  
Technique: Employs transfer learning by freezing the convolutional base and training a custom classifier head.  
Data Augmentation: Increases the dataset size and model robustness through transformations like rotation, shifting, zooming, and flipping.  
Performance: Achieves a validation accuracy of approximately 96.3%.  
  
### Object Detection and Tracking (YOLOv8)  
Custom Dataset: Constructed a custom dataset with more than 500 images using the Label-Img tool and executed object tracking with OpenCV.  
High-Performance Tracking: Trained the YOLOv8 architecture with the BoT-SORT tracker to achieve a precision of 92.99%, a mean IoU of 0.76, and an mAP50 of 85.59%.  
  
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
Technologies Used  
Deep Learning: TensorFlow, Keras, YOLOv8    
Computer Vision: OpenCV  
Data & ML: NumPy, Matplotlib, split-folders  
Annotation & Tracking: Label-Img, BoT-SORT  
  
## Results
The model was trained for 100 epochs, achieving a final validation accuracy of 96.34%.   
The training history shows that the model learned effectively and generalized well to the validation data.
<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/560e20d5-5aec-4a55-8a64-2a562bf0698e" />  
<img width="547" height="417" alt="image" src="https://github.com/user-attachments/assets/0ccde540-4bd4-455f-824f-993544ed130a" />
