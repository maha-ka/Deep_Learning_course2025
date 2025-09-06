# Deep_Learning_course2025

This repository contains a collection of practical assignments completed as part of a Deep Learning course. Each notebook explores a key concept in modern deep learning, from image classification and object detection to time-series modeling and generative adversarial networks (GANs).
The implementations are done in PyTorch and cover the workflow: data preprocessing, model design, training, evaluation, and visualization.


1. **Classification** (Classification_NetworkFromScratch.ipynb)
**Task: **Train a CNN from scratch to classify images of 15 types of vegetables (Kaggle dataset).
**Approach:**

- Data preprocessing and augmentation with torchvision (resizing, rotation, color jitter, normalization).

- Implemented a 5-layer CNN with batch normalization, max pooling, and fully connected layers.

- Trained with Cross-Entropy loss and Adam optimizer.
**Result:** Achieved ~96% accuracy with successful convergence.

2. **Classification with Pre-trained Architecture** (Classification_Pre-trainedArchitecture.ipynb)
**Task:** Apply transfer learning to the same classification dataset using ResNet101.
**Approach:**

- Replaced the final layer to match the dataset classes.

- Trained only the new layers first, then fine-tuned the whole network.
**Result:** ~93% accuracy after fine-tuning.

3. **Object Detection** (Object_Detection_CNN.ipynb)
**Task:** Object detection on a Fire and Smoke dataset from Roboflow using YOLOv11.
**Approach:**

- Used Ultralytics YOLOv11 framework with custom dataset.

- Evaluated using mAP50 and mAP50-95 metrics.
**Result:**

- Fire detection: mAP50 = 84.2%, strong performance.

- Smoke detection: mAP50 = 38.9%, weaker results due to limited data.
**Key Insight:** Increasing smoke-related training data and improving augmentation would enhance detection performance.

4. **Time Series Forecasting with RNNs** (Time_Series_RNN.ipynb)

**Task:** Predict future humidity values (weather dataset) using RNNs.
**Approach:**

- Preprocessed missing values, normalized data, and created time windows.

- Implemented a GRU model with dropout and fully connected layer.

- Used MSE loss and Adam optimizer.
**Result:**

- Achieved low error (MAE ≈ 0.0677).

- Predictions followed the actual trend but struggled with fine-grained details.
**Improvements:** Larger window size, higher dropout, or smoothing predictions.

5. **Image Generation with GANs** (Image_generation_part1_GAN_CIFAR.ipynb)
**Task:** Implement a basic GAN to generate CIFAR-10 images.
**Approach:**

- Generator: Fully connected layers with ReLU and Tanh activations.

- Discriminator: Fully connected layers with LeakyReLU and Sigmoid.

- Trained with binary cross-entropy loss and Adam optimizer.
**Result:** Training converged, but generated images were of low quality.

6. **Image Generation with DCGAN** (Image_generation_part2_DCGAN_CIFAR.ipynb)
**Task:** Improve GAN results using Deep Convolutional GAN (DCGAN).
**Approach:**

- Generator: Transposed convolutions with batch normalization and ReLU.

- Discriminator: Convolutional layers with LeakyReLU and dropout.
**Result:** Generated images were significantly better than GAN.
**Insight:** DCGANs leverage convolutional layers and batch normalization to produce more realistic results and train more stably.

**Technologies Used**

- Frameworks: PyTorch, Torchvision, Ultralytics YOLO

- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

- Datasets: Kaggle Vegetables Dataset, CIFAR-10, Roboflow Fire & Smoke, Weather (Humidity) Dataset
