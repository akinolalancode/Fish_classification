# Fish_classification


This code outlines a complete machine learning pipeline for classifying fish diseases using a custom Deep Learning model built with PyTorch.
Here is a summary of the key components:
1. Data Preparation & Visualization
•	Dataset Source: The code mounts Google Drive to access fish images organized into train_split and test_split folders.
•	Format Handling: It automatically identifies multiple image formats (JPG, PNG, BMP, etc.) and organizes them into Pandas DataFrames for easy tracking.
•	Analysis: It uses Seaborn to plot the distribution of classes, ensuring the user understands the balance of the dataset.
2. Image Preprocessing & Augmentation
To improve the model's ability to generalize, the code applies heavy Data Augmentation to the training set, including:
•	Random horizontal/vertical flips and rotations (30°).
•	Color jitter (adjusting brightness, contrast, and saturation).
•	Perspective changes and random grayscale.
•	Normalization: Standardizes images using ImageNet mean and standard deviation values.
•	Splitting: The training data is split into 80% training and 20% validation.
3. Model Architecture (FishCustomNet)
The code defines a custom Convolutional Neural Network (CNN) consisting of:
•	4 Convolutional Blocks: Each block uses a Conv2d layer, BatchNorm2d (to stabilize learning), and ReLU activation. The channels increase from 64 up to 512.
•	Classifier Head: A series of fully connected (Linear) layers with Dropout (to prevent overfitting) and BatchNorm1d.
•	Output: The model outputs predictions for 8 distinct fish disease classes.
4. Training and Evaluation
•	Training Loop: Runs for 100 epochs using the Adam optimizer and Cross-Entropy Loss.
•	Monitoring: It tracks training/validation loss and accuracy in real-time using tqdm progress bars.
•	Visualization: After training, it generates:
o	Loss and Accuracy curves.
o	A Confusion Matrix to show which diseases the model confuses.
o	A Classification Report (Precision, Recall, and F1-score).
•	Testing: It includes a visual validation step that displays random test images with their "True" vs. "Predicted" labels (colored green for correct, red for incorrect).
5. Exporting
•	The final trained model weights are saved as fish_classifier_model.pth both locally in the Colab environment and to Google Drive for future deployment.

