					Applying Neural Network Using PyTorch On Iris Dataset

-Data Loading and Preprocessing:
	Load the dataset and normalize features using StandardScaler.
	Encode the target species as numerical values.
	Split the data into training and validation sets.

-Model Definition:
	Define a neural network with two hidden layers using PyTorch.
	Use ReLU activation for hidden layers and softmax activation for the output layer.

-Model Training:
	Train the model using the Adam optimizer and Cross-Entropy Loss.
	Track and plot training and validation loss over epochs.

-Evaluation:
	Evaluate the model using accuracy, a confusion matrix, and a classification report.
	Plot the ROC curve and calculate the AUC score for each class.

-Model Architecture
	-The neural network has:
		An input layer with 4 neurons (for each feature).
		Two hidden layers with 16 and 8 neurons, respectively, using ReLU activation.
		An output layer with 3 neurons (one for each class) using softmax activation for multi-class classification.

	-Training Process
		The model is trained for 500 epochs with a learning rate of 0.01.
		The training process includes computing loss and accuracy at each epoch, with validation loss recorded.

	-Evaluation
		After training, the model is evaluated using:
			Accuracy: Measured on the validation set.
			Confusion Matrix: Displays the model's performance per class.
			Classification Report: Provides precision, recall, and F1-score.
			ROC Curve: Plots the ROC curve for each class with AUC scores.

	-Results
		The model's performance is assessed using the validation accuracy and the ROC-AUC score for each class.
		A confusion matrix and classification report provide additional details on the model's predictions.4
		![![alt text](https://)](image.png)
		![![alt text](https://)](image-1.png)