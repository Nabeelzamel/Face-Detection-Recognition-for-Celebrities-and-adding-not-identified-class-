Setup and Installation

Prerequisites

Python 3.8 or later
A working webcam
The following Python libraries:
OpenCV (cv2)
TensorFlow (tensorflow)
NumPy (numpy)
Installation Steps
Clone the repository:
bash

Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Place the dataset folder (105_classes_pins_dataset) in the project directory.
Ensure the pre-trained model (best_model.h5) and OpenCV face detection files (opencv_face_detector.pb and opencv_face_detector.pbtxt) are correctly placed.
Usage
Run the application:

bash
Copy code
python main.py
Controls:

Press E to exit the webcam feed.
Output:

The system will display a live video feed with rectangles around detected faces and predicted class labels.
Customization
Modify Dataset
To add more classes:

Add a new folder named pins_<class_name> in the 105_classes_pins_dataset directory.
Include sample images of the new class in the folder.
The system automatically updates the class map when restarted.
Adjust Cosine Similarity Threshold
Modify the threshold for classification by changing this section in the code:

python
Copy code
if max_similarity < 0.6:
    predicted_class_index = 0  # "not identified"
Technical Details
Face Detection:

Uses OpenCV's DNN module with a pre-trained model (opencv_face_detector).
Detects faces and draws bounding boxes with a confidence threshold of 70%.
Feature Extraction:

Leverages a TensorFlow model (best_model.h5) trained on a celebrity dataset.
Generates embeddings of dimension 128 for comparison.

Testing:
![![alt text](https://)](image.png)
Cosine Similarity:

Compares face embeddings with pre-computed class embeddings using:
Cosine Similarity
=
Embedding
1
⋅
Embedding
2
∥
Embedding
1
∥
∥
Embedding
2
∥
Cosine Similarity= 
∥Embedding 
1
​
 ∥∥Embedding 
2
​
 ∥
Embedding 
1
​
 ⋅Embedding 
2
​
 
​
 
Identifies the class with the highest similarity score.
Live Prediction:

Real-time predictions are displayed on the video feed with the predicted class name.
Limitations
Lighting Conditions: Accuracy may drop in poor lighting or with significant occlusions.
Threshold Sensitivity: Requires fine-tuning the similarity threshold for optimal results.
Dataset Bias: Performance is tied to the quality and diversity of the reference dataset.
Future Improvements
Implement multi-face tracking and simultaneous identification.
Add support for training on new datasets directly within the application.
Integrate with a database to save and retrieve user data dynamically.
License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

Acknowledgements
OpenCV
TensorFlow
Dataset sourced from the Pins dataset.
Contact
If you have questions or suggestions, feel free to reach out via email at nabeelzamel456@example.com.