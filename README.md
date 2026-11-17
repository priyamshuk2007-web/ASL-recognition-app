# ASL Fingerspelling Recognition

A Java desktop application that recognises American Sign Language (ASL) fingerspelling through a webcam and converts recognised letters into words and sentences in real time.

The project combines Python-based hand landmark detection and machine learning with Java-based inference and a desktop interface.

## What It Does

- Recognises ASL alphabet letters using a webcam.
- Builds words by combining recognised letters.
- Adds a space between words when the left hand is raised into the camera frame for approximately half a second.
- Uses the right hand for letter recognition.
- Ignores predictions that fall below a confidence threshold.

The project was built with the goal of exploring how computer vision and machine learning could support communication for Deaf and non-verbal people.

## How It Works

The application uses a combination of Python and Java.

1. **Hand Detection:** A Python script uses MediaPipe to detect 21 landmarks on each hand.
2. **Landmark Normalisation:** The landmarks are normalised relative to the wrist and scaled according to hand size.
3. **Communication:** The normalised landmark coordinates are sent to the Java application through a local socket connection on port `5000`.
4. **Classification:** A Random Forest classifier, trained using scikit-learn, predicts the corresponding ASL letter.
5. **ONNX Integration:** The trained classifier is exported to ONNX format and loaded in Java using ONNX Runtime.
6. **Letter Selection:** Java collects predictions while the hand remains in the camera frame. When the hand leaves, the application selects the winning prediction and appends the letter if its confidence is above the threshold.
7. **Word Separation:** Raising the left hand for approximately half a second inserts a space.

## Project Structure

```text
ASL-Recognition-App/
│
├── .gitignore
│
├── asl-app/
│   ├── pom.xml
│   ├── asl_letter_model.onnx
│   │
│   └── src/
│       ├── main/
│       │   ├── java/
│       │   │   └── com/priyam/aslapp/
│       │   │       ├── App.java
│       │   │       ├── LiveServer.java
│       │   │       └── OnnxTest.java
│       │   │
│       │   └── resources/
│       │       └── asl_letter_model.onnx
│       │
│       └── test/
│           └── java/
│               └── com/priyam/aslapp/
│                   └── AppTest.java
│
└── dataset/
    ├── asl_letter_model.onnx
    ├── classifer.py
    ├── hand_landmarker.task
    ├── landmarks.csv
    ├── model_export.py
    ├── script.py
    ├── socket_test.py
    └── test.py
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Java 11 | Desktop application and inference |
| Maven | Java dependency management |
| Swing | Desktop user interface |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand landmark detection |
| Python | Landmark extraction and model training |
| NumPy | Numerical operations |
| scikit-learn | Random Forest classification |
| skl2onnx | Exporting the classifier to ONNX |
| ONNX Runtime | Running the trained model in Java |

## Training and Results

The classifier was trained using images from the Kaggle **ASL Alphabet** dataset.

Instead of training directly on image pixels, the model uses normalised hand landmarks extracted using MediaPipe.

### Observed Results

- Achieved approximately **99.2% accuracy** on its own test split.
- This result is likely inflated because the dataset contains near-duplicate frames.
- Achieved 12/12 correct predictions on a separate independent test folder. This is a small test and does not establish general accuracy.
- In live webcam testing, approximately 10 letters worked reliably under suitable conditions.

### Current Limitations

Some visually similar hand shapes are frequently confused:

- C, O and D
- B and F
- P and Q
- Closed-fist letters such as A, S, N, M, T and E

J and Z are currently excluded because they require motion rather than static hand poses.

Other limitations:

- Accuracy depends on lighting, camera angle and hand positioning.
- The model has not been extensively tested across different people and environments.
- The application has only been tested on Windows.
- The Python execution code uses the Windows `py` launcher.

This is an experimental prototype, not a complete ASL translation system. It currently focuses on static alphabet fingerspelling rather than full sign language recognition.

## Requirements

Make sure you have the following installed:

- Java 11
- Maven
- Python
- A working webcam

### Python Dependencies

Install the required Python packages:

```bash
py -m pip install mediapipe opencv-python numpy scikit-learn skl2onnx
```

You also need the MediaPipe `hand_landmarker.task` model file.

## Running the Application

1. Clone this repository.
2. Install Java, Maven and the required Python dependencies.
3. Make sure the Python scripts and `hand_landmarker.task` file are available at the paths expected by the application.
4. Open the `asl-app` directory as a Maven project in your IDE.
5. Run `LiveServer.java`.

`LiveServer.java` starts the Python script and establishes communication with the Java application through the local socket connection.

Make sure port `5000` is available and that your webcam is accessible.

**Note:** The application currently uses the Windows Python `py` launcher, so the execution code may need modification on Linux or macOS.

## Dataset

The original Kaggle ASL Alphabet image dataset is not included in this repository.

You will need to download the dataset separately if you want to regenerate `landmarks.csv` and retrain the classifier.

The repository contains the extracted landmark data, training scripts, and exported ONNX model.

## Future Improvements

Some areas I would like to explore:

- Collecting additional training samples using a personal webcam, especially for letters that currently perform poorly.
- Adding finger-angle and inter-landmark distance features.
- Supporting motion-based letters such as J and Z.
- Extending recognition to numbers, common symbols and frequently used words.
- Improving classification between visually similar hand shapes.
- Adding cross-platform support for Windows, Linux and macOS.
- Testing with a wider range of users, cameras and lighting conditions.

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [ONNX Runtime](https://onnxruntime.ai/)
- [scikit-learn](https://scikit-learn.org/)
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

**Project Status:** Experimental / Learning Project

Built as a project to explore computer vision, machine learning, ONNX model integration and Java application development.