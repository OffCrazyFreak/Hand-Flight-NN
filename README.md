# Hand-Flight-NN

## About
Airplane flight control via hand gestures using neural network. Control an airplane in a 3D environment using hand gestures captured through your webcam. The system uses a neural network to interpret hand positions and movements, translating them into flight controls.

## Demo
<div align="center">
    <img alt="Demo" src="https://github.com/user-attachments/assets/2dcd9b00-ce39-4c71-bdba-e4df6d70a527">
</div>

## Prerequisites

- Python 3.11.9
- Unity 2022.3.28f1
- Webcam (supports 16:9 or 3:2 aspect ratios)
- Operating System: Windows, macOS, or Linux

## Installation

1. **Python Setup**
   ```bash
   # Install Python dependencies
   pip install tensorflow==2.17.1
   pip install numpy
   pip install matplotlib
   pip install scikit-learn
   pip install opencv-python-headless
   pip install pycocotools
   pip install mediapipe
   pip install portalocker
   ```

2. **Neural Network Model**
   - Download the model weights file (`weights_test_model_val_0.43_240.weights.h5`) from the [Releases](https://github.com/OffCrazyFreak/Hand-Flight-NN/releases) page
   - Place it in the `release` folder

3. **Unity Game Build**
   - Open Unity Hub
   - Install Unity version 2022.3.28f1 if not already installed
   - Open the `igra` project in Unity
   - Go to File > Build Settings
   - Choose your target platform (Windows, macOS, or Linux)
   - Click "Build" and select the `release/Build` folder as the destination

3. **Release Folder Structure**
   Ensure your `release` folder contains:
   ```
   release/
   ├── Build/              # Unity game build files
   │   ├── neumre[.exe/.app/etc]  # Platform-specific executable
   │   └── neumre_Data/
   ├── start.py           # Python script for hand tracking
   └── weights_test_model_val_0.43_240.weights.h5  # Neural network weights
   ```

## How to Run

1. Navigate to the release folder
   ```bash
   cd release
   ```

2. Run the Python script
   ```bash
   python start.py
   ```

3. The Unity game will automatically create a `shared.txt` file for communication between the neural network and the game.

4. Use your hand gestures to control the airplane:
   - Keep your hand visible to the webcam
   - The neural network will track your hand position and orientation
   - The airplane will respond to your hand movements in real-time

## Troubleshooting

- Ensure your webcam is properly connected and accessible
- Check that `shared.txt` is being created in the correct location
- Verify that all Python dependencies are installed correctly
- Make sure you're using the correct versions of Python (3.11.9) and Unity (2022.3.28f1)
- If using Linux or macOS, ensure proper permissions are set for the executable and `shared.txt`

## Documentation
Detailed documentation can be found in the [docs](./docs) folder of the repository.

## Authors
- Tomislav Matanović
- Jakov Jakovac
- Leonarda Pribanić
- Daniel Košmerl
- Nikola Perić
- Domagoj Marić

## License
Licensed under [MIT License](./LICENSE)

<p align="right"><a href="#about">back to top ⬆️</a></p>
