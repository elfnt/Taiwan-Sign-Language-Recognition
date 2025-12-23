# Taiwan-Sign-Language-Recognition

This project implements a video-based Taiwan Sign Language (TSL) recognition pipeline using **MediaPipe hand landmarks** and a deep learning model.  
The system focuses on **dynamic sign language recognition** from continuous video input.

## Environment Setup
1. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows
```
2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage Pipeline

1. Prepare Video Data

    Place all raw sign language videos into the same directory (the working directory or the directory specified inside the scripts).
    
    Or you can find our videos here: [Link](https://drive.google.com/drive/folders/1Gx0JUWA-EwP-6VC8IdZFmPRf0ocS75oi?usp=drive_link)
    
2. Video Preprocessing

    Run the preprocessing script to standardize videos (e.g., resizing, trimming, format unification)
    ```bash
    python preprocess.py    
    ```

3. Video Flipping (Data Augmentation)

    Apply horizontal flipping to all videos for data augmentation:
    ```bash
    python flip.py
    ```

4. Generate Hand Landmarks with MediaPipe

    Extract hand landmarks from processed videos and generate a CSV dataset
    ```
    python gen_csv.py
    ```
    This step uses MediaPipe Hands to extract landmark coordinates from each frame. 

5. Data Augmentation and Model Training

    Perform landmark-based data augmentation and train the recognition model
    ```
    python train_aug.py
    ```
    After this step, the processed dataset and trained model will be generated.
6. Inference / Testing

    At this stage, all required data and trained models are already provided in this repository.
    You can directly run the following command to perform testing:
    ```
    python main.py
    ```