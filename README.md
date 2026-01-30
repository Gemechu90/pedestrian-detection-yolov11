Pedestrian Detection Using YOLO
📌 Overview

This project focuses on pedestrian detection using YOLO (You Only Look Once).
The main objective is to prepare a pedestrian dataset, convert annotations into YOLO format, visualize bounding boxes, and make the data ready for training YOLO-based object detection models.

The implementation is provided in a Jupyter Notebook.

📂 Project Structure
yolo-pedestrian/
│
├── yolo-pedestrian.ipynb
├── datasets/
│   ├── images/
│   ├── labels/
│   ├── data.yaml
│   └── README.md
│
└── README.md

🛠 Requirements

Python 3.x

NumPy

Pandas

OpenCV

Matplotlib

Seaborn

Plotly

SciPy

tqdm

Install dependencies:

pip install numpy pandas opencv-python matplotlib seaborn plotly scipy tqdm

⚙️ Workflow

Load pedestrian dataset

Extract bounding box annotations

Convert annotations to YOLO format

Split dataset into training and validation sets

Visualize bounding boxes

Prepare data for YOLO training

🎯 Application

Pedestrian detection

Intelligent transportation systems

Surveillance and smart city applications

