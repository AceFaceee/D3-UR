# Ewaste-Net: Structured OCR Pipeline for Electronic Waste Recognition

**Authors**: Robert Ke, Rong Gu, Yijun Sun, Shengjie Wang, Yuyang Wang  
**Sponsor**: D3 Embedded  
**Course**: Capstone 383W  
**Supervisor**: Prof. Cantay Caliskan  

---

##  Project Overview

**Ewaste-Net** is an end-to-end, modular deep learning system designed to extract and structure information from photographs of discarded electronics. The pipeline is tailored for real-world recycling facility use and is optimized for robustness, speed, and precision across diverse conditions.

---

##  Pipeline Summary

The pipeline consists of three tightly coupled stages:

1. **Stage I — Text Region Detection**  
   Uses a YOLOv8-based object detector trained on rotated bounding boxes (OBB) to localize small and angled text (e.g., serial numbers, model tags) in cluttered device images.

2. **Stage II — OCR with Qwen 3B**  
   Cropped image snippets are passed into a multilingual vision-language model (Qwen2.5-VL-3B) to transcribe text. Output is benchmarked against human-labeled ground truth using WER/CER.

---

Continue below for detailed descriptions of each stage.



## Stage I: Customized Text-Oriented Bounded Box Detection

The first stage of **Ewaste-Net** focuses on localizing embedded textual information on discarded electronics using a YOLOv8-based object detection framework. This stage is implemented in [`ewaste_net.py`](./ewaste_net.py).

### 📂 Code Location

- [`ewaste_net.py`](./ewaste_net.py): Contains the full workflow for training and deploying a YOLOv8 OBB (oriented bounding box) model on Roboflow. It also includes:
  - Setup and environment configuration (via Google Colab or local scripts)
  - Dataset loading and preprocessing via Roboflow API
  - Inference on local or cloud images
  - Cropping and saving detected text regions for OCR processing

> ⚠ This script was designed for a Colab workflow but is easily adaptable for local or cloud execution. It requires access to Roboflow API and an Inference server (local or hosted).

### Datasets Used

- **TextOBB-1681**  
  A curated dataset with 1,681 rotated bounding box (OBB) labels targeting embedded text on devices.  
  👉 [Browse on Roboflow](https://app.roboflow.com/d3ewastedataset/d3-merged-dataset-obb-only/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

- **Ambient-125**  
  A supplementary dataset capturing environmental variation (blur, lighting, angle).  
  👉 [Browse on Roboflow](https://app.roboflow.com/dscc391-aqjzv/d3-ocr-ambient/3)

- **Ewaste-70**  
  Real-world device images collected from recycling facilities (courtesy of D3).  
  👉 [Preprocessing & Cropping](https://app.roboflow.com/d3ewastedataset/d3-ewaste-dataset/generate/preprocessing)

### Features

- **Model**: YOLOv8 (Roboflow 3.0) with rotation-sensitive bounding boxes
- **Augmentations**:
  - Perspective shifts
  - Zoom & tilt
  - Rotation-invariant bounding boxes
- **Output**:
  - Cropped text images extracted from device photos
  - Optimized for OCR (Stage II)

This bounding box localization module drastically reduces image complexity for downstream text recognition, enabling faster and more accurate results.

## Stage II: Optical Character Recognition (OCR) with Qwen 3B

After text regions are localized and cropped in Stage I, they are passed into a lightweight vision-language OCR module powered by **Qwen 3B**. This step transcribes raw pixel regions into clean, machine-readable strings containing product names, model numbers, and serial codes.

### Code Location

- [`qwenocr_2.py`](./qwenocr_2.py): Inference script using `Qwen2.5-VL-3B-Instruct` from HuggingFace.
  - Loads the pre-trained Qwen model and processor
  - Performs inference on cropped images in batch
  - Saves predictions to `ocr_results.csv`

> 💡 Note: Make sure to install the required packages:
> ```bash
> pip install git+https://github.com/huggingface/transformers
> pip install qwen-vl-utils
> ```

### Ground Truth for Evaluation

- [`Qwen3b_GT.csv`](./Qwen3b_GT.csv):  
  This CSV contains human-labeled ground truth text for each cropped image. It is used to compute character-level and word-level benchmarking metrics:
  - `file_name`: image file name
  - `recognized_text`: output by Qwen
  - `Human-recognized Text`: manually verified ground truth

This dataset serves as the reference for model benchmarking and ablation comparisons.

### Evaluation Metrics

- [`Evaluation_method.py`](./Evaluation_method.py):  
  Custom metric functions to benchmark OCR output:
  - **CER** (Character Error Rate)
  - **WER** (Word Error Rate)
  - **VER** (Visual Error Rate, if applicable)

These metrics quantify the fidelity of transcription against ground truth labels across 1,600+ device samples. Preliminary results show a **WER below 1.1** on average, with strong performance even in noisy and low-resolution conditions.

### ⚙Highlights

- **Model**: Qwen2.5-VL-3B-Instruct (efficient multilingual vision-language model)
- **Input**: Cropped images from Stage I (`*.jpg`, `*.png`)
- **Output**: Text strings written to `ocr_results.csv`
- **Average Accuracy**: High fidelity in transcribing model numbers and identifiers under real-world noise

> This stage is critical in bridging unstructured visual content into structured identifiers for downstream structuring.

---

## Experiment 2: OCR Benchmarking Trials

The `Experiment_2/` folder contains additional OCR benchmarking experiments used during the model selection phase for Stage II.

- Models trialed include: Tesseract, EasyOCR, Keras-OCR, and Doctr
- Each model was evaluated on the same set of cropped inputs for comparative CER/WER performance
- Final selection of **Qwen2.5-VL-3B** was based on empirical accuracy, inference time, and robustness to non-standard fonts

 Every code module in this repository is provided in both `.py` and `.ipynb` formats to support both script-based execution and interactive development.

These experiments validate the architectural choices and serve as reproducible baselines for future work or ablation studies.



