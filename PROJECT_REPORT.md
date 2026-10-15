# Project Report: Aishwarya NR — Satellite Road Segmentation

## Project author
- Aishwarya NR

## Abstract
This project aims to identify road pixels in satellite imagery and convert predicted road regions into vector features that can be reviewed in GIS workflows. The proposed deep-learning architecture uses a ResNet-34 encoder with a U-Net decoder and skip connections. Focal-Tversky loss is selected to address the imbalance between road and background pixels. A Flask application provides image upload, inference, binary masks, vector exports, basic statistics, and automated PDF reporting.

## Problem statement
Road extraction from satellite images is challenging because roads may be narrow, partially occluded, unpaved, or visually similar to surrounding surfaces. Manual digitization is time-consuming. This system automates pixel-level road segmentation and provides a pathway to vector outputs for GIS analysis.

## Objectives
1. Segment road pixels from satellite imagery.
2. Preserve spatial detail through encoder-decoder skip connections.
3. Reduce class-imbalance effects using Focal-Tversky loss.
4. Convert masks to polygon features.
5. Export GeoJSON and KML files and summarize detected regions.
6. Provide a simple web interface and PDF report.

## Technology stack
- PyTorch and Torchvision: model architecture and training
- ResNet-34: pretrained feature encoder
- U-Net decoder: multi-scale feature reconstruction and skip connections
- Focal-Tversky Loss: imbalance-aware segmentation loss
- OpenCV: image loading, resizing, thresholding, contours
- Flask: web interface and inference API routes
- Shapely: polygon construction and geometry cleanup
- ReportLab: PDF report generation

## Workflow
1. Collect satellite images and matching road masks.
2. Resize and normalize images; convert masks to binary labels.
3. Train ResNet-34 U-Net using Focal-Tversky loss.
4. Run inference and threshold probabilities into a binary mask.
5. Extract contours and convert them to polygon features.
6. Export GeoJSON/KML, statistics JSON, and PDF report.
7. Evaluate with pixel accuracy, precision, Dice/F1, and IoU on held-out labels.

## Model architecture
ResNet-34 serves as the encoder, learning hierarchical visual features. U-Net-style decoder blocks upsample feature maps and fuse encoder features through skip connections to retain spatial information. The final 1×1 convolution produces one road logit per pixel. Sigmoid converts logits into probabilities.

## Evaluation
Do not use the previously supplied project figures (96.77% pixel accuracy, 79.78% precision, 69.69% Dice/F1) as measured results; they have not been reproduced. The existing checkpoint is a baseline trained on 64 Massachusetts Roads image/mask pairs and evaluated on 16 separate tiles. Its measured scores were 89.82% pixel accuracy, 24.18% precision, 32.30% Dice/F1, and 19.26% IoU at a validation-selected threshold of 0.58. These baseline figures are not SpaceNet results and are not production-grade.

The internship submission workflow targets the full SpaceNet 3 AOI 2 Vegas labeled training imagery, rasterizes its georeferenced road centerlines, and reserves geographically separated 5 km blocks for training, validation, and final testing. Use validation to select the model and threshold, and report final metrics from the untouched test blocks. SpaceNet's public challenge test imagery does not include its evaluation labels. SpaceNet-specific metrics are pending full-data preparation and retraining; do not attribute the baseline results above to SpaceNet.

## Outputs
- Binary segmentation mask
- GeoJSON vector features
- KML vector features
- Road polygon count and pixel-area/perimeter statistics
- Automated PDF report

## GIS accuracy note
When a georeferenced TIFF is uploaded, GeoJSON and KML coordinates are transformed from its source CRS to EPSG:4326. For PNG/JPG (or TIFF without a CRS), the output coordinates are image-pixel placeholders and are not valid longitude/latitude. Area and perimeter summary values remain in pixel units; valid physical measurements additionally require pixel resolution and suitable projected-CRS calculations.

## Limitations and future improvements
- Evaluate on held-out cities and imagery from different acquisition dates before claiming cross-region generalization.
- Convert pixel area/perimeter to physical units only when the source CRS and pixel dimensions are known.
- Generate road centerlines and connected road graphs, not only polygons.
- Profile tiled inference on large satellite images and tune overlap blending.
- Improve post-processing for small gaps, false positives, and fragmented roads.
- Record dataset source, train/validation/test split, thresholds, and reproducible metric calculations.

## Conclusion
The project demonstrates a deep-learning pipeline for road segmentation and mask-to-vector conversion. With a trained checkpoint, representative test data, and correct georeferencing, it can support semi-automated road mapping workflows.
