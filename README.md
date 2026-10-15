# Aishwarya NR — Satellite Road Segmentation

Project by: Aishwarya NR

A starter implementation for road segmentation and GIS vectorization using PyTorch, ResNet-34 U-Net, OpenCV, Flask, Shapely, and ReportLab.

## Features
- ResNet-34 encoder + U-Net decoder with skip connections
- Focal-Tversky loss and a training script
- Flask web interface to upload a satellite image and run inference
- Result page previews the uploaded image beside a green road-prediction overlay
- Binary segmentation mask output
- Polygon vectorization to GeoJSON and KML
- Road pixel area and approximate length statistics
- PDF report generation

## Important limitations
1. Road extraction performance is dataset- and location-dependent. Report only metrics measured on a held-out split and include the dataset, split, threshold, and configuration.
2. The official SpaceNet 3 public test archive does not provide its evaluation labels; the preparation workflow therefore creates geographically blocked validation and test splits from the labeled training AOI.
3. GeoJSON/KML from a georeferenced GeoTIFF are transformed to EPSG:4326. Ordinary PNG/JPG inputs have no geographic reference and produce pixel-coordinate exports only.
4. Area/perimeter statistics are pixel-based. They are not square meters/meters, even when vector exports are georeferenced.

## Setup (Windows PowerShell)
```powershell
cd path\to\Satellite_Road_Segmentation_Project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Dataset format
Prepare a dataset with paired images and binary masks:
```
data/
  images/
    001.png
    002.png
  masks/
    001.png
    002.png
```
Images and masks must have matching filenames and dimensions. Masks should be black background (0) and white road pixels (255).

## SpaceNet 3 Vegas dataset
The project targets the official [SpaceNet 3 Road Network Detection dataset](https://spacenet.ai/spacenet-roads-dataset/). Its AOI 2 (Las Vegas) full-resolution image archive is about 25 GB, and the road-centerline annotations are published separately. SpaceNet states that its dataset is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); include attribution and the license when sharing results or derived material.

From the project directory in PowerShell, download and extract the two official training archives:
```powershell
New-Item -ItemType Directory -Force data_spacenet\source | Out-Null
Invoke-WebRequest `
  -Uri "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas.tar.gz" `
  -OutFile data_spacenet\source\SN3_roads_train_AOI_2_Vegas.tar.gz
Invoke-WebRequest `
  -Uri "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz" `
  -OutFile data_spacenet\source\SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz
tar.exe -xzf data_spacenet\source\SN3_roads_train_AOI_2_Vegas.tar.gz -C data_spacenet\source
tar.exe -xzf data_spacenet\source\SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz -C data_spacenet\source
```

Rasterize the labeled road centerlines into binary masks and form spatially separated training/validation/test sets. Use an empty output directory; preprocessing refuses to mix new results with stale split files:
```powershell
python prepare_spacenet.py `
  --images_dir data_spacenet\source\AOI_2_Vegas\PS-MS `
  --roads_dir data_spacenet\source\AOI_2_Vegas\geojson_roads_speed `
  --output_dir data_spacenet\processed `
  --validation_fraction 0.2 `
  --test_fraction 0.15 `
  --spatial_block_meters 5000 `
  --road_width 3
```
The preprocessing uses the visible red, green, and blue bands (5, 3, 2), a per-image robust intensity stretch, and a three-pixel road-centerline mask. The generated manifest records the source and processing settings.

Train:
```powershell
python train.py `
  --data_dir data_spacenet\processed\train `
  --val_dir data_spacenet\processed\validation `
  --epochs 30 --batch_size 4 --image_size 256
```
The best-validation-loss checkpoint is saved to `weights/road_resunet.pth`. Training augments tiles with right-angle rotations and flips, selects a probability threshold on the validation data, and stores the selected threshold and epoch in the checkpoint. CPU-only training on the full-resolution dataset can take many hours.

## Final evaluation
Use the separate spatial test split only after training and selecting the threshold on validation:
```powershell
python evaluate.py `
  --data_dir data_spacenet\processed\test `
  --checkpoint weights\road_resunet.pth
```
The evaluator reports pixel accuracy, precision, Dice/F1, IoU, and the threshold chosen on the separate validation split. Do not use the test split to tune hyperparameters. The project includes a small Massachusetts Roads baseline checkpoint from earlier testing; it is not a SpaceNet-trained model, and its metrics must not be presented as SpaceNet results. Retrain with the full SpaceNet data before claiming SpaceNet performance.

Run the preprocessing/vectorization tests with:
```powershell
python -m unittest discover -s tests -v
```

## GIS exports
- GeoJSON/KML from georeferenced GeoTIFFs use real longitude/latitude coordinates (EPSG:4326).
- Uploads without georeferencing yield pixel-coordinate placeholders and must not be treated as real-world map data.
- Displayed area/perimeter values remain pixel-based; ground-distance measurements require appropriate projected CRS and pixel resolution.

## Project structure
- `model.py` — ResNet-34 U-Net architecture and Focal-Tversky loss
- `train.py` — training loop
- `prepare_spacenet.py` — geospatial road-line rasterization and spatial train/validation split
- `imagery.py` — visible-band selection and image loading
- `evaluate.py` — metrics
- `vectorize.py` — mask to polygons and KML/GeoJSON exports
- `reporting.py` — PDF report generation
- `app.py` — Flask interface
- `templates/index.html` — upload/results UI
