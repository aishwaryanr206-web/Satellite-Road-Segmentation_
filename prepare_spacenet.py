import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform, transform_geom

from imagery import stretch_to_uint8


IMAGE_ID = re.compile(r"_img(\d+)\.tiff?$", re.IGNORECASE)


def read_road_geometries(path, target_crs):
    collection = json.loads(path.read_text(encoding="utf-8"))
    source_crs = collection.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    source_crs = CRS.from_user_input(source_crs)
    geometries = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry")
        if geometry and geometry.get("type") in {"LineString", "MultiLineString"}:
            geometries.append(transform_geom(source_crs, target_crs, geometry))
    return geometries


def prepare_tile(image_path, roads_path, image_output, mask_output, road_width):
    with rasterio.open(image_path) as source:
        if source.crs is None:
            raise ValueError(f"Image has no coordinate reference system: {image_path}")
        if source.count < 5:
            raise ValueError(f"Expected an 8-band PS-MS image: {image_path}")

        rgb = stretch_to_uint8(source.read((5, 3, 2)))
        geometries = read_road_geometries(roads_path, source.crs)
        shapes = ((geometry, 1) for geometry in geometries)
        mask = rasterize(
            shapes,
            out_shape=(source.height, source.width),
            transform=source.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )
        center_x, center_y = source.transform @ (source.width / 2, source.height / 2)
        crs = source.crs
        if crs.is_geographic:
            projected_x, projected_y = transform(crs, CRS.from_epsg(3857), [center_x], [center_y])
            center_x, center_y = projected_x[0], projected_y[0]

    if road_width > 1:
        kernel = np.ones((road_width, road_width), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if not cv2.imwrite(str(image_output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write prepared image: {image_output}")
    if not cv2.imwrite(str(mask_output), mask * 255):
        raise OSError(f"Could not write prepared mask: {mask_output}")
    return center_x, center_y, int(np.count_nonzero(mask))


def main():
    parser = argparse.ArgumentParser(description="Prepare georeferenced SpaceNet 3 road tiles.")
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--roads_dir", required=True, type=Path)
    parser.add_argument("--output_dir", default=Path("data_spacenet/processed"), type=Path)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--spatial_block_meters", type=float, default=5000)
    parser.add_argument("--road_width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1:
        parser.error("--validation_fraction and --test_fraction must be between 0 and 1.")
    if args.validation_fraction + args.test_fraction >= 1:
        parser.error("Validation and test fractions must sum to less than 1.")
    if args.spatial_block_meters <= 0 or args.road_width < 1:
        parser.error("--spatial_block_meters and --road_width must be positive.")
    if not args.images_dir.is_dir() or not args.roads_dir.is_dir():
        parser.error("Both --images_dir and --roads_dir must be existing directories.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            "--output_dir must be empty to prevent stale files from mixing spatial splits; "
            "choose a new directory."
        )

    image_paths = sorted(args.images_dir.glob("*.tif")) + sorted(args.images_dir.glob("*.tiff"))
    pairs = []
    missing = []
    for image_path in image_paths:
        match = IMAGE_ID.search(image_path.name)
        if not match:
            continue
        roads_path = args.roads_dir / f"SN3_roads_train_AOI_2_Vegas_geojson_roads_speed_img{match.group(1)}.geojson"
        if roads_path.is_file():
            pairs.append((image_path, roads_path))
        else:
            missing.append(image_path.name)
    if len(pairs) < 2:
        raise RuntimeError(f"Found only {len(pairs)} matching SpaceNet image/road-label pairs.")
    if missing:
        print(f"Skipped {len(missing)} images without road annotations.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    for image_path, roads_path in pairs:
        match = IMAGE_ID.search(image_path.name)
        tile_id = f"img{match.group(1)}"
        with rasterio.open(image_path) as source:
            crs = source.crs
            center_x, center_y = source.transform @ (source.width / 2, source.height / 2)
            if crs is None:
                raise ValueError(f"Image has no coordinate reference system: {image_path}")
            if crs.is_geographic:
                px, py = transform(crs, CRS.from_epsg(3857), [center_x], [center_y])
                center_x, center_y = px[0], py[0]
        block = (
            int(center_x // args.spatial_block_meters),
            int(center_y // args.spatial_block_meters),
        )
        processed.append((tile_id, image_path, roads_path, block))

    blocks = sorted({item[3] for item in processed})
    if len(blocks) < 3:
        raise RuntimeError(
            "The tiles occupy fewer than three spatial blocks; reduce --spatial_block_meters "
            "to create separate spatial training, validation, and test splits."
        )
    random.Random(args.seed).shuffle(blocks)
    target_validation = max(1, round(len(processed) * args.validation_fraction))
    target_test = max(1, round(len(processed) * args.test_fraction))
    validation_blocks = set()
    test_blocks = set()
    validation_count = 0
    test_count = 0
    for block in blocks:
        if (
            test_count < target_test
            and test_count + validation_count < len(processed) - 1
            and len(test_blocks) + len(validation_blocks) < len(blocks) - 1
        ):
            test_blocks.add(block)
            test_count += sum(item[3] == block for item in processed)
        elif (
            validation_count < target_validation
            and test_count + validation_count < len(processed) - 1
            and len(test_blocks) + len(validation_blocks) < len(blocks) - 1
        ):
            validation_blocks.add(block)
            validation_count += sum(item[3] == block for item in processed)
        if (
            test_count >= target_test
            and validation_count >= target_validation
        ) or len(test_blocks) + len(validation_blocks) == len(blocks) - 1:
            break
    if not validation_blocks or not test_blocks:
        raise RuntimeError(
            "Could not create non-empty spatial validation and test splits; reduce "
            "--spatial_block_meters or the requested split fractions."
        )

    split_counts = {"train": 0, "validation": 0, "test": 0}
    total_road_pixels = 0
    for tile_id, image_path, roads_path, block in processed:
        if block in test_blocks:
            split = "test"
        elif block in validation_blocks:
            split = "validation"
        else:
            split = "train"
        image_dir = args.output_dir / split / "images"
        mask_dir = args.output_dir / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        output_image = image_dir / f"{tile_id}.png"
        output_mask = mask_dir / f"{tile_id}.png"
        _, _, road_pixels = prepare_tile(
            image_path, roads_path, output_image, output_mask, args.road_width
        )
        split_counts[split] += 1
        total_road_pixels += road_pixels
        print(f"{split}: {tile_id} roads={road_pixels:,}")

    manifest = {
        "dataset": "SpaceNet 3 Road Network Detection, AOI 2 Vegas",
        "source": "https://spacenet.ai/spacenet-roads-dataset/",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International",
        "images": len(processed),
        "splits": split_counts,
        "spatial_block_meters": args.spatial_block_meters,
        "road_mask_dilation_pixels": args.road_width,
        "road_pixels": total_road_pixels,
        "color_bands": [5, 3, 2],
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
