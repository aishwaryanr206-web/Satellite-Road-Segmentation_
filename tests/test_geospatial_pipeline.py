import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from imagery import read_rgb_image
from prepare_spacenet import main as prepare_spacenet_main, prepare_tile
from train import RoadDataset
from vectorize import mask_to_vectors


class GeospatialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_path = self.root / "spacenet.tif"
        self.roads_path = self.root / "roads.geojson"
        self._write_fixture()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_fixture(self, image_path=None, roads_path=None, origin_x=500000):
        image_path = image_path or self.image_path
        roads_path = roads_path or self.roads_path
        width = height = 64
        transform_affine = from_origin(origin_x, 4000000, 1, 1)
        row, column = np.indices((height, width))
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=8,
            dtype="uint16",
            crs="EPSG:32611",
            transform=transform_affine,
        ) as destination:
            for band in range(1, 9):
                destination.write((1000 + band * 100 + row * 3 + column).astype(np.uint16), band)

        xs, ys = transform(
            "EPSG:32611",
            "EPSG:4326",
            [origin_x + 5.5, origin_x + 58.5],
            [3999968.5, 3999968.5],
        )
        feature_collection = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {"road_type": "5"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[xs[0], ys[0]], [xs[1], ys[1]]],
                    },
                }
            ],
        }
        roads_path.write_text(json.dumps(feature_collection), encoding="utf-8")

    def test_space_net_lines_rasterize_and_dataset_loads(self):
        image_path = self.root / "images" / "img1.png"
        mask_path = self.root / "masks" / "img1.png"
        image_path.parent.mkdir()
        mask_path.parent.mkdir()
        prepare_tile(self.image_path, self.roads_path, image_path, mask_path, road_width=3)

        with rasterio.open(self.image_path) as source:
            expected_shape = (source.height, source.width, 3)
        rgb = read_rgb_image(self.image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(rgb.shape, expected_shape)
        self.assertEqual(mask.shape, expected_shape[:2])
        self.assertGreater(np.count_nonzero(mask), 0)
        dataset = RoadDataset(self.root, image_size=32)
        image_tensor, mask_tensor = dataset[0]
        self.assertEqual(tuple(image_tensor.shape), (3, 32, 32))
        self.assertEqual(tuple(mask_tensor.shape), (1, 32, 32))
        self.assertTrue(set(mask_tensor.unique().tolist()).issubset({0.0, 1.0}))

    def test_single_band_uint16_tiff_is_stretched_for_inference(self):
        image_path = self.root / "grayscale.tif"
        pixels = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 10
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(-120, 40, 0.001, 0.001),
        ) as destination:
            destination.write(pixels, 1)

        rgb = read_rgb_image(image_path)

        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb.min()), 0)
        self.assertEqual(int(rgb.max()), 255)

    def test_georeferenced_vectors_use_longitude_latitude(self):
        image_path = self.root / "images" / "img1.png"
        mask_path = self.root / "masks" / "img1.png"
        image_path.parent.mkdir()
        mask_path.parent.mkdir()
        prepare_tile(self.image_path, self.roads_path, image_path, mask_path, road_width=3)
        output_dir = self.root / "vectors"

        stats, geojson_path, kml_path = mask_to_vectors(
            mask_path, output_dir, georeference=self.image_path
        )

        collection = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
        self.assertTrue(stats["georeferenced"])
        self.assertGreater(stats["polygon_count"], 0)
        self.assertTrue(collection["features"])
        coordinates = collection["features"][0]["geometry"]["coordinates"][0]
        self.assertTrue(all(-180 <= point[0] <= 180 for point in coordinates))
        self.assertTrue(all(-90 <= point[1] <= 90 for point in coordinates))
        self.assertIn("<coordinates>", Path(kml_path).read_text(encoding="utf-8"))

    def test_preprocessor_assigns_three_spatial_splits(self):
        images_dir = self.root / "source_images"
        roads_dir = self.root / "source_roads"
        images_dir.mkdir()
        roads_dir.mkdir()
        for tile_id, origin_x in ((1, 500000), (2, 510000), (3, 520000)):
            image_path = images_dir / f"SN3_roads_train_AOI_2_Vegas_PS-MS_img{tile_id}.tif"
            roads_path = roads_dir / (
                f"SN3_roads_train_AOI_2_Vegas_geojson_roads_speed_img{tile_id}.geojson"
            )
            self._write_fixture(image_path, roads_path, origin_x)

        output_dir = self.root / "prepared"
        arguments = [
            "prepare_spacenet.py",
            "--images_dir", str(images_dir),
            "--roads_dir", str(roads_dir),
            "--output_dir", str(output_dir),
            "--validation_fraction", "0.2",
            "--test_fraction", "0.2",
            "--spatial_block_meters", "5000",
        ]
        with patch.object(sys, "argv", arguments):
            prepare_spacenet_main()

        manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["images"], 3)
        self.assertEqual(set(manifest["splits"]), {"train", "validation", "test"})
        self.assertTrue(all(count == 1 for count in manifest["splits"].values()))
        with contextlib.redirect_stderr(io.StringIO()):
            with patch.object(sys, "argv", arguments), self.assertRaises(SystemExit) as error:
                prepare_spacenet_main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
