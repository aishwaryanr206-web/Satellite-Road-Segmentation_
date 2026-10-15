import json, math
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import mapping, shape
import rasterio
from rasterio.warp import transform_geom

def mask_to_vectors(mask_path, output_dir, min_area=8, pixel_size=None, origin=(0,0), georeference=None):
    """Vectorize contours to GeoJSON/KML; georeference TIFF inputs when available."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(mask_path)
    affine=None; source_crs=None
    if georeference is not None:
        with rasterio.open(georeference) as source:
            if source.crs is None:
                raise ValueError("Uploaded GeoTIFF does not contain a coordinate reference system.")
            if (source.height,source.width)!=mask.shape:
                raise ValueError("GeoTIFF dimensions do not match the predicted road mask.")
            affine=source.transform; source_crs=source.crs
    binary=(mask>127).astype(np.uint8)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    features=[]; polygons=[]; total_area=0.0; total_perimeter=0.0
    for c in contours:
        if cv2.contourArea(c)<min_area or len(c)<3: continue
        coords=c[:,0,:].astype(float)
        pixel_ring=[[float(x+origin[0]),float(y+origin[1])] for x,y in coords]
        if pixel_ring[0]!=pixel_ring[-1]: pixel_ring.append(pixel_ring[0])
        from shapely.geometry import Polygon
        poly=Polygon(pixel_ring)
        if not poly.is_valid: poly=poly.buffer(0)
        if poly.is_empty or poly.area<=0: continue
        parts=[poly] if poly.geom_type=="Polygon" else [part for part in poly.geoms if part.geom_type=="Polygon"]
        for part in parts:
            if part.is_empty or part.area<=0: continue
            total_area+=float(part.area); total_perimeter+=float(part.length)
            if affine is None:
                output_geometry=part
            else:
                world_ring=[list(affine @ (x,y)) for x,y in part.exterior.coords]
                output_geometry=Polygon(world_ring)
                if not output_geometry.is_valid: output_geometry=output_geometry.buffer(0)
                output_geometry=shape(transform_geom(source_crs,"EPSG:4326",mapping(output_geometry)))
            geometry_parts=[output_geometry] if output_geometry.geom_type=="Polygon" else [
                item for item in output_geometry.geoms if item.geom_type=="Polygon"
            ]
            for geometry_part in geometry_parts:
                polygons.append(geometry_part)
                features.append({"type":"Feature","properties":{"pixel_area":float(part.area)},"geometry":mapping(geometry_part)})
    fc={"type":"FeatureCollection","name":"road_segments","features":features}
    geojson=output_dir/"roads.geojson"; geojson.write_text(json.dumps(fc,indent=2),encoding="utf-8")
    # KML coordinates convention is lon,lat. Pixel coordinates here are placeholders only.
    kml_parts=['<?xml version="1.0" encoding="UTF-8"?>','<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Road Segments</name>']
    for poly in polygons:
        pts=" ".join(f"{x},{y},0" for x,y in poly.exterior.coords)
        kml_parts.append(f"<Placemark><name>Road polygon</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{pts}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    kml_parts.append("</Document></kml>")
    kml=output_dir/"roads.kml"; kml.write_text("\n".join(kml_parts),encoding="utf-8")
    stats={"polygon_count":len(polygons),"road_area_pixels":total_area,"polygon_perimeter_pixels":total_perimeter,
           "pixel_size":pixel_size,"georeferenced":affine is not None,
           "coordinate_system":"EPSG:4326" if affine is not None else "image pixels",
           "note":"Pixel units only unless georeferencing and pixel size are supplied."}
    (output_dir/"statistics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    return stats, str(geojson), str(kml)
