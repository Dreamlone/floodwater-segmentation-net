import os

from pystac_client import Client
import planetary_computer as pc
from PIL import Image
from urllib.request import urlopen
import numpy as np
from geotiff import GeoTiff

Image.MAX_IMAGE_PIXELS = 1700000000

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
data = catalog.get_child("jrc-gsw")
items = data.get_items()


def return_zone_of_interest(geotiff_obj):
    bbox = geotiff_obj.tif_bBox_wgs_84
    print(bbox)
    p1 = list(bbox[0])
    p2 = [bbox[1][0], bbox[0][1]]
    p3 = list(bbox[1])
    p4 = [bbox[0][0], bbox[1][1]]
    return [p1, p2, p3, p4, p1]


def print_all_collections():
    collections = catalog.get_children()
    for collection in collections:
        print(f"{collection.id} - {collection.title}")


def print_items():
    i=0
    for item in items:
        print(i+1)
        i=i+1
        print(item.to_dict())


def print_av_layers():
    for item in items:
        print(item.assets.keys())
        break


def get_all_images(layer_name):
    for item in items:
        asset = item.assets[layer_name]
        signed_href = pc.sign(asset.href)
        response = urlopen(signed_href)
        pic = Image.open(response)
        pix = np.array(pic)
        print(pix)


def get_image_by_bbox(bbox: list):
    zone = {
                "type": "Polygon",
                "coordinates": [
                    bbox
                ],
            }
    search = catalog.search(collections=["jrc-gsw"], intersects=zone)
    items = search.get_items()
    for item in items:
        print(item.to_dict())
