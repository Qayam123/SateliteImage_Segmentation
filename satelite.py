import os
import leafmap
import torch
from samgeo import SamGeo, tms_to_geotiff
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

m = leafmap.Map(center=[34.9716, 83.5946], zoom=15)
m.add_basemap('SATELLITE')
if m.user_roi_bounds() is not None:
    bbox = m.user_roi_bounds()
else:
    bbox = [-95.3704, 39.6762, -95.368, 39.6775]

image = "./outputs/satprocess.tif"
tms_to_geotiff(output=image, bbox=bbox, zoom=20, source='Satellite',overwrite=True)
m.add_raster(image, layer_name='Image')

checkpoint = 'sam_vit_h_4b8939.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = SamGeo(
    checkpoint=checkpoint,
    model_type='vit_h',
    device=device,
    erosion_kernel=(3, 3),
    mask_multiplier=255,
    sam_kwargs=None,
)
mask = 'segment.tiff'
sam.generate(image, mask)
vector = 'segment.gpkg'
sam.tiff_to_gpkg(mask, vector, simplify_tolerance=None)
shapefile = 'segment.shp'
sam.tiff_to_vector(mask, shapefile)

style = {
    'color': '#3388ff',
    'weight': 2,
    'fillColor': '#7c4185',
    'fillOpacity': 0.5,
}
m.add_vector(vector, layer_name='Vector', style=style)

