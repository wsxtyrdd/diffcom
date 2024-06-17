import os
from pathlib import Path
from PIL import Image
import shutil

def convert_to_webp(file_path: Path):
    img = Image.open(file_path).convert('RGB')
    new_path = str(file_path).replace(file_path.suffix, '.webp')
    img.save(new_path, 'webp')

all_files = Path('public/imgs').glob('**/*')
all_imgs = [x for x in all_files if str(x).endswith(".png") or str(x).endswith(".jpg")]

for img_path in all_imgs:
    convert_to_webp(img_path)
    os.remove(img_path)
