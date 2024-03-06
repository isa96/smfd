from fastapi import APIRouter, File, UploadFile, Form
from typing import Optional
from PIL import Image
import urllib.request
from io import BytesIO
from routers.mask_inference import process


router = APIRouter()

def count_values(obj):

    if isinstance(obj, dict):
        count = 0
        for value in obj.values():
            count += count_values(value)
        return count
    elif isinstance(obj, list):
        count = 0
        for item in obj:
            count += count_values(item)
        return count
    else:
        return 1


@router.post("/inference")
async def run_inference(file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None),
                        model_in_use: str = Form('mask')):

    result = []
    if file:
        if file.content_type not in ["image/jpeg", "image/jpg"]:
            return {"error": "Invalid file type. Only JPG images are allowed."}

        # image = Image.open(BytesIO(await file.read()))
        image = await file.read()
        processing_time = 0
        if model_in_use == 'mask':
            result, processing_time = process(image)
        print(f"Processing time inference: {processing_time:.2f} seconds")

    elif image_url:
        with urllib.request.urlopen(image_url) as response:
            content_type = response.info().get_content_type()
            if content_type in ["image/jpeg", "image/jpg"]:
                # image = Image.open(BytesIO(response.read()))
                image = response.read()
            else:
                return {"error": "Invalid file type. Only JPG images are allowed."}

        processing_time = 0
        if model_in_use == 'mask':
            result, processing_time = process(image)

        print(f"Processing time inference: {processing_time:.2f} seconds")

    else:
        result = {"info": "No input provided"}

    return result


