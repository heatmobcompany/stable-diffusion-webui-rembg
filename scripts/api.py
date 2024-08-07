from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

from modules.api.models import *
from modules.api import api
import gradio as gr
from PIL import Image
import numpy as np
import io
import base64

import rembg

import time
try:
    from helper.logging import Logger
    logger = Logger("REMBG")
except Exception:
    import logging
    logger = logging.getLogger("REMBG")

# models = [
#     "None",
#     "u2net",
#     "u2netp",
#     "u2net_human_seg",
#     "u2net_cloth_seg",
#     "silueta",
# ]

class ImageRequest(BaseModel):
    image: str
    width: int = None
    height: int = None

def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
        input_image: str = Body("", title='rembg input image'),
        model: str = Body("u2net", title='rembg model'), 
        return_mask: bool = Body(False, title='return mask'), 
        alpha_matting: bool = Body(False, title='alpha matting'), 
        alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'), 
        alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'), 
        alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        logger.info("===== API/rembg start =====")
        start_time = time.time()
        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)

        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        logger.info("===== API /rembg end in {:.3f} seconds =====".format(time.time() - start_time))
        return {"image": api.encode_pil_to_base64(image).decode("utf-8")}


    @app.post("/sdapi/v2/rembg")
    async def rembg_remove_v2(
        input_image: str = Body("", title='rembg input image'),
        model: str = Body("u2net", title='rembg model'), 
        return_mask: bool = Body(False, title='return mask'), 
        alpha_matting: bool = Body(False, title='alpha matting'), 
        alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'), 
        alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'), 
        alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        logger.info("===== API /sdapi/v2/rembg start =====")
        start_time = time.time()
        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)
        input_image = input_image.convert("RGB")

        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        left, upper, right, lower = image.getbbox()
        logger.info("===== API /sdapi/v2/rembg end in {:.3f} seconds =====".format(time.time() - start_time))
        return  {
                    "image": api.encode_pil_to_base64(image).decode("utf-8"),
                    "box": {
                        "x": left, "y" : upper, "width" : right - left, "height" : lower - upper,
                    }
                }

    def rgba_to_rgb(img):
        img_np = np.array(img)
        
        # Ensure the input image is in RGBA format
        if img_np.shape[2] != 4:
            raise ValueError("Input image must have 4 channels (RGBA)")

        r, g, b, a = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], img_np[:, :, 3]
        
        # Mask to find foreground pixels
        mask = a > 250
        foreground_r = r[mask]
        foreground_g = g[mask]
        foreground_b = b[mask]

        if len(foreground_r) == 0:
            raise ValueError("No foreground pixels found.")

        # Calculate average foreground brightness
        avg_r = np.mean(foreground_r)
        avg_g = np.mean(foreground_g)
        avg_b = np.mean(foreground_b)
        avg_foreground = (avg_r + avg_g + avg_b) / 3

        # Determine background color based on average brightness
        print("xxxx avg_foreground", avg_foreground)
        if avg_foreground > 230:
            background_color = [0, 0, 0]  # Black background
        else:
            background_color = [255, 255, 255]  # White background

        # Create new RGB image with the selected background color
        new_img_np = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        new_img_np[:, :] = background_color

        # Blend the RGBA image with the background
        alpha = a / 255.0
        for c in range(3):
            new_img_np[:, :, c] = alpha * img_np[:, :, c] + (1 - alpha) * background_color[c]

        # Return the new image
        new_img = Image.fromarray(new_img_np)
        return new_img

    @app.post("/sdapi/v2/rembg-outfit")
    async def rembg_remove_v2(
        input_image: str = Body("", title='rembg input image'),
        model: str = Body("u2net", title='rembg model'), 
        return_mask: bool = Body(False, title='return mask'), 
        alpha_matting: bool = Body(False, title='alpha matting'), 
        alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'), 
        alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'), 
        alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        logger.info("===== API /sdapi/v2/rembg start =====")
        start_time = time.time()
        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)
        input_image = input_image.convert("RGB")

        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        left, upper, right, lower = image.getbbox()
        logger.info("===== API /sdapi/v2/rembg end in {:.3f} seconds =====".format(time.time() - start_time))
        return  {
                    "image": api.encode_pil_to_base64(rgba_to_rgb(image)).decode("utf-8"),
                    "box": {
                        "x": left, "y" : upper, "width" : right - left, "height" : lower - upper,
                    }
                }
    
    def crop_and_resize_image(image, width=None, height=None):
        def get_bbox(img, alpha_threshold=10):
            width, height = img.size
            pixels = img.load()
            left, top, right, bottom = width, height, 0, 0

            for y in range(height):
                for x in range(width):
                    if pixels[x, y][3] > alpha_threshold:
                        left, right = min(left, x), max(right, x)
                        top, bottom = min(top, y), max(bottom, y)
            return (left, top, right, bottom) if left <= right and top <= bottom else None

        logger.info(f"Cropping and resizing image: {width} {height}")
        image = image.convert("RGBA")

        left, upper, right, lower = get_bbox(image, 10)
        cropped_image = image.crop((left, upper, right, lower))

        if width and height:
            cropped_image.thumbnail((width, height), Image.ANTIALIAS)
            output_image = Image.new("RGBA", (width, height))
            output_image.paste(cropped_image, ((width - cropped_image.width) // 2, (height - cropped_image.height) // 2))
        else:
            size = max(cropped_image.width, cropped_image.height)
            output_image = Image.new("RGBA", (size, size))
            output_image.paste(cropped_image, ((size - cropped_image.width) // 2, (size - cropped_image.height) // 2))

        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return output_base64


    @app.post("/sdapi/v2/auto-crop")
    async def auto_crop(data: ImageRequest):
        logger.info("===== API /sdapi/v2/auto-crop start =====")
        start_time = time.time()
        try:
            image_base64 = data.image
            if image_base64.startswith("data:image/"):
                image_base64 = image_base64.split(";")[1].split(",")[1]
            image = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image))

            width = data.width
            height = data.height

            output_base64 = crop_and_resize_image(image, width, height)
            logger.info("===== API /sdapi/v2/auto-crop end in {:.3f} seconds =====".format(time.time() - start_time))
            return {"image": output_base64}
        except Exception as e:
            logger.error("===== API /sdapi/v2/auto-crop end in {:.3f} seconds =====".format(time.time() - start_time))
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/sdapi/v2/rembg-crop")
    async def rembg_remove(
        input_image: str = Body("", title='rembg input image'),
        model: str = Body("u2net", title='rembg model'), 
        return_mask: bool = Body(False, title='return mask'), 
        alpha_matting: bool = Body(False, title='alpha matting'), 
        alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'), 
        alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'), 
        alpha_matting_erode_size: int = Body(10, title='alpha matting erode size'),
        width: int = Body(0, title='width of the output image'),
        height: int = Body(0, title='height of the output image'),
    ):
        logger.info("===== API /sdapi/v2/rembg-crop start =====")
        start_time = time.time()

        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)

        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )
        
        image = crop_and_resize_image(image, width, height)
        logger.info("===== API /sdapi/v2/rembg-crop end in {:.3f} seconds =====".format(time.time() - start_time))
        return {"image": image}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
