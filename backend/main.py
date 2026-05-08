from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
import numpy as np
from typing import Optional

app = FastAPI(title="PixelForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_image(file: UploadFile) -> Image.Image:
    contents = file.file.read()
    return Image.open(io.BytesIO(contents)).convert("RGBA")

@app.get("/api/")
def root():
    return {"status": "ok", "service": "PixelForge API"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.get("/api/operations")
def operations():
    return {
        "operations": [
            {"id": "grayscale", "name": "Grayscale", "category": "Color", "params": []},
            {"id": "invert", "name": "Invert", "category": "Color", "params": []},
            {"id": "sepia", "name": "Sepia", "category": "Color", "params": []},
            {"id": "blur", "name": "Blur", "category": "Filters", "params": [{"name": "radius", "type": "float", "default": 2.0, "min": 0, "max": 20}]},
            {"id": "sharpen", "name": "Sharpen", "category": "Filters", "params": [{"name": "factor", "type": "float", "default": 2.0, "min": 1, "max": 10}]},
            {"id": "edge-detect", "name": "Edge Detect", "category": "Filters", "params": []},
            {"id": "emboss", "name": "Emboss", "category": "Filters", "params": []},
            {"id": "pixelate", "name": "Pixelate", "category": "Filters", "params": [{"name": "block_size", "type": "int", "default": 10, "min": 2, "max": 50}]},
            {"id": "brightness", "name": "Brightness", "category": "Adjustments", "params": [{"name": "factor", "type": "float", "default": 1.0, "min": 0, "max": 3}]},
            {"id": "contrast", "name": "Contrast", "category": "Adjustments", "params": [{"name": "factor", "type": "float", "default": 1.0, "min": 0, "max": 3}]},
            {"id": "saturation", "name": "Saturation", "category": "Adjustments", "params": [{"name": "factor", "type": "float", "default": 1.0, "min": 0, "max": 3}]},
            {"id": "rotate", "name": "Rotate", "category": "Transform", "params": [{"name": "angle", "type": "float", "default": 90.0, "min": -360, "max": 360}]},
            {"id": "flip", "name": "Flip", "category": "Transform", "params": [{"name": "direction", "type": "select", "options": ["horizontal", "vertical"], "default": "horizontal"}]},
            {"id": "resize", "name": "Resize", "category": "Transform", "params": [{"name": "width", "type": "int", "default": 800}, {"name": "height", "type": "int", "default": 600}]},
            {"id": "crop", "name": "Crop", "category": "Transform", "params": [{"name": "left", "type": "int", "default": 0}, {"name": "top", "type": "int", "default": 0}, {"name": "right", "type": "int", "default": 100}, {"name": "bottom", "type": "int", "default": 100}]},
        ]
    }

@app.post("/api/process/grayscale")
async def grayscale(file: UploadFile = File(...)):
    img = load_image(file)
    r, g, b, a = img.split()
    gray = ImageOps.grayscale(img.convert("RGB"))
    gray_rgba = Image.merge("RGBA", (gray, gray, gray, a))
    return {"result": image_to_base64(gray_rgba), "operation": "grayscale"}

@app.post("/api/process/blur")
async def blur(file: UploadFile = File(...), radius: float = Form(2.0)):
    img = load_image(file)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return {"result": image_to_base64(blurred), "operation": "blur", "radius": radius}

@app.post("/api/process/sharpen")
async def sharpen(file: UploadFile = File(...), factor: float = Form(2.0)):
    img = load_image(file)
    enhancer = ImageEnhance.Sharpness(img)
    sharpened = enhancer.enhance(factor)
    return {"result": image_to_base64(sharpened), "operation": "sharpen", "factor": factor}

@app.post("/api/process/brightness")
async def brightness(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = load_image(file)
    enhancer = ImageEnhance.Brightness(img)
    result = enhancer.enhance(factor)
    return {"result": image_to_base64(result), "operation": "brightness", "factor": factor}

@app.post("/api/process/contrast")
async def contrast(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = load_image(file)
    enhancer = ImageEnhance.Contrast(img)
    result = enhancer.enhance(factor)
    return {"result": image_to_base64(result), "operation": "contrast", "factor": factor}

@app.post("/api/process/saturation")
async def saturation(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = load_image(file)
    rgb = img.convert("RGB")
    enhancer = ImageEnhance.Color(rgb)
    result = enhancer.enhance(factor).convert("RGBA")
    r, g, b, _ = result.split()
    _, _, _, a = img.split()
    result = Image.merge("RGBA", (r, g, b, a))
    return {"result": image_to_base64(result), "operation": "saturation", "factor": factor}

@app.post("/api/process/invert")
async def invert(file: UploadFile = File(...)):
    img = load_image(file)
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    inverted = ImageOps.invert(rgb)
    ir, ig, ib = inverted.split()
    result = Image.merge("RGBA", (ir, ig, ib, a))
    return {"result": image_to_base64(result), "operation": "invert"}

@app.post("/api/process/sepia")
async def sepia(file: UploadFile = File(...)):
    img = load_image(file)
    r, g, b, a = img.split()
    gray = ImageOps.grayscale(Image.merge("RGB", (r, g, b)))
    sepia_r = gray.point(lambda i: min(255, i * 1.07))
    sepia_g = gray.point(lambda i: min(255, i * 0.74))
    sepia_b = gray.point(lambda i: min(255, i * 0.43))
    result = Image.merge("RGBA", (sepia_r, sepia_g, sepia_b, a))
    return {"result": image_to_base64(result), "operation": "sepia"}

@app.post("/api/process/rotate")
async def rotate(file: UploadFile = File(...), angle: float = Form(90.0)):
    img = load_image(file)
    result = img.rotate(-angle, expand=True)
    return {"result": image_to_base64(result), "operation": "rotate", "angle": angle}

@app.post("/api/process/flip")
async def flip(file: UploadFile = File(...), direction: str = Form("horizontal")):
    img = load_image(file)
    if direction == "horizontal":
        result = ImageOps.mirror(img)
    else:
        result = ImageOps.flip(img)
    return {"result": image_to_base64(result), "operation": "flip", "direction": direction}

@app.post("/api/process/edge-detect")
async def edge_detect(file: UploadFile = File(...)):
    img = load_image(file)
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    edges = rgb.filter(ImageFilter.FIND_EDGES)
    er, eg, eb = edges.split()
    result = Image.merge("RGBA", (er, eg, eb, a))
    return {"result": image_to_base64(result), "operation": "edge-detect"}

@app.post("/api/process/emboss")
async def emboss(file: UploadFile = File(...)):
    img = load_image(file)
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    embossed = rgb.filter(ImageFilter.EMBOSS)
    er, eg, eb = embossed.split()
    result = Image.merge("RGBA", (er, eg, eb, a))
    return {"result": image_to_base64(result), "operation": "emboss"}

@app.post("/api/process/pixelate")
async def pixelate(file: UploadFile = File(...), block_size: int = Form(10)):
    img = load_image(file)
    w, h = img.size
    block_size = max(1, block_size)
    small = img.resize((max(1, w // block_size), max(1, h // block_size)), Image.NEAREST)
    result = small.resize((w, h), Image.NEAREST)
    return {"result": image_to_base64(result), "operation": "pixelate", "block_size": block_size}

@app.post("/api/process/resize")
async def resize(file: UploadFile = File(...), width: int = Form(800), height: int = Form(600)):
    img = load_image(file)
    result = img.resize((width, height), Image.LANCZOS)
    return {"result": image_to_base64(result), "operation": "resize", "width": width, "height": height}

@app.post("/api/process/crop")
async def crop(file: UploadFile = File(...), left: int = Form(0), top: int = Form(0), right: int = Form(100), bottom: int = Form(100)):
    img = load_image(file)
    w, h = img.size
    left = max(0, min(left, w))
    top = max(0, min(top, h))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    result = img.crop((left, top, right, bottom))
    return {"result": image_to_base64(result), "operation": "crop"}