from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
import cv2
import numpy as np
from typing import Optional

app = FastAPI(title="PixelForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")


# FIX: Made this function async and used `await file.read()`
async def load_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # If image has no alpha channel, convert BGR to BGRA
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # If grayscale image, convert to BGRA
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    return img


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
    img = await load_image(file)

    b, g, r, a = cv2.split(img)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    result = cv2.merge((gray, gray, gray, a))

    return {"result": image_to_base64(result), "operation": "grayscale"}


@app.post("/api/process/blur")
async def blur(file: UploadFile = File(...), radius: float = Form(2.0)):
    img = await load_image(file)

    radius = max(0, radius)

    # Kernel size must be odd and greater than 0
    kernel_size = int(radius * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)

    result = cv2.GaussianBlur(img, (kernel_size, kernel_size), radius)

    return {"result": image_to_base64(result), "operation": "blur", "radius": radius}


@app.post("/api/process/sharpen")
async def sharpen(file: UploadFile = File(...), factor: float = Form(2.0)):
    img = await load_image(file)

    # Sharpening kernel controlled by factor
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + factor, -1],
        [0, -1, 0]
    ])

    result = cv2.filter2D(img, -1, kernel)

    return {"result": image_to_base64(result), "operation": "sharpen", "factor": factor}


@app.post("/api/process/brightness")
async def brightness(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    result_bgr = cv2.convertScaleAbs(bgr, alpha=factor, beta=0)
    result = cv2.merge((result_bgr[:, :, 0], result_bgr[:, :, 1], result_bgr[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "brightness", "factor": factor}


@app.post("/api/process/contrast")
async def contrast(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    result_bgr = cv2.convertScaleAbs(bgr, alpha=factor, beta=0)
    result = cv2.merge((result_bgr[:, :, 0], result_bgr[:, :, 1], result_bgr[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "contrast", "factor": factor}


@app.post("/api/process/saturation")
async def saturation(file: UploadFile = File(...), factor: float = Form(1.0)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result = cv2.merge((result_bgr[:, :, 0], result_bgr[:, :, 1], result_bgr[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "saturation", "factor": factor}


@app.post("/api/process/invert")
async def invert(file: UploadFile = File(...)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    inverted_bgr = cv2.bitwise_not(bgr)
    result = cv2.merge((inverted_bgr[:, :, 0], inverted_bgr[:, :, 1], inverted_bgr[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "invert"}


@app.post("/api/process/sepia")
async def sepia(file: UploadFile = File(...)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    sepia_kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sepia_rgb = cv2.transform(rgb, sepia_kernel)
    sepia_rgb = np.clip(sepia_rgb, 0, 255).astype(np.uint8)

    sepia_bgr = cv2.cvtColor(sepia_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.merge((sepia_bgr[:, :, 0], sepia_bgr[:, :, 1], sepia_bgr[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "sepia"}


@app.post("/api/process/rotate")
async def rotate(file: UploadFile = File(...), angle: float = Form(90.0)):
    img = await load_image(file)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Negative angle to match your previous Pillow behavior
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    result = cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return {"result": image_to_base64(result), "operation": "rotate", "angle": angle}


@app.post("/api/process/flip")
async def flip(file: UploadFile = File(...), direction: str = Form("horizontal")):
    img = await load_image(file)

    if direction == "horizontal":
        result = cv2.flip(img, 1)
    else:
        result = cv2.flip(img, 0)

    return {"result": image_to_base64(result), "operation": "flip", "direction": direction}


@app.post("/api/process/edge-detect")
async def edge_detect(file: UploadFile = File(...)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    result = cv2.merge((edges, edges, edges, alpha))

    return {"result": image_to_base64(result), "operation": "edge-detect"}


@app.post("/api/process/emboss")
async def emboss(file: UploadFile = File(...)):
    img = await load_image(file)

    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    embossed = cv2.filter2D(bgr, -1, emboss_kernel)
    embossed = cv2.convertScaleAbs(embossed, alpha=1, beta=128)

    result = cv2.merge((embossed[:, :, 0], embossed[:, :, 1], embossed[:, :, 2], alpha))

    return {"result": image_to_base64(result), "operation": "emboss"}


@app.post("/api/process/pixelate")
async def pixelate(file: UploadFile = File(...), block_size: int = Form(10)):
    img = await load_image(file)

    h, w = img.shape[:2]
    block_size = max(1, block_size)

    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)

    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return {"result": image_to_base64(result), "operation": "pixelate", "block_size": block_size}


@app.post("/api/process/resize")
async def resize(file: UploadFile = File(...), width: int = Form(800), height: int = Form(600)):
    img = await load_image(file)

    result = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return {"result": image_to_base64(result), "operation": "resize", "width": width, "height": height}


@app.post("/api/process/crop")
async def crop(
    file: UploadFile = File(...),
    left: int = Form(0),
    top: int = Form(0),
    right: int = Form(100),
    bottom: int = Form(100)
):
    img = await load_image(file)

    h, w = img.shape[:2]

    left = max(0, min(left, w))
    top = max(0, min(top, h))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    result = img[top:bottom, left:right]

    return {"result": image_to_base64(result), "operation": "crop"}