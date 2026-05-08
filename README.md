# PixelForge — Digital Image Studio

A full-stack image processing app with a **FastAPI backend** and **Studio Precision** frontend.  
The frontend is a fully functional single-page app matching the design system from your UI specs.

---

## 🗂 Project Structure

```
pixelforge/
├── backend/
│   ├── main.py           # FastAPI app — all 17 processing endpoints
│   └── requirements.txt  # Python dependencies
└── frontend/
    └── index.html        # Full SPA — Landing, Editor, Export, Compare
```

---

## 🚀 Backend Setup (FastAPI)

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the API server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: **http://localhost:8000/docs**

---

## 🌐 Frontend Setup

```bash
cd frontend
python3 -m http.server 3000
# Open http://localhost:3000
```

Or open `frontend/index.html` directly in your browser.  
The frontend auto-detects the API at `http://localhost:8000`.

---

## ⚡ API Endpoints

| Method | Endpoint | Params |
|--------|----------|--------|
| GET | `/health` | — |
| GET | `/operations` | — |
| POST | `/process/grayscale` | — |
| POST | `/process/blur` | `radius` (float) |
| POST | `/process/sharpen` | `factor` (float) |
| POST | `/process/brightness` | `factor` (float) |
| POST | `/process/contrast` | `factor` (float) |
| POST | `/process/saturation` | `factor` (float) |
| POST | `/process/invert` | — |
| POST | `/process/sepia` | — |
| POST | `/process/rotate` | `angle` (float) |
| POST | `/process/flip` | `direction` (horizontal/vertical) |
| POST | `/process/edge-detect` | — |
| POST | `/process/emboss` | — |
| POST | `/process/pixelate` | `block_size` (int) |
| POST | `/process/resize` | `width`, `height` (int) |
| POST | `/process/crop` | `left`, `top`, `right`, `bottom` (int) |

---

## 🎨 Frontend Features

- **Landing screen** — drag & drop or click to import any image
- **Editor workspace** — 3-column layout: tool rail, canvas, controls panel
- **17 operations** across 4 categories: Color, Adjustments, Filters, Transform
- **Sliders + numeric inputs** synced in real-time for precise control
- **Operation history** — bottom tray shows applied steps, click to jump back
- **Undo** — step back one operation at a time
- **Before/After compare** — drag slider to reveal original vs processed
- **Export modal** — download as PNG, JPEG, or WEBP with quality control
- **API status indicator** — live connection check in nav bar
- **Zoom controls** — 10% to 400% canvas zoom

---

## 🔧 Design System

The UI implements the **Studio Precision** design system:
- **Fonts**: Inter (UI) + Space Grotesk (numeric inputs)
- **Color palette**: Warm-neutral surfaces with Deep Indigo accents
- **Layout**: 48px left tool rail + fluid canvas + 280px right control panel + 60px history tray
- **Typography scale**: 11px labels → 13px UI → 14px headings
