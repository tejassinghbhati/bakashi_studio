# 🎨 Bakashi Studio — Neural Style Transfer

<p align="center">
  <img src="https://img.shields.io/badge/Maintained-yes-green.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Tech-React%20%2B%20FastAPI-blue.svg" />
</p>

<p align="center"><i>Transforming your reality into a masterpiece through Neural Style Transfer.</i></p>

---

## ✨ Overview

**Bakashi Studio** is a high-performance web application for artistic exploration.  
It uses **Deep Learning** and **Neural Style Transfer** to transform real-time webcam feeds and uploaded images into works of art inspired by masters like Van Gogh, Monet, and Picasso.

The interface follows a **classical museum aesthetic**, blending modern AI with timeless fine art.

---

## 🚀 Key Features

- 🎭 **Real-time Artistic Vision** — Apply 10+ styles to live webcam feed with low latency  
- 🖼️ **Image Laboratory** — Upload photos and reimagine them artistically  
- 🎚️ **Precision Controls** — Adjustable artistic intensity  
- 🏛️ **Personal Museum** — Capture, preview, and download creations  
- ⚜️ **Premium UI** — Ornate design with gold accents and Libre Caslon Text  
- ⚡ **High Performance** — FastAPI + PyTorch optimized pipeline  

---

## 🎨 Artistic Styles

| Style | Inspiration |
|------|------------|
| Starry Night | Vincent van Gogh |
| The Scream | Edvard Munch |
| Great Wave | Hokusai |
| Mosaic | Ancient Roman Art |
| Monet | Claude Monet |
| Udnie | Francis Picabia |

---

## 🛠️ Tech Stack

### Frontend
- React 18 (TypeScript)
- Tailwind CSS + Custom CSS
- WebSockets
- Vite

### Backend
- Python 3.10+
- FastAPI
- PyTorch & TorchVision
- OpenCV & PIL

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Optional: NVIDIA GPU (CUDA)

### Clone Repository
```bash
git clone https://github.com/tejassinghbhati/Bakashi-Studio.git
cd Bakashi-Studio
```

### 1. Run with One Click (Windows)
Double-click the `start_app.bat` file in the root directory.

### 2. Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 🌐 Deployment Configuration

To make the app work on the public internet (Vercel + Render), you must link them.

### 1. Backend (Render)
Deploy the `backend` folder to Render as a **Web Service**.
- **Build Command:** `sh render_build.sh`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 2. Frontend (Vercel)
Deploy the `frontend` folder to Vercel.
**Crucial:** You must add the following **Environment Variables** in your Vercel Project Settings so the frontend knows where the backend is.

| Variable Name | Value Example | Note |
|--------------|--------------|------|
| `VITE_API_URL` | `https://your-app-name.onrender.com` | No trailing slash |
| `VITE_WS_URL` | `wss://your-app-name.onrender.com/ws/style` | Must be `wss://` for secure SSL |

After adding these, **Redeploy** your Vercel project.
