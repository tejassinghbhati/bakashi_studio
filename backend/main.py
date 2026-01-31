from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from style_transfer import StyleTransferModel
import uvicorn
import json
import asyncio

app = FastAPI(title="Neural Style Transfer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize style transfer model
style_model = StyleTransferModel()

@app.get("/")
async def root():
    return {
        "message": "Neural Style Transfer Backend",
        "status": "running",
        "device": str(style_model.device),
        "available_styles": [
            "vangogh", "picasso", "monet", "candy", 
            "mosaic", "udnie", "scream", "wave", "rain"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(style_model.device)}

@app.post("/process-image")
async def process_image(request: dict):
    """
    Process a single uploaded image with style transfer
    
    Request body:
    {
        "image": "base64_encoded_image",
        "style": "vangogh|picasso|monet|candy|mosaic|udnie|scream|wave|rain_princess|none",
        "intensity": 0-100
    }
    """
    try:
        # Extract parameters
        image_base64 = request.get('image', '')
        style_name = request.get('style', 'none')
        intensity = request.get('intensity', 100)
        
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Validate parameters
        if not image_base64:
            return {"error": "No image provided"}, 400
        
        # Map rain_princess to rain for backend compatibility
        if style_name == 'rain_princess':
            style_name = 'rain_princess'
        
        print(f"Processing image - Style: {style_name}, Intensity: {intensity}")
        
        # Process the image
        styled_image_base64 = style_model.process_frame(
            image_base64,
            style_name,
            intensity
        )
        
        if styled_image_base64:
            # Return as data URL for frontend
            return {
                "success": True,
                "image": f"data:image/jpeg;base64,{styled_image_base64}",
                "style": style_name,
                "intensity": intensity
            }
        else:
            return {"error": "Failed to process image"}, 500
            
    except Exception as e:
        print(f"Error in /process-image: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.websocket("/ws/style")
async def style_transfer_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket")
    frame_count = 0
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            frame_count += 1
            
            try:
                # Parse JSON data
                frame_data = json.loads(data)
                image_base64 = frame_data.get('image')
                style_name = frame_data.get('style', 'none')
                intensity = frame_data.get('intensity', 100)
                
                if frame_count == 1:
                    print(f"First frame received - Style: {style_name}, Intensity: {intensity}")
                    print(f"Image data length: {len(image_base64) if image_base64 else 0}")
                
                # Process frame with style transfer
                styled_image = style_model.process_frame(
                    image_base64, 
                    style_name, 
                    intensity
                )
                
                # Send processed frame back to client
                if styled_image:
                    if frame_count == 1:
                        print(f"Styled image generated, length: {len(styled_image)}")
                    await websocket.send_text(styled_image)
                else:
                    print(f"Frame {frame_count}: No styled image returned, sending original")
                    await websocket.send_text(image_base64)
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Processing error on frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except WebSocketDisconnect:
        print(f"Client disconnected after {frame_count} frames")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print("Neural Style Transfer Backend")
    print("=" * 50)
    print(f"Device: {style_model.device}")
    print("Starting server on http://0.0.0.0:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/style")
    print("=" * 50)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
