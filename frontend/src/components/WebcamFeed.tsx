import { useRef, useEffect, useState, useCallback } from 'react';
import { getWsUrl } from '../config';

interface WebcamFeedProps {
  selectedStyle: string;
  intensity: number;
  onSnapshot: (imageData: string) => void;
}

export default function WebcamFeed({ selectedStyle, intensity, onSnapshot }: WebcamFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<string>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const styleRef = useRef(selectedStyle);
  const intensityRef = useRef(intensity);
  const isProcessingRef = useRef(false);
  const styledFrameRef = useRef<HTMLImageElement | null>(null);  // Cache styled frame
  const isLoopingRef = useRef(false);
  const isMountedRef = useRef(true); // Track component mount status

  // Keep refs in sync with props
  useEffect(() => {
    styleRef.current = selectedStyle;
    // Clear cached styled frame when style changes
    styledFrameRef.current = null;
  }, [selectedStyle]);

  useEffect(() => {
    intensityRef.current = intensity;
  }, [intensity]);

  // Start webcam on mount and handle cleanup
  useEffect(() => {
    isMountedRef.current = true;
    startWebcam();
    return () => {
      isMountedRef.current = false;
      stopWebcam();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Connect WebSocket when streaming starts
  useEffect(() => {
    if (isStreaming) {
      connectWebSocket();
    }
  }, [isStreaming]);

  // Start drawing loop as soon as we have a stream
  useEffect(() => {
    if (isStreaming && !isLoopingRef.current) {
      isLoopingRef.current = true;
      processFrames();
    }
  }, [isStreaming]);

  const startWebcam = async () => {
    try {
      // Check for secure context (required for camera access on non-localhost)
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          "Camera access is blocked. Browsers require HTTPS for camera access on mobile/network devices. Please test on 'localhost' or deploy to Vercel (HTTPS)."
        );
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        // Use 'ideal' constraints to prevent failure on devices that don't match exact resolution
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: "user" 
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);
        setError(null);
      }
    } catch (err: any) {
      console.error('Webcam error:', err);
      // Show the specific error message if it's our custom one, otherwise generic
      setError(err.message || 'Failed to access webcam. Please ensure you are using HTTPS or Localhost.');
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  };

  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      setWsStatus('connecting');
      const wsUrl = getWsUrl();
      
      if (!wsUrl) {
        throw new Error("Backend URL not configured. Please set VITE_API_URL or edit config.ts");
      }
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsStatus('connected');
      };
      
      ws.onmessage = (event) => {
        isProcessingRef.current = false;
        console.log('Received message from backend, data length:', event.data?.length || 0);
        
        // Receive styled frame from backend
        if (canvasRef.current && styleRef.current !== 'none') {
          const img = new Image();
          img.onload = () => {
            // Cache the styled frame so we can redraw it
            styledFrameRef.current = img;
          };
          img.onerror = (e) => {
            console.error('Failed to load styled image:', e);
          };
          // Check if event.data is valid
          if (event.data && typeof event.data === 'string') {
            img.src = 'data:image/jpeg;base64,' + event.data;
          } else {
            console.error('Invalid data received from backend:', typeof event.data);
          }
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsStatus('error');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setWsStatus('disconnected');
        isProcessingRef.current = false;
        
        // Attempt to reconnect after a delay only if still mounted
        setTimeout(() => {
          if (isMountedRef.current && isStreaming) {
            connectWebSocket();
          }
        }, 2000);
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setWsStatus('error');
    }
  }, [isStreaming]);

  const processFrames = useCallback(() => {
    const processFrame = () => {
      // Stop loop if component unmounted
      if (!isMountedRef.current) return;

      if (!videoRef.current || !canvasRef.current) {
        requestAnimationFrame(processFrame);
        return;
      }
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        requestAnimationFrame(processFrame);
        return;
      }

      // Wait for video to have data before drawing
      if (videoRef.current.readyState < 2) {
        requestAnimationFrame(processFrame);
        return;
      }

      // Set canvas dimensions
      if (canvas.width !== (videoRef.current.videoWidth || 640)) {
        console.log('Setting canvas size:', videoRef.current.videoWidth, videoRef.current.videoHeight);
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 480;
      }
      
      const currentStyle = styleRef.current;
      
      // Draw either the cached styled frame or raw video
      if (currentStyle !== 'none' && styledFrameRef.current) {
        // We have a styled frame cached, draw it
        ctx.drawImage(styledFrameRef.current, 0, 0, canvas.width, canvas.height);
      } else {
        // Draw raw video (mirrored) as fallback
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
      }
      
      // Calculate FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
      
      // Send frame to backend if style is selected, WebSocket is open, and not waiting for response
      if (
        currentStyle !== 'none' && 
        wsRef.current?.readyState === WebSocket.OPEN &&
        !isProcessingRef.current
      ) {
        console.log('Sending frame to backend for style:', currentStyle);
        try {
          // Create a temp canvas for sending the mirrored frame
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = canvas.width;
          tempCanvas.height = canvas.height;
          const tempCtx = tempCanvas.getContext('2d');
          if (tempCtx) {
            tempCtx.save();
            tempCtx.scale(-1, 1);
            tempCtx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
            tempCtx.restore();
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.6);
            const payload = JSON.stringify({
              image: imageData.split(',')[1],
              style: currentStyle,
              intensity: intensityRef.current
            });
            wsRef.current.send(payload);
            console.log('Sent frame to backend, style:', currentStyle);
            isProcessingRef.current = true;
          }
        } catch (e) {
          console.error('Error sending frame:', e);
        }
      }
      
      requestAnimationFrame(processFrame);
    };
    
    processFrame();
  }, []);

  const handleSnapshot = () => {
    if (canvasRef.current) {
      const imageData = canvasRef.current.toDataURL('image/png');
      onSnapshot(imageData);
    }
  };

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {error && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="bg-red-900/80 px-6 py-4 rounded-lg text-red-200">
            {error}
          </div>
        </div>
      )}
      
      <video
        ref={videoRef}
        className="hidden"
        autoPlay
        playsInline
        muted
      />
      
      <canvas
        ref={canvasRef}
        className="w-full h-full object-contain"
      />
      
      {/* Status indicators */}
      <div className="absolute top-6 left-6 flex flex-col gap-3 z-10 pointer-events-none">
        <div className="bg-black/40 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-[10px] uppercase tracking-widest text-white flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-white opacity-50 animate-pulse" />
          Style: <span className="text-[#e5b567] font-bold">{selectedStyle === 'none' ? 'Authentic' : selectedStyle}</span>
        </div>
        <div className="bg-black/40 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-[10px] uppercase tracking-widest text-white flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full ${
            wsStatus === 'connected' ? 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.5)]' : 
            wsStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' : 'bg-red-400'
          }`} />
          Network: <span className="opacity-80">{wsStatus}</span>
        </div>
      </div>
      
      {/* FPS Counter */}
      <div className="absolute top-6 right-6 z-10 pointer-events-none">
        <div className="bg-black/40 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full font-mono text-[10px] text-white tracking-widest uppercase">
          <span className="opacity-50">Studio Speed:</span> <span className="text-green-400 font-bold">{fps}</span> <span className="opacity-50 text-[8px]">FPS</span>
        </div>
      </div>
      
      {/* Snapshot Button - Aesthetic Shutter Design */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 z-10 flex flex-col items-center gap-2">
        <button
          onClick={handleSnapshot}
          className="group relative flex items-center justify-center"
        >
          {/* Outer Ring */}
          <div className="absolute inset-0 -m-3 border-2 border-[#e5b567] rounded-full opacity-0 group-hover:opacity-100 group-active:scale-95 transition-all duration-300 scale-75 group-hover:scale-100" />
          
          {/* Main Button */}
          <div className="relative w-16 h-16 bg-white rounded-full shadow-2xl flex items-center justify-center overflow-hidden transition-transform duration-300 group-hover:scale-110 group-active:scale-90 border-4 border-[#e5b567]/20">
            {/* Shutter Effect Background */}
            <div className="absolute inset-0 bg-[#e5b567] opacity-0 group-hover:opacity-10 transition-opacity" />
            
            {/* Camera Icon */}
            <svg 
              className="w-8 h-8 text-[#1a1a1a] transition-transform duration-500 group-hover:rotate-12" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1.5} 
                d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" 
              />
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1.5} 
                d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" 
              />
            </svg>
          </div>
        </button>
        <span className="text-white/40 text-[9px] uppercase tracking-[0.2em] font-medium pointer-events-none group-hover:text-white transition-colors">Capture Artwork</span>
      </div>
    </div>
  );
}
