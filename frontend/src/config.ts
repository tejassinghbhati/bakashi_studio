const isProduction = import.meta.env.PROD;

// ============================================
// 🔧 QUICK FIX CONFIGURATION
// If you are having trouble with Vercel Environment Variables,
// you can simply paste your Render Backend URL correctly below.
// 
// Example: "https://bakashi-backend.onrender.com"
// (Do NOT include the trailing slash /)
// ============================================
const HARDCODED_BACKEND_URL = ""; 

/**
 * Determines the API URL based on environment or configuration.
 */
export const getApiUrl = (): string => {
  // 1. Priority: Hardcoded URL (easiest for debugging)
  if (HARDCODED_BACKEND_URL) {
    // Remove trailing slash if present
    return HARDCODED_BACKEND_URL.replace(/\/$/, "");
  }

  // 2. Priority: Environment Variable (Best practice for Vercel)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL.replace(/\/$/, "");
  }
  
  // 3. Fallback: Assumption for Local Development
  // If we are NOT in production, we assume the backend is running on port 8000
  // on the same network address as the frontend.
  if (!isProduction) {
     const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
     const host = window.location.hostname;
     return `${protocol}//${host}:8000`;
  }
  
  // 4. Production Fallback (Safety net)
  // If no config is present, we try to connect to the same host (assumes backend is served from same origin or proxy)
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const host = window.location.hostname;
  return `${protocol}//${host}:8000`; 
}

/**
 * Determines the WebSocket URL based on the API URL.
 */
export const getWsUrl = (): string => {
  // If VITE_WS_URL is explicitly set, use it.
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }

  const apiUrl = getApiUrl();
  
  if (!apiUrl) return "";
  
  // Convert http/https to ws/wss
  const wsProtocol = apiUrl.startsWith('https') ? 'wss:' : 'ws:';
  // Remove protocol from API URL to get the domain keys
  const urlWithoutProtocol = apiUrl.replace(/^https?:\/\//, '');
  
  return `${wsProtocol}//${urlWithoutProtocol}/ws/style`;
}
