import { useState } from 'react';

interface Style {
  id: string;
  name: string;
  preview: string;
}

const styles: Style[] = [
  { id: 'none', name: 'Original', preview: '/style_previews/original.png' },
  { id: 'vangogh', name: 'Van Gogh', preview: '/style_previews/vangogh.png' },
  { id: 'picasso', name: 'Picasso', preview: '/style_previews/picasso.png' },
  { id: 'monet', name: 'Monet', preview: '/style_previews/monet.png' },
  { id: 'candy', name: 'Candy', preview: '/style_previews/candy.png' },
  { id: 'rain_princess', name: 'Rain Princess', preview: '/style_previews/rain_princess.png' },
  { id: 'mosaic', name: 'Mosaic', preview: '/style_previews/mosaic.png' },
  { id: 'udnie', name: 'Udnie', preview: '/style_previews/udnie.png' },
  { id: 'scream', name: 'The Scream', preview: '/style_previews/scream.png' },
  { id: 'wave', name: 'Great Wave', preview: '/style_previews/wave.png' },
];

export default function ImageUpload() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [styledImage, setStyledImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState('none');
  const [intensity, setIntensity] = useState(100);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageData = e.target?.result as string;
        setUploadedImage(imageData);
        setStyledImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const applyStyle = async () => {
    if (!uploadedImage) return;
    
    if (selectedStyle === 'none') {
      setStyledImage(uploadedImage);
      return;
    }

    setIsProcessing(true);
    try {
      const apiUrl = getApiUrl();
      if (!apiUrl) {
         alert("Backend URL not configured. Please set VITE_API_URL or edit config.ts");
         setIsProcessing(false);
         return;
      }
      const response = await fetch(`${apiUrl}/process-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: uploadedImage,
          style: selectedStyle,
          intensity: intensity,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setStyledImage(data.image);
      }
    } catch (error) {
      console.error('Error processing image:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadImage = () => {
    if (!styledImage) return;
    const link = document.createElement('a');
    link.href = styledImage;
    link.download = `styled-image-${selectedStyle}-${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="image-upload-section">
      <h2 className="section-title">Upload & Style Images</h2>
      
      <div className="upload-main-container">
        {/* Left: Upload and Output */}
        <div className="upload-left-panel">
          <div className="upload-container">
            {/* Upload Area */}
            <div className="upload-area">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="image-upload"
              />
              <label htmlFor="image-upload" className="upload-label">
                {uploadedImage ? (
                  <img src={uploadedImage} alt="Uploaded" className="uploaded-preview" />
                ) : (
                  <div className="upload-placeholder">
                    <svg className="w-16 h-16 text-gray-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-gray-600 font-medium">Click to upload an image</p>
                    <p className="text-gray-400 text-sm mt-1">PNG, JPG up to 10MB</p>
                  </div>
                )}
              </label>
            </div>

            {/* Styled Output Area */}
            <div className="styled-output-area">
              {styledImage ? (
                <img src={styledImage} alt="Styled" className="styled-preview" />
              ) : uploadedImage ? (
                <div className="output-placeholder">
                  <svg className="w-12 h-12 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-gray-500 text-sm">Upload an image to begin</p>
                </div>
              ) : (
                <div className="output-placeholder">
                  <p className="text-gray-400 text-sm">Upload an image to style it</p>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          {uploadedImage && (
            <div className="upload-actions">
              <button 
                onClick={applyStyle}
                disabled={isProcessing}
                className="btn-primary"
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </>
                ) : (
                  'Apply Style'
                )}
              </button>
              {styledImage && (
                <button onClick={downloadImage} className="btn-secondary">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
          )}
        </div>

        {/* Right: Style Selector */}
        <div className="upload-right-panel">
          <h3 className="upload-panel-title">Select Style</h3>
          <div className="style-grid">
            {styles.map((style) => (
              <div key={style.id} className="flex flex-col items-center gap-1">
                <button
                  onClick={() => setSelectedStyle(style.id)}
                  className={`style-card w-full ${selectedStyle === style.id ? 'selected' : ''}`}
                >
                  <img 
                    src={style.preview} 
                    alt={style.name}
                    className="w-full h-full object-cover"
                  />
                </button>
                <span className="style-card-label">{style.name}</span>
              </div>
            ))}
          </div>

          {/* Intensity Slider */}
          <div className="intensity-control">
            <label className="intensity-label">
              <span>Intensity</span>
              <span className="intensity-value">{intensity}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={intensity}
              onChange={(e) => setIntensity(parseInt(e.target.value))}
              className="intensity-slider"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
