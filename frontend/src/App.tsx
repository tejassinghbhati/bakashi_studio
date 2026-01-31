import { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import StyleSelector from './components/StyleSelector';
import ImageUpload from './components/ImageUpload';
import './index.css';

function App() {
  const [selectedStyle, setSelectedStyle] = useState('none');
  const [intensity, setIntensity] = useState(100);
  const [snapshots, setSnapshots] = useState<string[]>([]);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);

  const handleSnapshot = (imageData: string) => {
    setSnapshots(prev => [imageData, ...prev]);
  };

  const downloadSnapshot = (imageData: string, index: number) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `bakashi-studio-${Date.now()}-${index}.png`;
    link.click();
  };

  const deleteSnapshot = (index: number) => {
    setSnapshots(prev => prev.filter((_, i) => i !== index));
    // Close preview if deleting current image
    if (previewIndex === index) {
      setPreviewIndex(null);
    } else if (previewIndex !== null && index < previewIndex) {
      setPreviewIndex(previewIndex - 1);
    }
  };

  const openPreview = (index: number) => {
    setPreviewIndex(index);
  };

  const closePreview = () => {
    setPreviewIndex(null);
  };

  const goToPrev = () => {
    if (previewIndex !== null && previewIndex > 0) {
      setPreviewIndex(previewIndex - 1);
    }
  };

  const goToNext = () => {
    if (previewIndex !== null && previewIndex < snapshots.length - 1) {
      setPreviewIndex(previewIndex + 1);
    }
  };

  // Keyboard navigation for preview
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (previewIndex === null) return;
    if (e.key === 'ArrowLeft') goToPrev();
    if (e.key === 'ArrowRight') goToNext();
    if (e.key === 'Escape') closePreview();
  };

  return (
    <div className="min-h-screen bg-[#12121a]">
      {/* Header */}
      <header className="studio-header">
        <h1 className="studio-title">Bakashi Studio</h1>
      </header>

      {/* Main Content - Two Column Layout */}
      <main className="flex flex-col lg:flex-row">
        {/* Left Column - Preview Panel */}
        <div className="preview-panel flex-1 lg:flex-[2] flex items-start justify-center">
          <div className="preview-canvas-wrapper w-full max-w-2xl">
            <WebcamFeed 
              selectedStyle={selectedStyle}
              intensity={intensity}
              onSnapshot={handleSnapshot}
            />
          </div>
        </div>

        {/* Right Column - Style Selector + Gallery Panel */}
        <div className="style-panel flex-1 lg:flex-[1]">
          {/* Style Selector Section */}
          <StyleSelector 
            selectedStyle={selectedStyle}
            onStyleChange={setSelectedStyle}
          />

          {/* Intensity Slider - Visual Fix & Build Fix */}
          <div className="intensity-control mt-4 mb-2 px-1">
            <label className="intensity-label mb-2">
              <span className="text-sm font-medium">Artistic Intensity</span>
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

          {/* Gallery Section - Below Style Selector */}
          <div className="gallery-section">
            <div className="gallery-header">
              <h3 className="gallery-title">
                Your Museum
                {snapshots.length > 0 && (
                  <span className="gallery-count">{snapshots.length}</span>
                )}
              </h3>
            </div>

            {snapshots.length === 0 ? (
              <div className="gallery-empty">
                <svg className="w-12 h-12 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <p className="text-gray-500 text-sm">No artworks yet</p>
                <p className="text-gray-400 text-xs">Take a photo to save it here</p>
              </div>
            ) : (
              <div className="gallery-grid">
                {snapshots.map((snapshot, index) => (
                  <div 
                    key={index} 
                    className="gallery-item"
                    onClick={() => openPreview(index)}
                  >
                    <img 
                      src={snapshot} 
                      alt={`Snapshot ${index + 1}`}
                      className="gallery-image"
                    />
                    <div className="gallery-item-overlay">
                      <button
                        onClick={(e) => { e.stopPropagation(); downloadSnapshot(snapshot, index); }}
                        className="gallery-action-btn"
                        title="Download"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteSnapshot(index); }}
                        className="gallery-action-btn gallery-action-btn-danger"
                        title="Delete"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Lightbox Preview Modal */}
      {previewIndex !== null && snapshots[previewIndex] && (
        <div 
          className="lightbox-overlay"
          onClick={closePreview}
          onKeyDown={handleKeyDown}
          tabIndex={0}
          ref={(el) => el?.focus()}
        >
          {/* Close Button */}
          <button className="lightbox-close" onClick={closePreview}>
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Previous Button */}
          {previewIndex > 0 && (
            <button 
              className="lightbox-nav lightbox-prev"
              onClick={(e) => { e.stopPropagation(); goToPrev(); }}
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}

          {/* Image */}
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <img 
              src={snapshots[previewIndex]} 
              alt={`Preview ${previewIndex + 1}`}
              className="lightbox-image"
            />
            
            {/* Bottom Actions */}
            <div className="lightbox-actions">
              <span className="lightbox-counter">
                {previewIndex + 1} / {snapshots.length}
              </span>
              <div className="lightbox-buttons">
                <button
                  onClick={() => downloadSnapshot(snapshots[previewIndex], previewIndex)}
                  className="lightbox-btn"
                  title="Download"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
                <button
                  onClick={() => deleteSnapshot(previewIndex)}
                  className="lightbox-btn lightbox-btn-danger"
                  title="Delete"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Delete
                </button>
              </div>
            </div>
          </div>

          {/* Next Button */}
          {previewIndex < snapshots.length - 1 && (
            <button 
              className="lightbox-nav lightbox-next"
              onClick={(e) => { e.stopPropagation(); goToNext(); }}
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          )}
        </div>
      )}

      {/* Image Upload Section - Below main content */}
      <ImageUpload />

      {/* Footer */}
      <footer className="studio-footer">
        <p className="footer-text">
          Built by Tejas SIngh Bhati <span className="footer-divider">~</span> Thankyou for visiting
        </p>
      </footer>
    </div>
  );
}

export default App;
