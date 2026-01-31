interface ControlsProps {
  intensity: number;
  onIntensityChange: (value: number) => void;
}

export default function Controls({ intensity, onIntensityChange }: ControlsProps) {
  return (
    <div className="w-full space-y-4">
      <h2 className="text-xl font-bold bg-gradient-to-r from-primary-400 to-purple-500 bg-clip-text text-transparent">
        Controls
      </h2>
      
      <div className="glass p-6 rounded-xl space-y-4">
        {/* Intensity Slider */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium text-gray-300">
              Style Intensity
            </label>
            <span className="text-sm font-mono text-primary-400 font-bold">
              {intensity}%
            </span>
          </div>
          
          <div className="relative">
            <input
              type="range"
              min="0"
              max="100"
              value={intensity}
              onChange={(e) => onIntensityChange(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                         slider-thumb"
              style={{
                background: `linear-gradient(to right, 
                  rgb(14, 165, 233) 0%, 
                  rgb(14, 165, 233) ${intensity}%, 
                  rgb(55, 65, 81) ${intensity}%, 
                  rgb(55, 65, 81) 100%)`
              }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-gray-500">
            <span>Original</span>
            <span>Full Style</span>
          </div>
        </div>

        {/* Info */}
        <div className="pt-4 border-t border-gray-700">
          <p className="text-xs text-gray-400 leading-relaxed">
            Adjust the intensity to blend between your original video and the artistic style.
            Higher values apply more of the selected style.
          </p>
        </div>
      </div>

      <style>{`
        .slider-thumb::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
          cursor: pointer;
          box-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
          transition: all 0.2s;
        }
        
        .slider-thumb::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 15px rgba(14, 165, 233, 0.8);
        }
        
        .slider-thumb::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
          cursor: pointer;
          border: none;
          box-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
          transition: all 0.2s;
        }
        
        .slider-thumb::-moz-range-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 15px rgba(14, 165, 233, 0.8);
        }
      `}</style>
    </div>
  );
}
