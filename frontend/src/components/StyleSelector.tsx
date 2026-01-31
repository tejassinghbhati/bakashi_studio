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

interface StyleSelectorProps {
  selectedStyle: string;
  onStyleChange: (styleId: string) => void;
}

export default function StyleSelector({ selectedStyle, onStyleChange }: StyleSelectorProps) {
  return (
    <div className="w-full">
      <div className="style-grid">
        {styles.map((style) => (
          <div key={style.id} className="flex flex-col items-center gap-1">
            <button
              onClick={() => onStyleChange(style.id)}
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
    </div>
  );
}
