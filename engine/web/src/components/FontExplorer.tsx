import React, { useState } from 'react';

interface FontOption {
  name: string;
  className: string;
  description: string;
  category: string;
}

const fontOptions: FontOption[] = [
  // Modern & Clean
  { name: 'Inter', className: 'font-inter', description: 'Modern, highly readable, tech industry favorite', category: 'Modern & Clean' },
  { name: 'Space Grotesk', className: 'font-space', description: 'Geometric, futuristic, great for headings', category: 'Modern & Clean' },
  { name: 'Manrope', className: 'font-manrope', description: 'Open, friendly, excellent readability', category: 'Modern & Clean' },
  
  // Popular & Friendly
  { name: 'Poppins', className: 'font-poppins', description: 'Rounded, approachable, very popular', category: 'Popular & Friendly' },
  { name: 'Nunito', className: 'font-nunito', description: 'Soft, rounded, warm feeling', category: 'Popular & Friendly' },
  
  // Professional & Corporate
  { name: 'Source Sans 3', className: 'font-source', description: 'Adobe\'s workhorse, professional', category: 'Professional & Corporate' },
  { name: 'Roboto', className: 'font-roboto', description: 'Google\'s standard, mechanical precision', category: 'Professional & Corporate' },
  { name: 'IBM Plex Sans', className: 'font-ibm', description: 'Corporate, technical, modern', category: 'Professional & Corporate' },
  
  // Current
  { name: 'System Default', className: 'font-system', description: 'Current system font stack', category: 'Current' },
];

export const FontExplorer: React.FC = () => {
  const [selectedFont, setSelectedFont] = useState<FontOption>(fontOptions[0]);

  const sampleTexts = {
    heading: 'Rival Chess',
    subheading: 'Advanced AI Chess Training',
    body: 'Experience the future of chess training with our advanced AI. Challenge yourself against sophisticated algorithms that adapt to your playing style and help you improve your game.',
    ui: 'Settings • New Game • Community Challenge',
    numbers: '1,234 games played • 87% win rate • Elo 1,847'
  };

  const groupedFonts = fontOptions.reduce((acc, font) => {
    if (!acc[font.category]) acc[font.category] = [];
    acc[font.category].push(font);
    return acc;
  }, {} as Record<string, FontOption[]>);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-8">
          Font Explorer
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Font Selection Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
              <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
                Choose Font
              </h2>
              
              {Object.entries(groupedFonts).map(([category, fonts]) => (
                <div key={category} className="mb-6">
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
                    {category}
                  </h3>
                  <div className="space-y-2">
                    {fonts.map((font) => (
                      <button
                        key={font.name}
                        onClick={() => setSelectedFont(font)}
                        className={`w-full text-left p-3 rounded-lg transition-colors ${
                          selectedFont.name === font.name
                            ? 'bg-blue-100 dark:bg-blue-900 border-2 border-blue-500'
                            : 'bg-gray-50 dark:bg-gray-700 border-2 border-transparent hover:bg-gray-100 dark:hover:bg-gray-600'
                        }`}
                      >
                        <div className={`font-medium ${font.className} text-gray-900 dark:text-gray-100`}>
                          {font.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {font.description}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Preview Panel */}
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-8 shadow-lg">
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
                  Preview: {selectedFont.name}
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  {selectedFont.description}
                </p>
              </div>

              <div className={`space-y-8 ${selectedFont.className}`}>
                {/* Main Heading */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Main Heading
                  </label>
                  <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mt-2">
                    {sampleTexts.heading}
                  </h1>
                </div>

                {/* Subheading */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Subheading
                  </label>
                  <h2 className="text-xl font-medium text-gray-700 dark:text-gray-300 mt-2">
                    {sampleTexts.subheading}
                  </h2>
                </div>

                {/* Body Text */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Body Text
                  </label>
                  <p className="text-base text-gray-600 dark:text-gray-300 leading-relaxed mt-2">
                    {sampleTexts.body}
                  </p>
                </div>

                {/* UI Elements */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    UI Elements
                  </label>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                    {sampleTexts.ui}
                  </p>
                </div>

                {/* Numbers */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Numbers & Stats
                  </label>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mt-2">
                    {sampleTexts.numbers}
                  </p>
                </div>

                {/* Font Weights */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Font Weights
                  </label>
                  <div className="mt-2 space-y-1">
                    <p className="font-light text-gray-600 dark:text-gray-300">Light: The quick brown fox</p>
                    <p className="font-normal text-gray-600 dark:text-gray-300">Regular: The quick brown fox</p>
                    <p className="font-medium text-gray-600 dark:text-gray-300">Medium: The quick brown fox</p>
                    <p className="font-semibold text-gray-600 dark:text-gray-300">Semibold: The quick brown fox</p>
                    <p className="font-bold text-gray-600 dark:text-gray-300">Bold: The quick brown fox</p>
                  </div>
                </div>

                {/* Button Preview */}
                <div>
                  <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Buttons
                  </label>
                  <div className="flex gap-3 mt-2">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">
                      Primary Button
                    </button>
                    <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                      Secondary
                    </button>
                  </div>
                </div>
              </div>

              {/* Apply Button */}
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => {
                    // Remove any existing font classes
                    const fontClasses = ['font-inter', 'font-space', 'font-manrope', 'font-poppins', 'font-nunito', 'font-source', 'font-roboto', 'font-ibm', 'font-system'];
                    fontClasses.forEach(cls => document.body.classList.remove(cls));
                    
                    // Apply the selected font class to body
                    document.body.classList.add(selectedFont.className);
                    
                    // Also update CSS variable as fallback
                    const fontFamily = selectedFont.className === 'font-inter' ? "'Inter', system-ui, sans-serif" :
                      selectedFont.className === 'font-space' ? "'Space Grotesk', system-ui, sans-serif" :
                      selectedFont.className === 'font-manrope' ? "'Manrope', system-ui, sans-serif" :
                      selectedFont.className === 'font-poppins' ? "'Poppins', system-ui, sans-serif" :
                      selectedFont.className === 'font-nunito' ? "'Nunito', system-ui, sans-serif" :
                      selectedFont.className === 'font-source' ? "'Source Sans 3', system-ui, sans-serif" :
                      selectedFont.className === 'font-roboto' ? "'Roboto', system-ui, sans-serif" :
                      selectedFont.className === 'font-ibm' ? "'IBM Plex Sans', system-ui, sans-serif" :
                      "system-ui, Avenir, Helvetica, Arial, sans-serif";
                    
                    document.documentElement.style.setProperty('--font-family', fontFamily);
                    
                    alert(`${selectedFont.name} applied globally! Navigate to other pages to see it in action.`);
                  }}
                  className="w-full px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors"
                >
                  Apply {selectedFont.name} Globally
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 