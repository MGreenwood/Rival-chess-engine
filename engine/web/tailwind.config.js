/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'chess-bg': '#1a1b26',
        'chess-dark': '#13141f',
        'chess-darker': '#0d0e14',
        'chess-light': '#282a3b',
        'chess-lighter': '#363850',
        'chess-border': '#2a2b3d',
      },
      fontFamily: {
        // Modern & Clean
        'inter': ['Inter', 'system-ui', 'sans-serif'],
        'space': ['Space Grotesk', 'system-ui', 'sans-serif'],
        'manrope': ['Manrope', 'system-ui', 'sans-serif'],
        
        // Popular & Friendly
        'poppins': ['Poppins', 'system-ui', 'sans-serif'],
        'nunito': ['Nunito', 'system-ui', 'sans-serif'],
        
        // Professional & Corporate
        'source': ['Source Sans 3', 'system-ui', 'sans-serif'],
        'roboto': ['Roboto', 'system-ui', 'sans-serif'],
        'ibm': ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        
        // Current system default
        'system': ['system-ui', 'Avenir', 'Helvetica', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [],
} 