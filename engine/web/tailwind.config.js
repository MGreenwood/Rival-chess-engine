/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
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
    },
  },
  plugins: [],
} 