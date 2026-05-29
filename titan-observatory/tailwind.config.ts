import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: 'class',
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        titan: {
          bg: 'var(--titan-bg)',
          card: 'var(--titan-card)',
          metal: 'var(--titan-metal)',
          haze: 'var(--titan-haze)',
          pulse: 'var(--titan-pulse)',
          growth: 'var(--titan-growth)',
        },
      },
      boxShadow: {
        'haze-glow': '0 0 15px -1px rgba(229, 199, 158, 0.4)',
        'pulse-glow': '0 0 20px -1px rgba(153, 69, 255, 0.5)',
        'growth-glow': '0 0 15px -1px rgba(119, 204, 204, 0.3)',
        'card': 'var(--titan-card-shadow)',
      },
      fontFamily: {
        titan: ['Poppins', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        breathe: 'breathe 4s ease-in-out infinite',
        spark: 'spark 0.5s ease-out forwards',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { boxShadow: '0 0 15px -1px rgba(229, 199, 158, 0.2)' },
          '50%': { boxShadow: '0 0 25px -1px rgba(229, 199, 158, 0.6)' },
        },
        spark: {
          '0%': { opacity: '1', transform: 'scale(1)' },
          '100%': { opacity: '0', transform: 'scale(0.3)' },
        },
      },
    },
  },
  plugins: [],
};
export default config;
