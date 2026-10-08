import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Cross-origin isolation lets the in-browser model use several threads.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { headers: isolation },
  preview: { headers: isolation },
})
