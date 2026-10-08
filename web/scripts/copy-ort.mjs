// Copy the ONNX Runtime WebAssembly binary and its loader next to the app, so it has no CDN
// dependency and its worker threads load the loader alone (see src/engine/model.js).
import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const source = join(root, 'node_modules', 'onnxruntime-web', 'dist')
const target = join(root, 'public', 'ort')

mkdirSync(target, { recursive: true })
for (const file of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
  copyFileSync(join(source, file), join(target, file))
}
