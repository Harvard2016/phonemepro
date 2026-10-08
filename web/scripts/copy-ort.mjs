// Copy the ONNX Runtime WebAssembly binary next to the app so it has no CDN dependency.
import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const source = join(root, 'node_modules', 'onnxruntime-web', 'dist')
const target = join(root, 'public', 'ort')

mkdirSync(target, { recursive: true })
copyFileSync(join(source, 'ort-wasm-simd-threaded.wasm'), join(target, 'ort-wasm-simd-threaded.wasm'))
