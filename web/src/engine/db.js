import { isRegion } from '../lib/regions'

// One IndexedDB database for everything kept in this browser. Nothing here is
// uploaded: there is no server to upload to.
//
//   attempts        practice history (engine/history.js)
//   contributions   accent summaries kept when "Help it learn" is on (engine/contributions.js)

const DB_NAME = 'phonemepro'
const DB_VERSION = 3

let opening = null

function open() {
  if (!opening) {
    opening = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION)
      request.onupgradeneeded = () => {
        const db = request.result
        if (!db.objectStoreNames.contains('attempts')) {
          db.createObjectStore('attempts', { keyPath: 'id', autoIncrement: true })
        }
        if (!db.objectStoreNames.contains('contributions')) {
          db.createObjectStore('contributions', { keyPath: 'id' })
        }
        // Version 3: region became a fixed code. Takes stored before that may hold typed
        // text; the text is dropped and the take is kept.
        const cursor = request.transaction.objectStore('contributions').openCursor()
        cursor.onsuccess = () => {
          const row = cursor.result
          if (!row) return
          if (row.value.region !== null && !isRegion(row.value.country, row.value.region)) {
            row.update({ ...row.value, region: null })
          }
          row.continue()
        }
      }
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => reject(request.error)
    })
    opening.catch(() => { opening = null })
  }
  return opening
}

// Run `work(store)` in one transaction and resolve with its result once the transaction commits.
export async function run(storeName, mode, work) {
  const db = await open()
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, mode)
    const result = work(transaction.objectStore(storeName))
    transaction.oncomplete = () => resolve(result?.result ?? result)
    transaction.onerror = () => reject(transaction.error)
    transaction.onabort = () => reject(transaction.error)
  })
}
