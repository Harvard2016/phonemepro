// One IndexedDB database for everything kept in this browser. Nothing here is
// uploaded: there is no server to upload to.
//
//   attempts        practice history (engine/history.js)
//   contributions   accent summaries kept when "Help it learn" is on (engine/contributions.js)

const DB_NAME = 'phonemepro'
const DB_VERSION = 2

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
