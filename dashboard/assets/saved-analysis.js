// Keeps the visitor's latest analysis in this browser (IndexedDB) and remembers whether the dashboard should
// show it or the MetroGuard results. Nothing here leaves the device.

const DATABASE = "metroguard";
const STORE = "analyses";
const KEY = "latest";
const SOURCE_KEY = "mg-source";

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore(mode, action) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, mode);
    const request = action(transaction.objectStore(STORE));
    transaction.oncomplete = () => { db.close(); resolve(request.result); };
    transaction.onerror = () => { db.close(); reject(transaction.error); };
    transaction.onabort = () => { db.close(); reject(transaction.error ?? new Error("The browser refused to store the analysis.")); };
  });
}

export const saveAnalysis = (bundle) => withStore("readwrite", (store) => store.put(bundle, KEY));
export const loadAnalysis = () => withStore("readonly", (store) => store.get(KEY));
export const deleteAnalysis = () => withStore("readwrite", (store) => store.delete(KEY));

export function showingAnalysis() {
  try {
    return localStorage.getItem(SOURCE_KEY) === "upload";
  } catch {
    return false;
  }
}

export function setShowingAnalysis(on) {
  try {
    if (on) localStorage.setItem(SOURCE_KEY, "upload");
    else localStorage.removeItem(SOURCE_KEY);
  } catch {
    /* storage unavailable: the dashboard keeps showing the MetroGuard results */
  }
}
