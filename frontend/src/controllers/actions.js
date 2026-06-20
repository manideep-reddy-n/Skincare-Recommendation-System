import { apiUrl } from "./api";
import { compressImage } from "./imageUtils";

const REQUEST_TIMEOUT_MS = 120000;

const fetchWithTimeout = (url, options, timeoutMs = REQUEST_TIMEOUT_MS) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() =>
    clearTimeout(timeoutId)
  );
};

export const UploadImage = async (imageSrc, navigate) => {
  const compressed = await compressImage(imageSrc);
  const res = await fetchWithTimeout(apiUrl("/upload"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: compressed }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Upload failed (${res.status})`);
  }
  if (data.error) {
    throw new Error(data.error);
  }

  navigate("/form", { state: { data } });
};

export const putForm = async (features, currType, currTone, currAcne, navigate) => {
  const res = await fetchWithTimeout(apiUrl("/recommend"), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      features,
      type: currType,
      tone: currTone,
      acne: currAcne,
    }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Recommendation failed (${res.status})`);
  }
  if (data.error) {
    throw new Error(data.error);
  }

  navigate("/recs", { state: { data } });
};
