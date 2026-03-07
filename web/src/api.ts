import type { CatalogResponse, VoicesResponse, SynthesizeRequest, ExportVoiceRequest } from './types';

const BASE = '/api';

export async function fetchVoices(): Promise<VoicesResponse> {
  const res = await fetch(`${BASE}/voices`);
  if (!res.ok) throw new Error('Failed to fetch voices');
  return res.json();
}

export async function fetchCatalog(): Promise<CatalogResponse> {
  const res = await fetch(`${BASE}/catalog`);
  if (!res.ok) throw new Error('Failed to fetch catalog');
  return res.json();
}

export async function synthesize(req: SynthesizeRequest): Promise<ArrayBuffer> {
  const res = await fetch(`${BASE}/synthesize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Synthesis failed: ${detail}`);
  }
  return res.arrayBuffer();
}

export async function uploadVoice(file: File): Promise<{ filename: string; name: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/upload-voice`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed: ${detail}`);
  }
  return res.json();
}

export async function exportVoice(req: ExportVoiceRequest): Promise<ArrayBuffer> {
  const res = await fetch(`${BASE}/export-voice`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Export failed: ${detail}`);
  }
  return res.arrayBuffer();
}
