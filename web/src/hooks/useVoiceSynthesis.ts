import { useRef, useCallback, useState } from 'react';
import { synthesize } from '../api';

const DEBOUNCE_MS = 400;

export interface VoiceSynthesisHook {
  loading: boolean;
  requestSynthesis: (voice: string, text: string, styleCoefficients: number[]) => void;
}

export function useVoiceSynthesis(
  onAudioReady: (buffer: ArrayBuffer) => void,
): VoiceSynthesisHook {
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  const requestSynthesis = useCallback(
    (voice: string, text: string, styleCoefficients: number[]) => {
      // Clear pending debounce
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      timerRef.current = setTimeout(async () => {
        // Cancel in-flight request
        if (abortRef.current) {
          abortRef.current.abort();
        }
        abortRef.current = new AbortController();

        const id = ++requestIdRef.current;
        setLoading(true);

        try {
          const buffer = await synthesize({ voice, text, styleCoefficients });
          // Only use result if this is still the latest request
          if (id === requestIdRef.current) {
            onAudioReady(buffer);
          }
        } catch (err: unknown) {
          if (err instanceof DOMException && err.name === 'AbortError') return;
          console.error('Synthesis error:', err);
        } finally {
          if (id === requestIdRef.current) {
            setLoading(false);
          }
        }
      }, DEBOUNCE_MS);
    },
    [onAudioReady],
  );

  return { loading, requestSynthesis };
}
