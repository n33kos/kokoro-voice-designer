import React, { useEffect, useState, useCallback, useRef } from 'react';
import styles from './App.module.css';
import SpiderChart from './components/SpiderChart';
import VoiceSelector from './components/VoiceSelector';
import TextInput from './components/TextInput';
import PlaybackControls from './components/PlaybackControls';
import { useAudioEngine } from './hooks/useAudioEngine';
import { useVoiceSynthesis } from './hooks/useVoiceSynthesis';
import { fetchVoices, fetchCatalog, exportVoice, uploadVoice } from './api';
import type { VoiceInfo, StyleFeature } from './types';

const DEFAULT_TEXT = 'Hello, my name is Alex. How can I help you today?';

export default function App() {
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [text, setText] = useState(DEFAULT_TEXT);
  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [exportingVoice, setExportingVoice] = useState(false);
  // Slider values only. The 256-dim directions they map to live server-side, so
  // the client never needs the style map itself.
  const [styleCoefficients, setStyleCoefficients] = useState<number[]>([]);
  const [styleFeatures, setStyleFeatures] = useState<StyleFeature[]>([]);

  const audioEngine = useAudioEngine();
  const { loading, requestSynthesis } = useVoiceSynthesis(
    useCallback(
      (buffer: ArrayBuffer) => {
        audioEngine.loadAudio(buffer);
      },
      [audioEngine],
    ),
  );

  // Track whether we need to trigger initial synthesis after data loads
  const initialSynthDone = useRef(false);

  // Load data on mount
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const [voicesRes, catalogRes] = await Promise.all([
          fetchVoices(),
          fetchCatalog(),
        ]);
        if (cancelled) return;
        setVoices(voicesRes.voices);

        if (!catalogRes.hasStyleMap || !catalogRes.styleFeatures) {
          setInitError(
            'No style map found. Run "uv run python build_style_map.py" to generate one.',
          );
          return;
        }
        setStyleFeatures(catalogRes.styleFeatures);
        setStyleCoefficients(new Array(catalogRes.styleFeatures.length).fill(0));

        if (voicesRes.voices.length > 0) {
          const heart = voicesRes.voices.find((v) => v.filename === 'af_heart.pt');
          setSelectedVoice(heart ? heart.filename : voicesRes.voices[0].filename);
        }
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          setInitError(
            'Could not connect to backend. Make sure the server is running on port 8000.',
          );
        }
      }
    }
    init();
    return () => {
      cancelled = true;
    };
  }, []);

  const resynthesize = useCallback(
    (voice: string, newText: string) => {
      requestSynthesis(voice, [], newText, styleCoefficients);
    },
    [styleCoefficients, requestSynthesis],
  );

  // Trigger initial synthesis once ready
  useEffect(() => {
    if (ready && selectedVoice && text && !initialSynthDone.current) {
      initialSynthDone.current = true;
      resynthesize(selectedVoice, text);
    }
  }, [ready, selectedVoice, text, resynthesize]);

  const handleVoiceChange = useCallback(
    (filename: string) => {
      setSelectedVoice(filename);
      resynthesize(filename, text);
    },
    [text, resynthesize],
  );

  const handleVoiceUpload = useCallback(
    async (file: File) => {
      try {
        const result = await uploadVoice(file);
        const voicesRes = await fetchVoices();
        setVoices(voicesRes.voices);
        setSelectedVoice(result.filename);
        resynthesize(result.filename, text);
      } catch (err) {
        console.error('Voice upload failed:', err);
      }
    },
    [text, resynthesize],
  );

  const handleTextChange = useCallback(
    (newText: string) => {
      setText(newText);
      if (newText.trim()) {
        resynthesize(selectedVoice, newText);
      }
    },
    [selectedVoice, resynthesize],
  );

  const handleAxisChange = useCallback(
    (index: number, value: number) => {
      setStyleCoefficients((prev) => {
        const next = [...prev];
        next[index] = value;
        requestSynthesis(selectedVoice, [], text, next);
        return next;
      });
    },
    [selectedVoice, text, requestSynthesis],
  );

  const handleReset = useCallback(() => {
    const zeroed = new Array(styleFeatures.length).fill(0);
    setStyleCoefficients(zeroed);
    requestSynthesis(selectedVoice, [], text, zeroed);
  }, [styleFeatures.length, selectedVoice, text, requestSynthesis]);

  const triggerDownload = useCallback((data: ArrayBuffer, filename: string, mime: string) => {
    const blob = new Blob([data], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const handleExportVoice = useCallback(async () => {
    if (!selectedVoice) return;
    setExportingVoice(true);
    try {
      const data = await exportVoice({
        voice: selectedVoice,
        coefficients: [],
        styleCoefficients,
      });
      triggerDownload(data, 'designed_voice.pt', 'application/octet-stream');
    } catch (err) {
      console.error('Voice export failed:', err);
    } finally {
      setExportingVoice(false);
    }
  }, [selectedVoice, styleCoefficients, triggerDownload]);

  const handleExportAudio = useCallback(() => {
    const wavBuffer = audioEngine.getLastWavBuffer();
    if (!wavBuffer) return;
    triggerDownload(wavBuffer, 'designed_voice.wav', 'audio/wav');
  }, [audioEngine, triggerDownload]);

  if (initError) {
    return (
      <div className={styles.app}>
        <div className={styles.error}>
          <h2>Connection Error</h2>
          <p>{initError}</p>
        </div>
      </div>
    );
  }

  if (!ready) {
    return (
      <div className={styles.app}>
        <div className={styles.loading}>
          <div className={styles.loadingSpinner} />
          <p>Loading Voice Designer...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.title}>Voice Designer</h1>
        <p className={styles.subtitle}>
          Drag points on the chart to shape the voice
        </p>
      </header>

      <main className={styles.main}>
        <div className={styles.chartPanel}>
          <SpiderChart
            labels={styleFeatures.map((f) => f.name)}
            values={styleCoefficients}
            onChange={handleAxisChange}
          />
        </div>

        <div className={styles.controlsPanel}>
          <VoiceSelector
            voices={voices}
            selected={selectedVoice}
            onChange={handleVoiceChange}
            onUpload={handleVoiceUpload}
          />

          <TextInput value={text} onChange={handleTextChange} />

          <PlaybackControls
            isPlaying={audioEngine.isPlaying}
            loading={loading}
            onPlay={audioEngine.play}
            onPause={audioEngine.pause}
            onReset={handleReset}
            onExportVoice={handleExportVoice}
            onExportAudio={handleExportAudio}
            exportingVoice={exportingVoice}
          />
        </div>
      </main>
    </div>
  );
}
