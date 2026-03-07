import React, { useEffect, useState, useCallback, useRef } from 'react';
import styles from './App.module.css';
import SpiderChart from './components/SpiderChart';
import VoiceSelector from './components/VoiceSelector';
import TextInput from './components/TextInput';
import PlaybackControls from './components/PlaybackControls';
import { useAudioEngine } from './hooks/useAudioEngine';
import { useVoiceSynthesis } from './hooks/useVoiceSynthesis';
import { fetchVoices, fetchCatalog, exportVoice, uploadVoice } from './api';
import type { VoiceInfo, ComponentInfo } from './types';

const DEFAULT_TEXT = 'Hello, my name is Alex. How can I help you today?';

export default function App() {
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [components, setComponents] = useState<ComponentInfo[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [text, setText] = useState(DEFAULT_TEXT);
  const [coefficients, setCoefficients] = useState<number[]>([]);
  const [displayCount, setDisplayCount] = useState(20);
  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [exportingVoice, setExportingVoice] = useState(false);

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
        setComponents(catalogRes.components);
        // Keep the hardcoded default of 20; don't override from catalog
        if (voicesRes.voices.length > 0) {
          // Pick af_heart if available, else first
          const heart = voicesRes.voices.find((v) => v.filename === 'af_heart.pt');
          setSelectedVoice(heart ? heart.filename : voicesRes.voices[0].filename);
        }
        setCoefficients(new Array(catalogRes.count).fill(0));
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

  // Trigger initial synthesis once ready
  useEffect(() => {
    if (ready && selectedVoice && text && !initialSynthDone.current) {
      initialSynthDone.current = true;
      requestSynthesis(selectedVoice, coefficients, text);
    }
  }, [ready, selectedVoice, text, coefficients, requestSynthesis]);

  // Re-synthesize when voice or text changes (but not on initial load)
  const handleVoiceChange = useCallback(
    (filename: string) => {
      setSelectedVoice(filename);
      requestSynthesis(filename, coefficients, text);
    },
    [coefficients, text, requestSynthesis],
  );

  const handleVoiceUpload = useCallback(
    async (file: File) => {
      try {
        const result = await uploadVoice(file);
        // Refresh voice list
        const voicesRes = await fetchVoices();
        setVoices(voicesRes.voices);
        // Select the uploaded voice
        setSelectedVoice(result.filename);
        requestSynthesis(result.filename, coefficients, text);
      } catch (err) {
        console.error('Voice upload failed:', err);
      }
    },
    [coefficients, text, requestSynthesis],
  );

  const handleTextChange = useCallback(
    (newText: string) => {
      setText(newText);
      if (newText.trim()) {
        requestSynthesis(selectedVoice, coefficients, newText);
      }
    },
    [selectedVoice, coefficients, requestSynthesis],
  );

  const handleAxisChange = useCallback(
    (index: number, value: number) => {
      setCoefficients((prev) => {
        const next = [...prev];
        next[index] = value;
        requestSynthesis(selectedVoice, next, text);
        return next;
      });
    },
    [selectedVoice, text, requestSynthesis],
  );

  const handleReset = useCallback(() => {
    const zeroed = new Array(components.length).fill(0);
    setCoefficients(zeroed);
    requestSynthesis(selectedVoice, zeroed, text);
  }, [components.length, selectedVoice, text, requestSynthesis]);

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
      const data = await exportVoice({ voice: selectedVoice, coefficients });
      triggerDownload(data, 'designed_voice.pt', 'application/octet-stream');
    } catch (err) {
      console.error('Voice export failed:', err);
    } finally {
      setExportingVoice(false);
    }
  }, [selectedVoice, coefficients, triggerDownload]);

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

  const visibleCount = Math.min(displayCount, components.length);
  const visibleLabels = components.slice(0, visibleCount).map((c) => c.name);
  const visibleValues = coefficients.slice(0, visibleCount);

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
            labels={visibleLabels}
            values={visibleValues}
            onChange={handleAxisChange}
          />
          <div className={styles.dimensionControl}>
            <label className={styles.dimensionLabel}>
              Dimensions: {visibleCount}
            </label>
            <input
              type="range"
              className={styles.dimensionSlider}
              min={4}
              max={components.length}
              value={visibleCount}
              onChange={(e) => setDisplayCount(Number(e.target.value))}
            />
          </div>
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
