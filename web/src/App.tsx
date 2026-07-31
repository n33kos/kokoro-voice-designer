import React, { useEffect, useState, useCallback, useRef } from 'react';
import styles from './App.module.css';
import SpiderChart from './components/SpiderChart';
import VoiceSelector from './components/VoiceSelector';
import TextInput from './components/TextInput';
import PlaybackControls from './components/PlaybackControls';
import { useAudioEngine } from './hooks/useAudioEngine';
import { useVoiceSynthesis } from './hooks/useVoiceSynthesis';
import { fetchVoices, fetchCatalog, exportVoice, uploadVoice } from './api';
import type { VoiceInfo, ComponentInfo, SemanticFeature, StyleFeature } from './types';

const DEFAULT_TEXT = 'Hello, my name is Alex. How can I help you today?';

/**
 * 'style' is the v2 map: axes in Kokoro's native 256-dim style space, measured
 * by backpropagation and masked to the half that controls them. 'semantic' is
 * the older component-space map, kept while the two are compared.
 */
type Mode = 'raw' | 'semantic' | 'style';

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
  const [mode, setMode] = useState<Mode>('raw');
  const [semanticCoefficients, setSemanticCoefficients] = useState<number[]>([]);
  const [semanticFeatures, setSemanticFeatures] = useState<SemanticFeature[]>([]);
  const [semanticDirections, setSemanticDirections] = useState<number[][] | null>(null);
  const [hasSemanticMap, setHasSemanticMap] = useState(false);
  const [styleCoefficients, setStyleCoefficients] = useState<number[]>([]);
  const [styleFeatures, setStyleFeatures] = useState<StyleFeature[]>([]);
  const [hasStyleMap, setHasStyleMap] = useState(false);

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
        if (catalogRes.hasSemanticMap && catalogRes.semanticFeatures && catalogRes.semanticDirections) {
          setHasSemanticMap(true);
          setSemanticFeatures(catalogRes.semanticFeatures);
          // Directions arrive L1-normalized from build_semantic_map.py, which
          // bounds the summed perturbation to roughly one raw slider at 1.0.
          // Don't renormalize here — doing so discards that scaling and lets
          // sliders past ~0.25 distort.
          setSemanticDirections(catalogRes.semanticDirections);
          setSemanticCoefficients(new Array(catalogRes.semanticFeatures.length).fill(0));
        }
        if (catalogRes.hasStyleMap && catalogRes.styleFeatures) {
          setHasStyleMap(true);
          setStyleFeatures(catalogRes.styleFeatures);
          setStyleCoefficients(new Array(catalogRes.styleFeatures.length).fill(0));
          // Style axes are both more accurate and more meaningful than raw
          // components, so lead with them when they're available.
          setMode('style');
        }
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

  /**
   * Synthesize with whichever coefficient set the current mode owns. Style mode
   * sends slider values only — the server holds the 256-dim directions.
   */
  const resynthesize = useCallback(
    (voice: string, newText: string) => {
      if (mode === 'style') {
        requestSynthesis(voice, [], newText, styleCoefficients);
      } else {
        requestSynthesis(voice, coefficients, newText);
      }
    },
    [mode, coefficients, styleCoefficients, requestSynthesis],
  );

  // Trigger initial synthesis once ready
  useEffect(() => {
    if (ready && selectedVoice && text && !initialSynthDone.current) {
      initialSynthDone.current = true;
      resynthesize(selectedVoice, text);
    }
  }, [ready, selectedVoice, text, resynthesize]);

  // Re-synthesize when voice or text changes (but not on initial load)
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
        // Refresh voice list
        const voicesRes = await fetchVoices();
        setVoices(voicesRes.voices);
        // Select the uploaded voice
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

  /** Multiply semantic coefficients by directions matrix to get raw coefficients */
  const semanticToRaw = useCallback(
    (semCoeffs: number[]): number[] => {
      if (!semanticDirections) return new Array(components.length).fill(0);
      const nComponents = components.length;
      const raw = new Array(nComponents).fill(0);
      for (let i = 0; i < semCoeffs.length; i++) {
        if (Math.abs(semCoeffs[i]) < 1e-10) continue;
        const dir = semanticDirections[i];
        if (!dir) continue;
        for (let j = 0; j < nComponents && j < dir.length; j++) {
          raw[j] += semCoeffs[i] * dir[j];
        }
      }
      return raw;
    },
    [semanticDirections, components.length],
  );

  const handleAxisChange = useCallback(
    (index: number, value: number) => {
      if (mode === 'style') {
        // Style directions live server-side in 256-dim space; we only send
        // slider values and the server applies the offsets during synthesis.
        setStyleCoefficients((prev) => {
          const next = [...prev];
          next[index] = value;
          requestSynthesis(selectedVoice, [], text, next);
          return next;
        });
      } else if (mode === 'semantic') {
        setSemanticCoefficients((prev) => {
          const next = [...prev];
          next[index] = value;
          const rawCoeffs = semanticToRaw(next);
          setCoefficients(rawCoeffs);
          requestSynthesis(selectedVoice, rawCoeffs, text);
          return next;
        });
      } else {
        setCoefficients((prev) => {
          const next = [...prev];
          next[index] = value;
          requestSynthesis(selectedVoice, next, text);
          return next;
        });
      }
    },
    [mode, selectedVoice, text, requestSynthesis, semanticToRaw],
  );

  const handleReset = useCallback(() => {
    if (mode === 'style') {
      const zeroed = new Array(styleFeatures.length).fill(0);
      setStyleCoefficients(zeroed);
      requestSynthesis(selectedVoice, [], text, zeroed);
    } else if (mode === 'semantic') {
      const zeroed = new Array(semanticFeatures.length).fill(0);
      setSemanticCoefficients(zeroed);
      const rawZeroed = new Array(components.length).fill(0);
      setCoefficients(rawZeroed);
      requestSynthesis(selectedVoice, rawZeroed, text);
    } else {
      const zeroed = new Array(components.length).fill(0);
      setCoefficients(zeroed);
      requestSynthesis(selectedVoice, zeroed, text);
    }
  }, [mode, components.length, semanticFeatures.length, styleFeatures.length,
      selectedVoice, text, requestSynthesis]);

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

  const handleModeChange = useCallback(
    (newMode: Mode) => {
      if (newMode === mode) return;
      setMode(newMode);
      if (newMode === 'style') {
        const zeroed = new Array(styleFeatures.length).fill(0);
        setStyleCoefficients(zeroed);
        requestSynthesis(selectedVoice, [], text, zeroed);
      } else if (newMode === 'semantic') {
        // Reset semantic coefficients when switching to semantic mode
        setSemanticCoefficients(new Array(semanticFeatures.length).fill(0));
        const rawZeroed = new Array(components.length).fill(0);
        setCoefficients(rawZeroed);
        requestSynthesis(selectedVoice, rawZeroed, text);
      } else {
        // Reset raw coefficients when switching to raw mode
        const zeroed = new Array(components.length).fill(0);
        setCoefficients(zeroed);
        requestSynthesis(selectedVoice, zeroed, text);
      }
    },
    [mode, semanticFeatures.length, styleFeatures.length, components.length,
     selectedVoice, text, requestSynthesis],
  );

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

  const isSemantic = mode === 'semantic';
  const isStyle = mode === 'style';
  const rawCount = Math.min(displayCount, components.length);
  const chartLabels = isStyle
    ? styleFeatures.map((f) => f.name)
    : isSemantic
      ? semanticFeatures.map((f) => f.name)
      : components.slice(0, rawCount).map((c) => c.name);
  const chartValues = isStyle
    ? styleCoefficients
    : isSemantic
      ? semanticCoefficients
      : coefficients.slice(0, rawCount);
  const visibleCount = isStyle
    ? styleFeatures.length
    : isSemantic
      ? semanticFeatures.length
      : rawCount;

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
          {(hasSemanticMap || hasStyleMap) && (
            <div className={styles.modeToggle}>
              {hasStyleMap && (
                <button
                  className={`${styles.modeBtn} ${isStyle ? styles.modeBtnActive : ''}`}
                  onClick={() => handleModeChange('style')}
                  title="Axes in Kokoro's native style space, measured by backpropagation"
                >
                  Style
                </button>
              )}
              <button
                className={`${styles.modeBtn} ${mode === 'raw' ? styles.modeBtnActive : ''}`}
                onClick={() => handleModeChange('raw')}
              >
                Raw
              </button>
              {hasSemanticMap && (
                <button
                  className={`${styles.modeBtn} ${isSemantic ? styles.modeBtnActive : ''}`}
                  onClick={() => handleModeChange('semantic')}
                  title="Older component-space map"
                >
                  Semantic
                </button>
              )}
            </div>
          )}
          <SpiderChart
            labels={chartLabels}
            values={chartValues}
            onChange={handleAxisChange}
          />
          {!isSemantic && !isStyle && (
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
          )}
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
