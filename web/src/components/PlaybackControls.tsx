import React from 'react';
import styles from './PlaybackControls.module.css';

interface Props {
  isPlaying: boolean;
  loading: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onExportVoice: () => void;
  onExportAudio: () => void;
  exportingVoice: boolean;
}

export default function PlaybackControls({
  isPlaying,
  loading,
  onPlay,
  onPause,
  onReset,
  onExportVoice,
  onExportAudio,
  exportingVoice,
}: Props) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.playRow}>
        <button
          className={`${styles.playBtn} ${isPlaying ? styles.playing : ''}`}
          onClick={isPlaying ? onPause : onPlay}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
              <rect x="4" y="3" width="4" height="14" rx="1" />
              <rect x="12" y="3" width="4" height="14" rx="1" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
              <path d="M5 3.5L16 10L5 16.5V3.5Z" />
            </svg>
          )}
        </button>

        <button className={styles.resetBtn} onClick={onReset} title="Reset all axes to neutral">
          Reset
        </button>

        {loading && (
          <div className={styles.loadingIndicator}>
            <div className={styles.spinner} />
            <span className={styles.loadingText}>synthesizing...</span>
          </div>
        )}
      </div>

      <div className={styles.exportRow}>
        <button
          className={styles.exportBtn}
          onClick={onExportVoice}
          disabled={exportingVoice}
          title="Download the designed voice as a .pt file"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M7 1v8M3.5 6.5L7 10l3.5-3.5M2 12h10" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {exportingVoice ? 'Saving...' : 'Save Voice (.pt)'}
        </button>
        <button
          className={styles.exportBtn}
          onClick={onExportAudio}
          title="Download the last synthesized audio as a .wav file"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M7 1v8M3.5 6.5L7 10l3.5-3.5M2 12h10" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Save Audio (.wav)
        </button>
      </div>
    </div>
  );
}
