import React, { useRef, useState } from 'react';
import styles from './VoiceSelector.module.css';
import type { VoiceInfo } from '../types';

interface Props {
  voices: VoiceInfo[];
  selected: string;
  onChange: (filename: string) => void;
  onUpload: (file: File) => Promise<void>;
}

export default function VoiceSelector({ voices, selected, onChange, onUpload }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await onUpload(file);
    } finally {
      setUploading(false);
      // Reset input so the same file can be re-uploaded
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className={styles.wrapper}>
      <label className={styles.label}>Base Voice</label>
      <div className={styles.row}>
        <select
          className={styles.select}
          value={selected}
          onChange={(e) => onChange(e.target.value)}
        >
          {[...voices].sort((a, b) => a.name.localeCompare(b.name)).map((v) => (
            <option key={v.filename} value={v.filename}>
              {v.name}
            </option>
          ))}
        </select>
        <button
          className={styles.uploadBtn}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title="Import a custom .pt voice file"
        >
          {uploading ? '...' : '+'}
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pt"
        className={styles.hiddenInput}
        onChange={handleFileChange}
      />
    </div>
  );
}
