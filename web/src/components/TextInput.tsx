import React from 'react';
import styles from './TextInput.module.css';

interface Props {
  value: string;
  onChange: (text: string) => void;
}

export default function TextInput({ value, onChange }: Props) {
  return (
    <div className={styles.wrapper}>
      <label className={styles.label}>Text to Speak</label>
      <textarea
        className={styles.textarea}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        placeholder="Enter text for the voice to speak..."
      />
    </div>
  );
}
