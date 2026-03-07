import { useRef, useCallback, useState } from 'react';

const CROSSFADE_DURATION = 0.3; // seconds

export interface AudioEngine {
  isPlaying: boolean;
  play: () => void;
  pause: () => void;
  loadAudio: (wavBuffer: ArrayBuffer) => Promise<void>;
  getLastWavBuffer: () => ArrayBuffer | null;
}

export function useAudioEngine(): AudioEngine {
  const ctxRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const currentGainRef = useRef<GainNode | null>(null);
  const currentBufferRef = useRef<AudioBuffer | null>(null);
  const lastWavBufferRef = useRef<ArrayBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const playingRef = useRef(false);
  // Track playback position for seamless looping across source swaps
  const playbackStartTimeRef = useRef(0);
  const playbackOffsetRef = useRef(0);

  const getContext = useCallback(() => {
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext();
    }
    return ctxRef.current;
  }, []);

  const startSource = useCallback(
    (buffer: AudioBuffer, gainNode: GainNode, startGain: number) => {
      const ctx = getContext();
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.loop = true;
      source.connect(gainNode);
      gainNode.connect(ctx.destination);
      gainNode.gain.setValueAtTime(startGain, ctx.currentTime);

      // Resume from the tracked offset so crossfade feels mid-sentence
      const offset = playbackOffsetRef.current % buffer.duration;
      source.start(0, offset);
      playbackStartTimeRef.current = ctx.currentTime;
      playbackOffsetRef.current = offset;
      return source;
    },
    [getContext],
  );

  const stopSource = useCallback(
    (source: AudioBufferSourceNode | null, gain: GainNode | null) => {
      if (source) {
        try {
          source.stop();
        } catch {
          // already stopped
        }
        source.disconnect();
      }
      if (gain) {
        gain.disconnect();
      }
    },
    [],
  );

  const snapshotOffset = useCallback(() => {
    const ctx = ctxRef.current;
    if (ctx && currentBufferRef.current) {
      const elapsed = ctx.currentTime - playbackStartTimeRef.current;
      playbackOffsetRef.current =
        (playbackOffsetRef.current + elapsed) % currentBufferRef.current.duration;
    }
  }, []);

  const play = useCallback(() => {
    const ctx = getContext();
    if (ctx.state === 'suspended') {
      ctx.resume();
    }
    const buffer = currentBufferRef.current;
    if (!buffer) return;

    if (currentSourceRef.current) {
      stopSource(currentSourceRef.current, currentGainRef.current);
    }

    const gain = ctx.createGain();
    const source = startSource(buffer, gain, 1.0);
    currentSourceRef.current = source;
    currentGainRef.current = gain;
    playingRef.current = true;
    setIsPlaying(true);
  }, [getContext, startSource, stopSource]);

  const pause = useCallback(() => {
    snapshotOffset();
    stopSource(currentSourceRef.current, currentGainRef.current);
    currentSourceRef.current = null;
    currentGainRef.current = null;
    playingRef.current = false;
    setIsPlaying(false);
  }, [snapshotOffset, stopSource]);

  const loadAudio = useCallback(
    async (wavBuffer: ArrayBuffer) => {
      lastWavBufferRef.current = wavBuffer.slice(0);
      const ctx = getContext();
      const newBuffer = await ctx.decodeAudioData(wavBuffer.slice(0));

      if (!playingRef.current) {
        // Not playing -- just store the buffer
        currentBufferRef.current = newBuffer;
        return;
      }

      // Crossfade from old to new
      snapshotOffset();
      const now = ctx.currentTime;

      // Fade out old
      if (currentGainRef.current) {
        currentGainRef.current.gain.setValueAtTime(1.0, now);
        currentGainRef.current.gain.linearRampToValueAtTime(0.0, now + CROSSFADE_DURATION);
      }
      const oldSource = currentSourceRef.current;
      const oldGain = currentGainRef.current;

      // Fade in new
      const newGain = ctx.createGain();
      const newSource = startSource(newBuffer, newGain, 0.0);
      newGain.gain.linearRampToValueAtTime(1.0, now + CROSSFADE_DURATION);

      currentSourceRef.current = newSource;
      currentGainRef.current = newGain;
      currentBufferRef.current = newBuffer;

      // Clean up old after crossfade completes
      setTimeout(() => {
        stopSource(oldSource, oldGain);
      }, CROSSFADE_DURATION * 1000 + 50);
    },
    [getContext, snapshotOffset, startSource, stopSource],
  );

  const getLastWavBuffer = useCallback(() => lastWavBufferRef.current, []);

  return { isPlaying, play, pause, loadAudio, getLastWavBuffer };
}
