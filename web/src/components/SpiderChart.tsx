import React, { useRef, useEffect, useCallback, useState } from 'react';
import styles from './SpiderChart.module.css';

interface Props {
  labels: string[];
  values: number[]; // each in [-1, 1]
  onChange: (index: number, value: number) => void;
}

const RINGS = [0.25, 0.5, 0.75, 1.0];
const BG = '#0e1117';
const GRID_COLOR = 'rgba(255,255,255,0.08)';
const AXIS_COLOR = 'rgba(255,255,255,0.15)';
const LABEL_COLOR = 'rgba(255,255,255,0.7)';
const ACCENT = 'rgba(99,179,237,0.85)';
const ACCENT_FILL = 'rgba(99,179,237,0.12)';
const DOT_COLOR = '#63b3ed';
const DOT_ACTIVE = '#90cdf4';

function toCanvas(
  index: number,
  value: number,
  count: number,
  cx: number,
  cy: number,
  radius: number,
): [number, number] {
  // Map value from [-1, 1] to [0, 1] for radius (center = -1, edge = 1, neutral = 0 at 0.5)
  const norm = (value + 1) / 2;
  const angle = (Math.PI * 2 * index) / count - Math.PI / 2;
  return [cx + Math.cos(angle) * radius * norm, cy + Math.sin(angle) * radius * norm];
}

/** Normalize angle to [0, 2*PI) */
function normalizeAngle(a: number): number {
  const twoPi = Math.PI * 2;
  return ((a % twoPi) + twoPi) % twoPi;
}

export default function SpiderChart({ labels, values, onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<number | null>(null);
  const isDraggingSweep = useRef(false);
  const [size, setSize] = useState(500);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const s = Math.min(entry.contentRect.width, entry.contentRect.height);
        setSize(Math.max(300, s));
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const count = labels.length;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.35;
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || count === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, size, size);

    // Grid rings
    for (const ring of RINGS) {
      ctx.beginPath();
      for (let i = 0; i <= count; i++) {
        const angle = (Math.PI * 2 * (i % count)) / count - Math.PI / 2;
        const x = cx + Math.cos(angle) * radius * ring;
        const y = cy + Math.sin(angle) * radius * ring;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Axis lines and labels
    ctx.font = `500 11px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
      const ex = cx + Math.cos(angle) * radius;
      const ey = cy + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = AXIS_COLOR;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      const labelDist = radius + 22;
      const lx = cx + Math.cos(angle) * labelDist;
      const ly = cy + Math.sin(angle) * labelDist;
      ctx.fillStyle = LABEL_COLOR;
      ctx.fillText(labels[i], lx, ly);
    }

    // Neutral ring indicator (value = 0 maps to 50% radius)
    ctx.beginPath();
    for (let i = 0; i <= count; i++) {
      const angle = (Math.PI * 2 * (i % count)) / count - Math.PI / 2;
      const x = cx + Math.cos(angle) * radius * 0.5;
      const y = cy + Math.sin(angle) * radius * 0.5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Value polygon
    ctx.beginPath();
    for (let i = 0; i <= count; i++) {
      const [x, y] = toCanvas(i % count, values[i % count] ?? 0, count, cx, cy, radius);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = ACCENT_FILL;
    ctx.fill();
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dots
    for (let i = 0; i < count; i++) {
      const [x, y] = toCanvas(i, values[i] ?? 0, count, cx, cy, radius);
      ctx.beginPath();
      ctx.arc(x, y, draggingRef.current === i ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = draggingRef.current === i ? DOT_ACTIVE : DOT_COLOR;
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [size, count, labels, values, dpr, cx, cy, radius]);

  // Interaction helpers
  const getCanvasPos = useCallback(
    (e: React.MouseEvent | React.TouchEvent): [number, number] => {
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const clientX = 'touches' in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : (e as React.MouseEvent).clientY;
      return [
        ((clientX - rect.left) / rect.width) * size,
        ((clientY - rect.top) / rect.height) * size,
      ];
    },
    [size],
  );

  const findNearest = useCallback(
    (mx: number, my: number): number | null => {
      let best = -1;
      let bestDist = 20; // pixel threshold
      for (let i = 0; i < count; i++) {
        const [x, y] = toCanvas(i, values[i] ?? 0, count, cx, cy, radius);
        const d = Math.hypot(mx - x, my - y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      return best >= 0 ? best : null;
    },
    [count, values, cx, cy, radius],
  );

  const projectOntoAxis = useCallback(
    (mx: number, my: number, axisIndex: number): number => {
      const angle = (Math.PI * 2 * axisIndex) / count - Math.PI / 2;
      const dx = mx - cx;
      const dy = my - cy;
      // Project onto axis direction
      const axDx = Math.cos(angle);
      const axDy = Math.sin(angle);
      const proj = dx * axDx + dy * axDy;
      // Normalize to [0, 1] relative to radius, then map to [-1, 1]
      const norm = Math.max(0, Math.min(1, proj / radius));
      return norm * 2 - 1;
    },
    [count, cx, cy, radius],
  );

  /** Find the axis index closest to the given angle from center */
  const findNearestAxisByAngle = useCallback(
    (mx: number, my: number): number => {
      const mouseAngle = normalizeAngle(Math.atan2(my - cy, mx - cx));
      let bestIdx = 0;
      let bestDelta = Infinity;
      for (let i = 0; i < count; i++) {
        const axisAngle = normalizeAngle((Math.PI * 2 * i) / count - Math.PI / 2);
        // Shortest angular distance
        let delta = Math.abs(mouseAngle - axisAngle);
        if (delta > Math.PI) delta = Math.PI * 2 - delta;
        if (delta < bestDelta) {
          bestDelta = delta;
          bestIdx = i;
        }
      }
      return bestIdx;
    },
    [count, cx, cy],
  );

  /** Compute value from radial distance: center=-1, middle ring=0, edge=1 */
  const valueFromRadialDistance = useCallback(
    (mx: number, my: number): number => {
      const dist = Math.hypot(mx - cx, my - cy);
      // Map dist from [0, radius] to [-1, 1]
      const norm = Math.max(0, Math.min(1, dist / radius));
      return norm * 2 - 1;
    },
    [cx, cy, radius],
  );

  const handlePointerDown = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const [mx, my] = getCanvasPos(e);
      const idx = findNearest(mx, my);
      if (idx !== null) {
        // Close to an existing dot: start precise single-axis drag
        draggingRef.current = idx;
        isDraggingSweep.current = false;
        e.preventDefault();
      } else {
        // Not near a dot: start sweep drag mode
        isDraggingSweep.current = true;
        draggingRef.current = null;
        e.preventDefault();
        // Immediately apply to nearest axis by angle
        const nearestAxis = findNearestAxisByAngle(mx, my);
        const val = valueFromRadialDistance(mx, my);
        onChange(nearestAxis, Math.round(val * 100) / 100);
      }
    },
    [getCanvasPos, findNearest, findNearestAxisByAngle, valueFromRadialDistance, onChange],
  );

  const handlePointerMove = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (draggingRef.current !== null) {
        // Single-axis drag mode
        e.preventDefault();
        const [mx, my] = getCanvasPos(e);
        const val = projectOntoAxis(mx, my, draggingRef.current);
        onChange(draggingRef.current, Math.round(val * 100) / 100);
      } else if (isDraggingSweep.current) {
        // Sweep drag mode: snap nearest axis by angle to radial distance
        e.preventDefault();
        const [mx, my] = getCanvasPos(e);
        const nearestAxis = findNearestAxisByAngle(mx, my);
        const val = valueFromRadialDistance(mx, my);
        onChange(nearestAxis, Math.round(val * 100) / 100);
      }
    },
    [getCanvasPos, projectOntoAxis, findNearestAxisByAngle, valueFromRadialDistance, onChange],
  );

  const handlePointerUp = useCallback(() => {
    draggingRef.current = null;
    isDraggingSweep.current = false;
  }, []);

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        style={{ width: size, height: size }}
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={handlePointerUp}
        onTouchStart={handlePointerDown}
        onTouchMove={handlePointerMove}
        onTouchEnd={handlePointerUp}
      />
    </div>
  );
}
