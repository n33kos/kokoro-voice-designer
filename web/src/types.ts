export interface VoiceInfo {
  filename: string;
  name: string;
}

/** One axis of the style map: a named, calibrated direction in Kokoro's 256-dim style space. */
export interface StyleFeature {
  name: string;
  index: number;
}

export interface CatalogResponse {
  hasStyleMap?: boolean;
  /** Directions stay server-side; the client only sends slider values. */
  styleFeatures?: StyleFeature[];
}

export interface VoicesResponse {
  voices: VoiceInfo[];
}

export interface SynthesizeRequest {
  voice: string;
  /** Kept for the raw component API; the UI sends styleCoefficients instead. */
  coefficients: number[];
  text: string;
  speed?: number;
  /** Slider values, one per style feature. */
  styleCoefficients?: number[];
}

export interface ExportVoiceRequest {
  voice: string;
  coefficients: number[];
  styleCoefficients?: number[];
}
