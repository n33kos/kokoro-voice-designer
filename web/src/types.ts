export interface VoiceInfo {
  filename: string;
  name: string;
}

export interface ComponentInfo {
  index: number;
  name: string;
}

export interface SemanticFeature {
  name: string;
  index: number;
}

/** Style map v2: axes in Kokoro's native 256-dim style space. */
export interface StyleFeature {
  name: string;
  index: number;
}

export interface CatalogResponse {
  components: ComponentInfo[];
  count: number;
  pcaCount: number;
  hasSemanticMap?: boolean;
  semanticFeatures?: SemanticFeature[];
  semanticDirections?: number[][];
  hasStyleMap?: boolean;
  /** Directions stay server-side; the client only sends slider values. */
  styleFeatures?: StyleFeature[];
}

export interface VoicesResponse {
  voices: VoiceInfo[];
}

export interface SynthesizeRequest {
  voice: string;
  coefficients: number[];
  text: string;
  speed?: number;
  /** When set, the server applies these in style space and ignores `coefficients`. */
  styleCoefficients?: number[];
}

export interface ExportVoiceRequest {
  voice: string;
  coefficients: number[];
}
