export interface VoiceInfo {
  filename: string;
  name: string;
}

export interface ComponentInfo {
  index: number;
  name: string;
}

export interface CatalogResponse {
  components: ComponentInfo[];
  count: number;
  pcaCount: number;
}

export interface VoicesResponse {
  voices: VoiceInfo[];
}

export interface SynthesizeRequest {
  voice: string;
  coefficients: number[];
  text: string;
  speed?: number;
}

export interface ExportVoiceRequest {
  voice: string;
  coefficients: number[];
}
