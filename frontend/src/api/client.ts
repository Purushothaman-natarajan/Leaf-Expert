/**
 * Leaf-Expert — Axios API Client
 * Points to FastAPI backend at VITE_API_URL (default: localhost:8000)
 */
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 minutes for LIME which can be slow
});

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PredictionResponse {
  label: string;
  confidence: number;
  all_class_probs: Record<string, number>;
}

export interface ExplainResponse extends PredictionResponse {
  gradcam_b64: string;
  lime_b64: string;
  gradcam_available: boolean;
  lime_available: boolean;
}

export interface DataPrepResponse {
  status: string;
  train_count: number;
  val_count: number;
  test_count: number;
  classes: string[];
  target_folder: string;
}

export interface TrainStartResponse {
  job_id: string;
  message: string;
  backbones: string[];
}

export interface TrainStatusResponse {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'not_found';
  current_epoch: number | null;
  total_epochs: number | null;
  train_loss: number | null;
  train_acc: number | null;
  val_loss: number | null;
  val_acc: number | null;
  best_val_acc: number | null;
  message: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  torch_version: string;
  cuda_available: boolean;
  device: string;
}

// ─── API Functions ────────────────────────────────────────────────────────────

export async function explainImage(
  imageFile: File,
  modelPath: string,
  numLimeSamples = 100,
  numLimeFeatures = 30,
  segmentationAlg = 'quickshift',
): Promise<ExplainResponse> {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('model_path', modelPath);
  form.append('num_lime_samples', String(numLimeSamples));
  form.append('num_lime_features', String(numLimeFeatures));
  form.append('segmentation_alg', segmentationAlg);
  const { data } = await api.post<ExplainResponse>('/predict/explain', form);
  return data;
}

export async function predictImage(
  imageFile: File,
  modelPath: string,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('model_path', modelPath);
  const { data } = await api.post<PredictionResponse>('/predict/', form);
  return data;
}

export async function prepareData(payload: {
  raw_dataset_path: string;
  target_folder: string;
  image_size?: number;
  augment?: boolean;
  train_ratio?: number;
  val_ratio?: number;
}): Promise<DataPrepResponse> {
  const { data } = await api.post<DataPrepResponse>('/data/prepare', payload);
  return data;
}

export async function startTraining(payload: {
  data_path: string;
  model_dir: string;
  log_dir: string;
  backbones: string[];
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  image_size: number;
  patience: number;
}): Promise<TrainStartResponse> {
  const { data } = await api.post<TrainStartResponse>('/train/start', payload);
  return data;
}

export async function getTrainStatus(jobId: string): Promise<TrainStatusResponse> {
  const { data } = await api.get<TrainStatusResponse>(`/train/status/${jobId}`);
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}
