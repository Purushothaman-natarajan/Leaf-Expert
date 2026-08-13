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
  databank_total?: number;
  databank_can_train?: boolean;
}

// ─── VLM Types ────────────────────────────────────────────────────────────────

export interface LeafScanResult {
  is_plant: boolean;
  crop_type: string;
  is_diseased: boolean;
  disease_name: string;
  scientific_name?: string;
  confidence: number;
  severity: 'none' | 'low' | 'moderate' | 'high' | 'critical';
  affected_area_percent: number;
  symptoms: string[];
  treatment: string[];
  prevention: string[];
  explanation: string;
  urgency: 'none' | 'low' | 'medium' | 'high';
  is_safe_to_consume?: boolean | null;
}

export interface ScanResponse {
  scan_id: string;
  provider_used: string;
  model_used: string;
  result: LeafScanResult;
  processing_time_ms: number;
}

export interface ProviderInfo {
  id: string;
  name: string;
  default_model: string;
  cost_tier: 'free' | 'low' | 'medium' | 'high';
  requires_api_key: boolean;
  description: string;
}

export interface DataPointRecord {
  id: string;
  image_url: string;
  vlm_provider: string;
  vlm_model: string;
  disease_name: string;
  user_label: string;
  confirmed: boolean;
  confidence: number;
  severity: string;
  notes?: string;
  collected_at: string;
  used_for_training: boolean;
}

export interface DataBankStats {
  total_points: number;
  confirmed_points: number;
  class_distribution: Record<string, number>;
  training_threshold: number;
  classes_ready: string[];
  classes_pending: Record<string, number>;
  can_train: boolean;
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

// ─── VLM Quick Scan ───────────────────────────────────────────────────────────

export async function scanLeaf(
  imageFile: File,
  provider: string,
  apiKey: string | null,
  modelName?: string,
  ollamaHost?: string,
): Promise<ScanResponse> {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('provider', provider);
  if (modelName) form.append('model_name', modelName);
  if (ollamaHost) form.append('ollama_host', ollamaHost);

  const headers: Record<string, string> = {};
  if (apiKey) headers['X-VLM-API-Key'] = apiKey;

  const { data } = await api.post<ScanResponse>('/vlm/scan', form, { headers });
  return data;
}

export async function getProviders(): Promise<ProviderInfo[]> {
  const { data } = await api.get<ProviderInfo[]>('/vlm/providers');
  return data;
}

// ─── DataStore (Data Flywheel) ────────────────────────────────────────────────

export async function saveDataPoint(payload: {
  scan_id: string;
  accepted_label: string;
  confirmed: boolean;
  notes?: string;
  vlm_provider: string;
  vlm_model: string;
  vlm_prediction: object;
}): Promise<{ status: string; id: string | null }> {
  const { data } = await api.post('/datastore/save_full', null, {
    params: {
      scan_id: payload.scan_id,
      accepted_label: payload.accepted_label,
      confirmed: payload.confirmed,
      notes: payload.notes,
      vlm_provider: payload.vlm_provider,
      vlm_model: payload.vlm_model,
    },
  });
  return data;
}

export async function listDataPoints(opts?: {
  label?: string;
  confirmed_only?: boolean;
  limit?: number;
  offset?: number;
}): Promise<DataPointRecord[]> {
  const { data } = await api.get<DataPointRecord[]>('/datastore/list', { params: opts });
  return data;
}

export async function getDataBankStats(): Promise<DataBankStats> {
  const { data } = await api.get<DataBankStats>('/datastore/stats');
  return data;
}

export async function deleteDataPoint(id: string): Promise<void> {
  await api.delete(`/datastore/${id}`);
}

export async function exportDataset(payload: {
  target_dir: string;
  confirmed_only?: boolean;
  val_ratio?: number;
  test_ratio?: number;
}): Promise<{
  status: string;
  target_dir: string;
  exported_counts: Record<string, number>;
  train_count: number;
  val_count: number;
  test_count: number;
  message: string;
}> {
  const { data } = await api.post('/datastore/export', payload);
  return data;
}
