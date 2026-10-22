/* Types of the Stage 1 export bundle (schema_version 1; docs/DASHBOARD_EXPORT.md section 3).
   Every numeric field that the source may leave undefined (NaN / Infinity in the original record) is typed
   `Num = number | null`: null means UNDEFINED and is never treated as 0. */

export type Num = number | null;

export interface FileEntry { size: number; sha256: string }
export interface SourceEntry { path: string; size: number; sha256: string }

export interface Manifest {
  schema_version: number;
  exporter_version: string;
  status: "PASSED" | "FAILED" | string;
  created_utc: string;
  seconds: number;
  experiment: string;
  integrity_notice: string;
  array_encoding: { name: string; steps: string[]; lossless: boolean; derived_not_stored: string };
  official_commit: string | null;
  sources: Record<string, SourceEntry>;
  json_normalization?: { rule: string; files: number; total_replaced: number };
  checks_summary: Record<CheckStatus, number>;
  files: Record<string, FileEntry>;
}

export type CheckStatus = "PASS" | "WARN" | "INFO" | "FAIL";
export interface Check { id: string; status: CheckStatus; message: string; details?: Record<string, unknown> }

export interface IntegrityReport {
  status: "PASSED" | "FAILED" | string;
  summary: Record<CheckStatus, number>;
  checks: Check[];
  fingerprint: { unchanged: boolean; n_files: number; bytes: number; added: string[]; removed: string[];
                 modified: string[]; mtime_only_changes: string[] };
  bundle_files: number;
  bundle_bytes_excluding_reports?: number;
}

export interface NormalizationRecord {
  source: string; source_sha256: string; source_bytes: number; n_replaced: number;
  by_kind: { nan: number; inf: number; "-inf": number }; paths: string[];
}
export interface NormalizationIndex { note: string; files: Record<string, NormalizationRecord>; total_replaced: number }

export interface UndefinedPath { path: string; kind: "nan" | "inf" | "-inf" }

export interface PlanRun { name: string; arm: "O" | "U"; seed: number; retest: boolean; cell_head_train_clamp: boolean;
                           epochs: number; steps: number }
export interface Thresholds {
  activity_std_min: number; activity_zero_max: number; p1_last_steps: number; p1_max_dead_frac: number;
  p2_last_steps: number; p2_min_improvement?: number; p2_borderline_low: number; p2_borderline_high: number;
  p3_min_cell_spearman: number; hsig_max_abs_wmse_change: number; hsig_max_cell_spearman: number;
  sign_test_alpha: number; fresh_official_atol: number; fresh_official_rtol: number;
}
export interface Plan {
  train: string[]; val: string[]; val_main: string[]; holdout: Record<string, string[]>; all_train: string[];
  epochs: number; steps_per_run: number; eval_every_steps: number; eval_steps: number[];
  thresholds: Thresholds; statistical_test: Record<string, string>; categories: string[];
  replication_step: number; runs: PlanRun[]; c_max: number;
}

export interface RunResult {
  name: string; arm: "O" | "U"; seed: number; retest: boolean; cell_head_train_clamp: boolean;
  valid: boolean; invalid_reasons: string[]; category: string; hsig_criteria_met: boolean;
  run_status: string;
  P1: { pass: boolean; final_cell_activity: string; final_cell_val_std: Num; final_cell_val_zero_frac: Num;
        dead_frac_last_steps: Num };
  P2: { outcome: string; cell_wmse_clamped_improvement_vs_zero: Num; cell_wmse_raw_improvement_vs_zero: Num;
        cell_wmse_loss_improvement_vs_zero: Num };
  P3: { pass: boolean; final_val_cell_spearman_indep: Num; final_val_cell_spearman_official: Num };
  grid: { final_activity: string; wmse_improvement_vs_zero: Num; final_val_spearman_indep: Num;
          final_val_spearman_official: Num };
}

export interface PrimaryPair { seed: number; O: string; U: string; discordant: boolean }
export interface PrimaryTest {
  name: string; endpoint: string; pairs: PrimaryPair[]; excluded: { seed: number; reason: string }[];
  n_valid_pairs: number; n_discordant: number; U_only: number; O_only: number; direction: string;
  p_two_sided_exact: number; alpha: number; significant: boolean;
}
export interface Analysis {
  experiment_status: string; status_reasons: string[]; historical_replication: string; verdict: string;
  run_statuses: Record<string, string>; category_counts: Record<string, Record<string, number>>;
  primary_test: PrimaryTest; hsig: Record<string, { n_seeds_meeting_hsig: number; seeds: number[]; n_valid_runs: number }>;
  conclusion: string; deviation_notice: string;
}
export interface Report {
  plan: Plan; results: RunResult[]; run_status_counts: Record<string, number>; analysis: Analysis;
  protected_artifacts?: { unchanged: boolean | null; n_files?: number };
  provenance: string;
}
export interface ReportFile { report: Report; undefined_paths: UndefinedPath[]; note: string }

export interface AnalysisFile {
  primary_test_rederived: PrimaryTest & { minimum_p_achievable_with_these_pairs: Num; excluded_seeds: number[] };
  threshold_note: { recorded_field: string; recorded_value: Num; implemented_rule: string;
                    recorded_value_used_by_code: boolean;
                    runs_in_disputed_interval: { name: string; improvement: number; implemented_outcome: string;
                                                 single_threshold_reading: string }[]; note: string };
  exploratory: { label: string; metrics: Record<string, unknown> };
  undefined_paths: UndefinedPath[];
}

export interface ArrayMeta {
  dtype: "float32" | "int32"; shape: number[]; encoding: string; itemsize: number; sha256_raw: string;
  nbytes_raw: number; nbytes_file: number; file: string;
}
export interface RunIndexEntry {
  name: string; arm: "O" | "U"; seed: number; retest: boolean; cell_head_train_clamp: boolean;
  primary_inference: boolean; maps: Record<string, Record<string, ArrayMeta>>;
}
export interface MapsIndexEntry {
  design: string; role: string; shape: [number, number]; n_cells_subsample: number; arrays: Record<string, ArrayMeta>;
}

/** Written by tools/run_dashboard.py (not part of the bundle): what the launcher verified before serving. */
export interface Session {
  app: string; zip_name: string; zip_path: string; zip_sha256: string; zip_bytes: number;
  verified: boolean; verification: { status: string; errors: string[]; files_checked: number; arrays_checked: number;
                                     json_strict: number };
  extract_dir: string; launched_utc: string; launcher_version: string;
}

/* ---- notebook 07 (context07/): a SEPARATE, earlier experiment. Never pooled with HEADPATH. ---- */
export interface C07PlanRun { name: string; arm: string; seed: number; precision: string; epochs: number; steps: number;
                              train: string[]; val: string[] }
export interface C07Plan {
  c_max: number; train_designs: string[]; val_designs: string[]; test_designs: string[]; val: string[]; val_main: string[];
  arms: Record<string, string[]>; single_design: string; budget_steps: number; common_steps: number; runs: C07PlanRun[];
}
export interface C07Run {
  name: string; arm: string; seed: number; precision: string; epochs: number; steps: number; n_train: number; n_val: number;
  steps_logged: number; zero_grad_steps: number; collapse_step: Num; zero_tail_len: number; collapsed: boolean;
  at_common_steps: { common_steps: number; collapsed: boolean; collapse_step: Num; zero_tail_len: number };
  max_grad_norm: Num; max_grad_step: Num; max_grad_sample: string | null; min_cell_raw_le0: Num; max_grid_raw_le0: Num;
  steps_both_heads_ge_0999: Num; scaler_skipped_steps: Num; final_val_avg_spearman: Num; best_val_avg_spearman: Num;
  final_val_pred_zero_frac: { cell: Num; grid: Num };
}
export interface C07ReportFile {
  report: { plan: C07Plan; results: C07Run[]; by_arm: Record<string, unknown>; provenance: string };
  undefined_paths: UndefinedPath[]; report_file: string; field_caveats: Record<string, string>;
}
export interface C07RunIndexEntry { name: string; arm: string; seed: number; precision: string; steps: boolean; history: boolean }
