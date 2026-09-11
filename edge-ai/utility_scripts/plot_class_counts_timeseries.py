#!/usr/bin/env python3
"""CLI orchestration for loading, correction, exports, and plots."""
from __future__ import annotations
import argparse, logging, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR.parent))
from bayesian_spence import derive_dirichlet_posterior_parameters,    extract_validation_error_model,    load_validation_uncertainty_dataframe
from ml_prediction_results_utilities import build_dataframe, get_blob_service_client, load_inference_rows_from_source, load_json_payload, parse_azure_blob_url, safe_part, write_long_counts_csv, write_long_uncertainty_csv
from plotting_predictions import plot_timeseries

LOG=logging.getLogger(__name__)
DEFAULT_INFERENCE_SOURCE="https://citprodc8603uksa.blob.core.windows.net/ml-prediction-results/runs/"
DEFAULT_VALIDATION_SOURCE="https://citprodc8603uksa.blob.core.windows.net/training-libs/validationsession_202608270702/validation.json"

@dataclass(frozen=True)
class Config:
    outdir:Path; stacked:bool; log_y:bool; y_min:Optional[float]; y_max:Optional[float]
    visible_classes:Optional[Tuple[str,...]]; inference_json:Optional[str]
    validation_json:Optional[str]; mc_samples:int; mc_seed:int; target_class:str; diagnostics:bool; diagnostic_prior_draws:int

def plot_validation_confusion_matrix(validation_source: str, output_html: Path, *, blob_service_client=None) -> None:
    """Create an interactive heatmap of posterior mean P(predicted | true)."""
    import numpy as np
    import plotly.graph_objects as go
    import plotly.offline as pyo
    payload=load_json_payload(validation_source,blob_service_client=blob_service_client)
    model=extract_validation_error_model(payload)
    if model is None: raise ValueError("Validation JSON has no classifier error model")
    classes=[str(x) for x in model.get("class_order",[]) if x is not None and str(x)]
    alpha=np.asarray(derive_dirichlet_posterior_parameters(model,classes),dtype=float)
    probability=alpha/alpha.sum(axis=1,keepdims=True)
    labels=np.vectorize(lambda x:f"{x:.3f}")(probability)
    fig=go.Figure(go.Heatmap(z=probability,x=classes,y=classes,customdata=alpha,
        colorscale="Blues",zmin=0,zmax=1,text=labels,texttemplate="%{text}",
        colorbar={"title":"Posterior mean<br>probability"},
        hovertemplate="True class=%{y}<br>Predicted class=%{x}<br>Posterior mean=%{z:.4f}<br>Dirichlet alpha=%{customdata:.2f}<extra></extra>"))
    fig.update_layout(title="Classifier confusion matrix: posterior mean P(predicted | true)",
        xaxis_title="Predicted class",yaxis_title="True class",
        yaxis={"autorange":"reversed","scaleanchor":"x","scaleratio":1},
        height=max(650,45*len(classes)+220),margin={"l":180,"r":100,"t":100,"b":180})
    fig.update_xaxes(tickangle=-45)
    pyo.plot(fig,filename=str(output_html),auto_open=False,include_plotlyjs="cdn",
             config={"responsive":True,"displaylogo":False})

def parse_args(argv:Optional[List[str]]=None)->Config:
    p=argparse.ArgumentParser(description="Plot class-count time series with optional Bayesian uncertainty envelopes.")
    p.add_argument("--outdir",default="summary_timeseries_out"); p.add_argument("--stacked",action="store_true")
    p.add_argument("--log-y",action="store_true"); p.add_argument("--y-min",type=float); p.add_argument("--y-max",type=float)
    p.add_argument("--visible-classes"); p.add_argument("--inference-json",default=DEFAULT_INFERENCE_SOURCE); p.add_argument("--input-json")
    p.add_argument("--validation-json",default=DEFAULT_VALIDATION_SOURCE); p.add_argument("--mc-samples",type=int,default=250)
    p.add_argument("--mc-seed",type=int,default=42); p.add_argument("--target-class",default="fish_larvae"); p.add_argument("--no-bayesian-diagnostics",action="store_true"); p.add_argument("--diagnostic-prior-draws",type=int,default=500); p.add_argument("--verbose",action="store_true")
    a=p.parse_args(argv); logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    visible=tuple(x.strip() for x in a.visible_classes.split(",") if x.strip()) if a.visible_classes else None
    return Config(Path(a.outdir),a.stacked,a.log_y,a.y_min,a.y_max,visible,a.input_json or a.inference_json,a.validation_json,max(1,a.mc_samples),a.mc_seed,a.target_class,not a.no_bayesian_diagnostics,max(20,a.diagnostic_prior_draws))

def run_bayesian_timeseries_analysis(
    inference_source: Optional[str],
    validation_source: Optional[str],
    output_dir: Path | str,
    *,
    stacked: bool = False,
    log_y: bool = False,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    visible_classes: Optional[Tuple[str, ...]] = None,
    mc_samples: int = 250,
    mc_seed: int = 42,
    target_class: str = "fish_larvae",
    diagnostics: bool = True,
    diagnostic_prior_draws: int = 500,
    blob_service_client=None,
) -> Dict[str, Optional[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not inference_source:
        raise RuntimeError("Provide an inference source")
    rows = load_inference_rows_from_source(inference_source, log_cb=LOG.warning)
    if not rows:
        raise RuntimeError(f"Inference source yielded no class counts: {inference_source}")
    plot_df, _ = build_dataframe(rows)
    counts_csv = output_dir / "class_counts_timeseries.csv"
    long_csv = output_dir / "class_counts_timeseries_long.csv"
    timeseries_html = output_dir / "class_counts_timeseries.html"
    plot_df.to_csv(counts_csv, index=False)
    write_long_counts_csv(plot_df, long_csv)
    client = blob_service_client
    if validation_source and client is None:
        blob = parse_azure_blob_url(validation_source) if validation_source else None
        if blob:
            client = get_blob_service_client(f"https://{blob[0]}", log_cb=LOG.warning)
    confusion_html = None
    if validation_source:
        try:
            confusion_html = output_dir / "confusion_matrix_posterior_mean.html"
            plot_validation_confusion_matrix(validation_source, confusion_html, blob_service_client=client)
        except Exception as exc:
            confusion_html = None
            LOG.warning("Could not plot confusion matrix: %s", exc)
    diagnostics_dir = output_dir / "bayesian_diagnostics" if diagnostics else None
    uncertainty = load_validation_uncertainty_dataframe(
        plot_df,
        validation_source,
        mc_samples=mc_samples,
        mc_seed=mc_seed,
        blob_service_client=client,
        diagnostics_dir=diagnostics_dir,
        diagnostic_prior_draws=diagnostic_prior_draws,
        timeseries_feedback=False,
    )
    corrected = corrected_long = None
    if uncertainty is not None:
        corrected = output_dir / "class_counts_timeseries_corrected.csv"
        corrected_long = output_dir / "class_counts_timeseries_corrected_long.csv"
        uncertainty.to_csv(corrected, index=False)
        write_long_uncertainty_csv(uncertainty, corrected_long)
    plot_timeseries(
        plot_df,
        timeseries_html,
        stacked,
        log_y,
        y_min,
        y_max,
        visible_classes,
        uncertainty,
    )
    target_dir = output_dir / safe_part(target_class) if target_class else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
    return {
        "counts_csv": counts_csv,
        "long_csv": long_csv,
        "timeseries_html": timeseries_html,
        "confusion_html": confusion_html,
        "corrected_csv": corrected,
        "corrected_long_csv": corrected_long,
        "diagnostics_dir": diagnostics_dir,
        "target_dir": target_dir,
    }


def main(argv:Optional[List[str]]=None)->int:
    cfg=parse_args(argv)
    outputs = run_bayesian_timeseries_analysis(
        inference_source=cfg.inference_json,
        validation_source=cfg.validation_json,
        output_dir=cfg.outdir,
        stacked=cfg.stacked,
        log_y=cfg.log_y,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
        visible_classes=cfg.visible_classes,
        mc_samples=cfg.mc_samples,
        mc_seed=cfg.mc_seed,
        target_class=cfg.target_class,
        diagnostics=cfg.diagnostics,
        diagnostic_prior_draws=cfg.diagnostic_prior_draws,
    )
    for path in (
        outputs["counts_csv"],
        outputs["long_csv"],
        outputs["timeseries_html"],
        outputs["confusion_html"],
        outputs["corrected_csv"],
        outputs["corrected_long_csv"],
    ):
        if path: print(f"Wrote: {path}")
    if outputs["target_dir"] is not None:
        print(f"Prepared target-class output directory: {outputs['target_dir']}")
    return 0

if __name__=="__main__": raise SystemExit(main())