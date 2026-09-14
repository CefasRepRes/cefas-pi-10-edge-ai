#!/usr/bin/env python3
"""Interactive Plotly rendering for observed and Bayesian-corrected counts."""
from __future__ import annotations
import logging
from itertools import cycle
from pathlib import Path
from typing import Optional,Tuple
import numpy as np
import pandas as pd
try:
 import plotly.graph_objects as go
 import plotly.offline as pyo
 from plotly.colors import hex_to_rgb,qualitative
except Exception: go=pyo=hex_to_rgb=qualitative=None
LOG=logging.getLogger(__name__)
META={"timestamp","run_name","blob_name"}
def plot_timeseries(df:pd.DataFrame,output_html:Path,stacked:bool,log_y:bool,y_min:Optional[float]=None,y_max:Optional[float]=None,visible_classes:Optional[Tuple[str,...]]=None,uncertainty_df:Optional[pd.DataFrame]=None)->None:
    if go is None or pyo is None:
        output_html.parent.mkdir(parents=True,exist_ok=True); output_html.write_text("<html><body>Plotly is not installed.</body></html>",encoding="utf-8"); return
    classes=[c for c in df.columns if c not in META]; totals=df[classes].sum().sort_values(ascending=False)
    requested=tuple(c for c in (visible_classes or ()) if c in classes); defaults=set(classes) if visible_classes is None else set(requested)
    fig=go.Figure(); colours=cycle(list(getattr(qualitative,"Plotly",None) or ["#636EFA","#EF553B","#00CC96","#AB63FA"])); custom=df[["run_name","blob_name"]].astype(str).to_numpy(); vis=[]; main=[]
    for cls in totals.index:
        colour=next(colours); trace_visible=True if cls in defaults else "legendonly"; median=f"{cls}_corrected_median"; low=f"{cls}_corrected_lower"; high=f"{cls}_corrected_upper"
        if uncertainty_df is not None and all(c in uncertainty_df.columns for c in (median,low,high)):
            lo=np.nan_to_num(uncertainty_df[low].to_numpy(float)); hi=np.nan_to_num(uncertainty_df[high].to_numpy(float)); rgb=hex_to_rgb(colour) if hex_to_rgb else (99,110,250); band_visible=cls in defaults
            fig.add_trace(go.Scatter(x=df.timestamp,y=hi,mode="lines",visible=band_visible,line={"color":colour,"width":0},hoverinfo="skip",showlegend=False,legendgroup=cls)); vis.append(band_visible); main.append(False)
            fig.add_trace(go.Scatter(x=df.timestamp,y=lo,mode="lines",visible=band_visible,line={"color":colour,"width":0},fill="tonexty",fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)",hoverinfo="skip",showlegend=False,legendgroup=cls)); vis.append(band_visible); main.append(False)
            fig.add_trace(go.Scatter(x=df.timestamp,y=df[cls],mode="lines",name=f"{cls} observed prediction",legendgroup=cls,visible=trace_visible,line={"color":colour,"width":1,"dash":"dot"},opacity=.6,customdata=custom,hovertemplate="<b>%{fullData.name}</b><br>Timestamp=%{x}<br>Predicted count=%{y}<br>Run=%{customdata[0]}<br>Blob=%{customdata[1]}<extra></extra>")); vis.append(trace_visible); main.append(False)
            y=uncertainty_df[median]
        else:y=df[cls]
        fig.add_trace(go.Scatter(x=df.timestamp,y=y,customdata=custom,mode="lines+markers",name=cls,legendgroup=cls,stackgroup="one" if stacked else None,visible=trace_visible,line={"color":colour,"width":1.4},hovertemplate="<b>%{fullData.name}</b><br>Timestamp=%{x}<br>Count=%{y}<br>Run=%{customdata[0]}<br>Blob=%{customdata[1]}<extra></extra>")); vis.append(trace_visible); main.append(True)
    if y_min is not None and y_max is not None and y_min>=y_max: raise ValueError("--y-min must be less than --y-max")
    if log_y and ((y_min is not None and y_min<=0) or (y_max is not None and y_max<=0)): raise ValueError("Log-axis limits must be positive")
    yr=None if y_min is None and y_max is None else [np.log10(y_min) if log_y and y_min is not None else y_min,np.log10(y_max) if log_y and y_max is not None else y_max]
    fig.update_layout(title="Class counts through time across all runs",xaxis_title="Timestamp",yaxis_title="Count, log scale" if log_y else "Count",hovermode="x unified",height=850,margin={"l":80,"r":260,"t":80,"b":80},legend={"title":"Class","x":1.02,"groupclick":"togglegroup"})
    fig.update_xaxes(rangeslider_visible=True); fig.update_yaxes(type="log" if log_y else "linear",range=yr,autorange=yr is None)
    fig.update_layout(updatemenus=[{"type":"buttons","direction":"right","x":0,"y":1.08,"showactive":False,"buttons":[{"label":"Show selected","method":"update","args":[{"visible":vis}]},{"label":"Show all","method":"update","args":[{"visible":[True]*len(vis)}]},{"label":"Hide all","method":"update","args":[{"visible":[False if not m else "legendonly" for m in main]}]}]}])
    pyo.plot(fig,filename=str(output_html),auto_open=False,include_plotlyjs="cdn",config={"responsive":True,"scrollZoom":True,"displaylogo":False})
