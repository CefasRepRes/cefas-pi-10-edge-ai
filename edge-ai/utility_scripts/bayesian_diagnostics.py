#!/usr/bin/env python3
"""Interactive narrative diagnostics for bayesian_spence_no_timeseries_feedback."""
from pathlib import Path
import html, json
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

def _write(fig, path):
    fig.update_layout(template="plotly_white", margin=dict(l=70,r=40,t=90,b=70))
    pyo.plot(fig, filename=str(path), auto_open=False, include_plotlyjs="cdn", config={"responsive":True,"displaylogo":False})

def _flat(x):
    a=np.asarray(x); return a.reshape((-1,)+a.shape[2:])

def write_bayesian_diagnostics(*, output_dir, mcmc, prior_samples, posterior_by_chain,
                               observed_counts, validation_counts, classes,
                               expected_true_counts, labels, seed):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); pages=[]
    # Confusion learning
    post=_flat(posterior_by_chain['confusion_matrix']); prior=np.asarray(prior_samples['confusion_matrix'])
    empirical=validation_counts/validation_counts.sum(1,keepdims=True)
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Validation counts','Empirical probabilities','Prior mean','Posterior mean'])
    for k,z in enumerate([validation_counts,empirical,prior.mean(0),post.mean(0)]):
        fig.add_trace(go.Heatmap(z=z,x=classes,y=classes,colorscale='Blues',text=np.vectorize(lambda x:f'{x:.3g}')(z),texttemplate='%{text}'),row=k//2+1,col=k%2+1)
    fig.update_yaxes(autorange='reversed'); fig.update_layout(height=950,title='Confusion matrix: evidence, prior and posterior')
    _write(fig,out/'02_confusion.html'); pages.append(('02_confusion.html','Confusion-matrix learning'))
    # Individual chains
    items=[]
    for name in ('ecological_concentration','mu','k'):
        if name in posterior_by_chain: items.append((name,np.asarray(posterior_by_chain[name])))
    cm=np.asarray(posterior_by_chain['confusion_matrix']); items += [(f'P({classes[0]}|{classes[0]})',cm[:,:,0,0])]
    fig=make_subplots(rows=len(items),cols=1,subplot_titles=[x[0] for x in items])
    for r,(name,a) in enumerate(items,1):
        for c in range(a.shape[0]): fig.add_trace(go.Scattergl(y=a[c],mode='lines',name=f'chain {c+1}',showlegend=r==1),row=r,col=1)
    fig.update_layout(height=max(700,240*len(items)),title='Individual retained chains')
    _write(fig,out/'03_chains.html'); pages.append(('03_chains.html','Individual chains'))
    # Prior vs posterior
    pairs=[]
    for name in ('ecological_concentration','mu','k'):
        if name in prior_samples and name in posterior_by_chain: pairs.append((name,np.asarray(prior_samples[name]).ravel(),np.asarray(posterior_by_chain[name]).ravel()))
    fig=make_subplots(rows=len(pairs),cols=1,subplot_titles=[x[0] for x in pairs])
    for r,(name,a,b) in enumerate(pairs,1):
        fig.add_trace(go.Histogram(x=a,histnorm='probability density',opacity=.45,name='prior',showlegend=r==1),row=r,col=1)
        fig.add_trace(go.Histogram(x=b,histnorm='probability density',opacity=.55,name='posterior',showlegend=r==1),row=r,col=1)
    fig.update_layout(barmode='overlay',height=max(650,250*len(pairs)),title='Prior to posterior')
    _write(fig,out/'04_prior_posterior.html'); pages.append(('04_prior_posterior.html','Prior versus posterior'))
    # Individual examples including largest correction
    expected=np.asarray(expected_true_counts); shift=np.abs(expected.mean(0)-observed_counts).sum(1)
    ids=list(dict.fromkeys([0,int(np.argsort(observed_counts.sum(1))[len(observed_counts)//2]),int(np.argmax(shift))]))
    fig=make_subplots(rows=len(ids),cols=2,subplot_titles=sum(([f'Observed vs corrected: {html.escape(labels[i])}',f'Posterior counts: {html.escape(labels[i])}'] for i in ids),[]))
    for r,i in enumerate(ids,1):
        fig.add_trace(go.Bar(x=classes,y=observed_counts[i],name='observed',showlegend=r==1),row=r,col=1)
        fig.add_trace(go.Bar(x=classes,y=expected[:,i,:].mean(0),name='corrected mean',showlegend=r==1),row=r,col=1)
        for j,cls in enumerate(classes): fig.add_trace(go.Violin(y=expected[:,i,j],name=cls,box_visible=True,meanline_visible=True,showlegend=r==1),row=r,col=2)
    fig.update_layout(barmode='group',height=max(750,480*len(ids)),title='Individual examples')
    _write(fig,out/'05_individual_examples.html'); pages.append(('05_individual_examples.html','Individual worked examples'))
    # Exact expected allocation for largest-correction example
    i=ids[-1]; p=post.mean(0); q=_flat(posterior_by_chain['prevalence']).mean(0)[i]; rp=q@p
    alloc=q[:,None]*p/np.where(rp[None,:]>0,rp[None,:],1); flow=alloc*observed_counts[i][None,:]; n=len(classes)
    source=[];target=[];value=[]
    for ti in range(n):
        for pj in range(n):
            if flow[ti,pj]>1e-9: source.append(n+pj);target.append(ti);value.append(float(flow[ti,pj]))
    fig=go.Figure(go.Sankey(node=dict(label=[f'True: {x}' for x in classes]+[f'Predicted: {x}' for x in classes]),link=dict(source=source,target=target,value=value)))
    fig.update_layout(title=f'Individual Bayes allocation: {html.escape(labels[i])}',height=850)
    _write(fig,out/'06_allocation.html'); pages.append(('06_allocation.html','Predicted-to-true allocation'))
    extra=mcmc.get_extra_fields(group_by_chain=True); summary={'divergences':int(np.asarray(extra.get('diverging',[])).sum())}
    (out/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    return out
