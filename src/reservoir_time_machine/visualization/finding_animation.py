"""Endpoint-only presentation automatically framed on a validated finding member."""
import numpy as np


def finding_animation(baseline,monitor,missing,result,finding):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    i,x,s=map(int,finding['representative']);i0,i1,x0,x1=result['bounds']
    if finding not in result['findings'] or result['labels'][i-i0,x-x0]!=finding['label']:
        raise ValueError('Animation focus must be a member of the selected finding')
    lo=max(0,s-24);hi=min(baseline.shape[2],s+25)
    # Views are bounded source crops, with exact source-index coordinates.
    planes=[(baseline[i0:i1,x0:x1,s],monitor[i0:i1,x0:x1,s],missing[i0:i1,x0:x1],
             np.arange(x0,x1),np.arange(i0,i1),x,i,'Sample plane','Crossline index','Inline index'),
            (baseline[i,x0:x1,lo:hi].T,monitor[i,x0:x1,lo:hi].T,
             np.broadcast_to(missing[i,x0:x1],(hi-lo,x1-x0)),np.arange(x0,x1),np.arange(lo,hi),x,s,'Selected inline','Crossline index','Sample index'),
            (baseline[i0:i1,x,lo:hi].T,monitor[i0:i1,x,lo:hi].T,
             np.broadcast_to(missing[i0:i1,x],(hi-lo,i1-i0)),np.arange(i0,i1),np.arange(lo,hi),i,s,'Selected crossline','Inline index','Sample index')]
    fig=make_subplots(rows=2,cols=3,subplot_titles=[p[7]+' · endpoint comparison' for p in planes]+[p[7]+' · Monitor − Baseline' for p in planes],vertical_spacing=.16)
    frame_traces=[];endpoints=[[],[]]
    for col,(b,m,mask,xs,ys,fx,fy,title,xt,yt) in enumerate(planes,1):
        valid=np.isfinite(b)&np.isfinite(m)&~mask
        b=np.where(valid,b,np.nan);m=np.where(valid,m,np.nan);d=m-b
        finite=np.concatenate((np.abs(b[np.isfinite(b)]),np.abs(m[np.isfinite(m)])))
        limit=float(np.percentile(finite,99)) if finite.size else 1.;limit=max(limit,1e-12)
        df=np.abs(d[np.isfinite(d)]);dl=max(float(np.percentile(df,99)) if df.size else 1.,1e-12)
        frame_traces.append(len(fig.data))
        for idx,z in enumerate((b,m)):
            endpoints[idx].append(go.Heatmap(z=z,x=xs,y=ys,zmin=-limit,zmax=limit,colorscale='RdBu',showscale=False))
        fig.add_trace(endpoints[0][-1],row=1,col=col)
        fig.add_trace(go.Heatmap(z=d,x=xs,y=ys,zmin=-dl,zmax=dl,colorscale='RdBu',colorbar={'title':'Δ amplitude','len':.32,'x':col/3-.01,'y':.17}),row=2,col=col)
        for row in (1,2):
            fig.add_trace(go.Scatter(x=[fx],y=[fy],mode='markers',marker={'symbol':'x','size':13,'color':'#ffb000'},showlegend=False,name='Finding representative'),row=row,col=col)
            fig.add_trace(go.Heatmap(z=np.where(~valid,1,np.nan),x=xs,y=ys,colorscale=[[0,'#b8b8b8'],[1,'#b8b8b8']],showscale=False,hoverinfo='skip'),row=row,col=col)
            fig.update_xaxes(title_text=xt,row=row,col=col);fig.update_yaxes(title_text=yt,row=row,col=col)
    membership=result['labels']==finding['label']
    # Membership is a distinct boolean layer; no region IDs are averaged or merged.
    fig.add_trace(go.Heatmap(z=np.where(membership,1,np.nan),x=planes[0][3],y=planes[0][4],
        colorscale=[[0,'#ffab00'],[1,'#ffab00']],showscale=False,opacity=.25,name='Selected component'),row=2,col=1)
    fig.frames=[go.Frame(name=name,data=data,traces=frame_traces,
        layout=go.Layout(title_text=f'{name} · {finding["id"]} · inline {i}, crossline {x}, sample {s}'))
        for name,data in zip(('Baseline','Monitor'),endpoints)]
    fig.update_layout(template="plotly_dark",paper_bgcolor="#181619",plot_bgcolor="#211f22",font={"size":15},height=850,title=f'Baseline · {finding["id"]} · inline {i}, crossline {x}, sample {s}',
        updatemenus=[{'type':'buttons','bgcolor':'#203b34','bordercolor':'#31edc2','font':{'color':'white'},'showactive':False,'buttons':[{'label':'Compare endpoints','method':'animate','args':[None,{'frame':{'duration':1000,'redraw':True},'transition':{'duration':0},'fromcurrent':False}]},
            {'label':'Pause','method':'animate','args':[[None],{'mode':'immediate','frame':{'duration':0}}]}]}],
        sliders=[{'steps':[{'label':n,'method':'animate','args':[[n],{'mode':'immediate','frame':{'duration':0,'redraw':True},'transition':{'duration':0}}]} for n in ('Baseline','Monitor')]}])
    return fig
