"""Plotly rendering for linked, display-only orthogonal seismic sections."""

import numpy as np

from .slices import display_mask, display_slice


def volume_figure(volume: np.ndarray, title: str, *, inline: int | None = None,
                  crossline: int | None = None, sample: int | None = None,
                  missing_trace_mask: np.ndarray | None = None,
                  max_axis: int = 160, colorscale: str = "RdBu",
                  value_range: tuple[float, float] | None = None, finding_context=None):
    """Render three linked section planes; all coordinates remain sample indices."""
    import plotly.graph_objects as go

    if volume.ndim != 3:
        raise ValueError("volume must use (inline, crossline, sample) axes")
    ni, nc, ns = volume.shape
    inline = ni // 2 if inline is None else int(inline)
    crossline = nc // 2 if crossline is None else int(crossline)
    sample = ns // 2 if sample is None else int(sample)
    if not (0 <= inline < ni and 0 <= crossline < nc and 0 <= sample < ns):
        raise IndexError("selected section lies outside the volume")
    if missing_trace_mask is not None and np.shape(missing_trace_mask) != (ni, nc):
        raise ValueError("missing trace mask must match inline/crossline grid")
    # Each plane is strided only for browser rendering. Coordinates retain original
    # grid indices, so the intersection remains aligned with the source volume.
    inline_plane = volume[inline, :, :]
    crossline_plane = volume[:, crossline, :]
    sample_plane = volume[:, :, sample]
    planes = (display_slice(inline_plane.T, max_axis), display_slice(crossline_plane.T, max_axis),
              display_slice(sample_plane.T, max_axis))
    strides = ((max(1, int(np.ceil(ns / max_axis))), max(1, int(np.ceil(nc / max_axis)))),
               (max(1, int(np.ceil(ns / max_axis))), max(1, int(np.ceil(ni / max_axis)))),
               (max(1, int(np.ceil(nc / max_axis))), max(1, int(np.ceil(ni / max_axis)))))
    if missing_trace_mask is not None:
        missing = np.asarray(missing_trace_mask, dtype=bool)
        plane_masks = (
            np.broadcast_to(missing[inline, :][None, :], (ns, nc)),
            np.broadcast_to(missing[:, crossline][None, :], (ns, ni)),
            missing.T,
        )
        pooled_masks = tuple(display_mask(mask, max_axis=max_axis) for mask in plane_masks)
        # These are display copies; source memmaps and analytical arrays remain untouched.
        planes = tuple(np.where(mask, np.nan, plane) for mask, plane in zip(pooled_masks, planes))
    limits = value_range or (float(np.nanpercentile(planes[0], 1)), float(np.nanpercentile(planes[0], 99)))
    fig = go.Figure()
    # inline section: x fixed; crossline and sample vary
    y = np.arange(0, nc, strides[0][1]); z = np.arange(0, ns, strides[0][0])
    yy, zz = np.meshgrid(y, z)
    fig.add_trace(go.Surface(x=np.full_like(yy, inline), y=yy, z=zz, surfacecolor=planes[0],
                             cmin=limits[0], cmax=limits[1], colorscale=colorscale,
                             showscale=True, colorbar={"title": "Amplitude"}, name="Inline section",
                             connectgaps=False))
    # crossline section: y fixed; inline and sample vary
    x = np.arange(0, ni, strides[1][1]); z = np.arange(0, ns, strides[1][0])
    xx, zz = np.meshgrid(x, z)
    fig.add_trace(go.Surface(x=xx, y=np.full_like(xx, crossline), z=zz,
                             surfacecolor=planes[1], cmin=limits[0], cmax=limits[1],
                             colorscale=colorscale, showscale=False, name="Crossline section",
                             connectgaps=False))
    # sample plane: z fixed; inline and crossline vary
    x = np.arange(0, ni, strides[2][1]); y = np.arange(0, nc, strides[2][0])
    xx, yy = np.meshgrid(x, y)
    fig.add_trace(go.Surface(x=xx, y=yy, z=np.full_like(xx, sample), surfacecolor=planes[2],
                             cmin=limits[0], cmax=limits[1], colorscale=colorscale,
                             showscale=False, name="Sample section", connectgaps=False))
    if finding_context is not None:
        result,finding=finding_context
        i,x,s=finding['representative'];i0,_,x0,_=result['bounds']
        if result['labels'][i-i0,x-x0]!=finding['label']:raise ValueError('Invalid finding representative')
        rows,cols=np.nonzero(result['labels']==finding['label'])
        fig.add_trace(go.Scatter3d(x=rows+i0,y=cols+x0,z=np.full(rows.shape,s),mode='markers',
            marker={'size':3,'color':'#31edc2','opacity':.55},name='2D classified footprint · not a 3D body'))
        fig.add_trace(go.Scatter3d(x=[i],y=[x],z=[s],mode='markers',marker={'size':8,'color':'#ffb000'},name='Finding representative'))
    fig.update_layout(title=title, scene={"xaxis_title": "Inline index", "yaxis_title": "Crossline index",
                                          "zaxis_title": "Sample Index", "aspectmode": "data",
                                          "uirevision": "rtm-camera"},
                      margin={"l": 0, "r": 0, "t": 48, "b": 0}, height=660,
                      legend={"orientation": "h"})
    return fig
