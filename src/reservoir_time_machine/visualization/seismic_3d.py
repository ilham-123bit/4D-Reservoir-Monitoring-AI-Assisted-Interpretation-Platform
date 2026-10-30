"""Lazy Plotly rendering for bounded-size 3D seismic display copies."""

import numpy as np

from .slices import display_volume


def volume_figure(volume: np.ndarray, title: str, *, max_axis: int = 40, colorscale: str = "RdBu",
                  value_range: tuple[float, float] | None = None):
    import plotly.graph_objects as go

    shown = display_volume(volume, max_axis=max_axis)
    ii, xx, ss = np.indices(shown.shape)
    limits = value_range or (float(np.nanpercentile(shown, 10)), float(np.nanpercentile(shown, 90)))
    fig = go.Figure(go.Volume(x=ii.ravel(), y=xx.ravel(), z=ss.ravel(), value=np.nan_to_num(shown).ravel(),
                              isomin=limits[0], isomax=limits[1],
                              opacity=0.10, surface_count=12, colorscale=colorscale,
                              caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}}))
    fig.update_layout(title=title, scene={"xaxis_title": "Inline index", "yaxis_title": "Crossline index",
                                          "zaxis_title": "Sample index", "aspectmode": "data"},
                      margin={"l": 0, "r": 0, "t": 44, "b": 0}, height=560)
    return fig
