"""Endpoint transition builder; intermediate frames are explicitly visual interpolation."""

import numpy as np


def interpolated_frames(baseline_slice: np.ndarray, monitor_slice: np.ndarray, frames: int = 9) -> tuple[list[np.ndarray], str]:
    if baseline_slice.shape != monitor_slice.shape or frames < 2:
        raise ValueError("matching endpoint arrays and at least two frames are required")
    result = [((1 - t) * baseline_slice + t * monitor_slice) for t in np.linspace(0, 1, frames)]
    return result, "Visual interpolation between Survey A and Survey B; intermediate frames are not measured vintages."


def animation_figure(baseline_slice: np.ndarray, monitor_slice: np.ndarray,
                     overlay_mask: np.ndarray | None = None):
    import plotly.graph_objects as go

    frames, _ = interpolated_frames(baseline_slice, monitor_slice)
    zmin, zmax = float(np.nanmin([baseline_slice, monitor_slice])), float(np.nanmax([baseline_slice, monitor_slice]))
    start = [go.Heatmap(z=frames[0], zmin=zmin, zmax=zmax, colorscale="RdBu", colorbar={"title": "Amplitude"})]
    overlay = None
    if overlay_mask is not None:
        if overlay_mask.shape != baseline_slice.shape:
            raise ValueError("overlay mask must match slice shape")
        overlay = go.Heatmap(z=overlay_mask.astype(float), zmin=0, zmax=1,
                             colorscale=[[0, "rgba(0,0,0,0)"], [1, "cyan"]],
                             opacity=0.35, showscale=False, name="Detected anomaly")
        start.append(overlay)
    frame_data = []
    for frame in frames:
        traces = [go.Heatmap(z=frame, zmin=zmin, zmax=zmax, colorscale="RdBu")]
        if overlay is not None:
            traces.append(overlay)
        frame_data.append(go.Frame(data=traces))
    fig = go.Figure(data=start, frames=frame_data)
    fig.update_layout(title="Endpoint transition (visual interpolation)",
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]}]}],
        sliders=[{"steps": [{"method": "animate", "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}], "label": f"{i/(len(frames)-1):.0%}"} for i in range(len(frames))]}])
    return fig
