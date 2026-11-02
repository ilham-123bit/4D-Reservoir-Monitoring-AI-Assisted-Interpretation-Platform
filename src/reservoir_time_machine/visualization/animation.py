"""Endpoint transition builder; intermediate frames are explicitly visual interpolation."""

import numpy as np
from .slices import display_mask


def interpolated_frames(baseline_slice: np.ndarray, monitor_slice: np.ndarray, frames: int = 9) -> tuple[list[np.ndarray], str]:
    if baseline_slice.shape != monitor_slice.shape or frames < 2:
        raise ValueError("matching endpoint arrays and at least two frames are required")
    result = [((1 - t) * baseline_slice + t * monitor_slice) for t in np.linspace(0, 1, frames)]
    return result, "Visual interpolation between Survey A and Survey B; intermediate frames are not measured vintages."


def animation_figure(
    baseline_slice: np.ndarray,
    monitor_slice: np.ndarray,
    overlay_mask: np.ndarray | None = None,
    *,
    missing_trace_mask: np.ndarray | None = None,
    max_axis: int = 600,
    frame_duration_ms: int = 180,
):
    """Build a display animation from reduced endpoint arrays."""
    import plotly.graph_objects as go

    baseline_data = np.asarray(baseline_slice)
    monitor_data = np.asarray(monitor_slice)

    if baseline_data.ndim != 2 or baseline_data.shape != monitor_data.shape:
        raise ValueError("matching 2D endpoint arrays are required")
    if max_axis < 1:
        raise ValueError("max_axis must be positive")
    if frame_duration_ms < 1:
        raise ValueError("frame_duration_ms must be positive")

    if overlay_mask is not None and np.shape(overlay_mask) != baseline_data.shape:
        raise ValueError("overlay mask must match original endpoint shape")
    if missing_trace_mask is not None and np.shape(missing_trace_mask) != baseline_data.shape:
        raise ValueError("missing trace mask must match original endpoint shape")

    row_stride = max(1, int(np.ceil(baseline_data.shape[0] / max_axis)))
    col_stride = max(1, int(np.ceil(baseline_data.shape[1] / max_axis)))

    baseline_display = baseline_data[::row_stride, ::col_stride]
    monitor_display = monitor_data[::row_stride, ::col_stride]

    overlay_display = None
    if overlay_mask is not None:
        overlay_display = display_mask(overlay_mask, max_axis=max_axis)
    x_coordinates = np.arange(0, baseline_data.shape[1], col_stride)
    y_coordinates = np.arange(0, baseline_data.shape[0], row_stride)
    missing_display = None
    if missing_trace_mask is not None:
        missing_display = display_mask(missing_trace_mask, max_axis=max_axis)
        baseline_display = np.where(missing_display, np.nan, baseline_display)
        monitor_display = np.where(missing_display, np.nan, monitor_display)

    frames, _ = interpolated_frames(
        baseline_display,
        monitor_display,
    )
    zmin = float(np.nanmin([baseline_display, monitor_display]))
    zmax = float(np.nanmax([baseline_display, monitor_display]))

    start = [
        go.Heatmap(
            z=frames[0],
            x=x_coordinates,
            y=y_coordinates,
            zmin=zmin,
            zmax=zmax,
            colorscale="RdBu",
            colorbar={"title": "Amplitude"},
        )
    ]

    if overlay_display is not None:
        start.append(
            go.Heatmap(
                z=overlay_display.astype(float),
                x=x_coordinates,
                y=y_coordinates,
                zmin=0,
                zmax=1,
                colorscale=[
                    [0, "rgba(0,0,0,0)"],
                    [1, "#ff6b3d"],
                ],
                opacity=0.35,
                showscale=False,
                name="Detected anomaly (display grid)",
            )
        )

    if missing_display is not None:
        start.append(go.Heatmap(z=np.where(missing_display, 1.0, np.nan), x=x_coordinates, y=y_coordinates, zmin=0, zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ded7d5"]], opacity=.9,
            showscale=False, name="Missing trace coverage"))

    frame_data = []
    for frame_index, frame in enumerate(frames):
        traces = [
            go.Heatmap(
                z=frame,
                x=x_coordinates,
                y=y_coordinates,
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu",
            )
        ]
        if overlay_display is not None:
            traces.append(
                go.Heatmap(
                    z=overlay_display.astype(float),
                    x=x_coordinates,
                    y=y_coordinates,
                    zmin=0,
                    zmax=1,
                    colorscale=[
                        [0, "rgba(0,0,0,0)"],
                        [1, "#ff6b3d"],
                    ],
                    opacity=0.35,
                    showscale=False,
                    name="Detected anomaly (display grid)",
                )
            )
        if missing_display is not None:
            traces.append(go.Heatmap(z=np.where(missing_display, 1.0, np.nan), x=x_coordinates, y=y_coordinates, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "#ded7d5"]], opacity=.9,
                showscale=False, name="Missing trace coverage"))
        # The slider targets these names; without names Plotly silently ignores
        # slider navigation in the dashboard.
        frame_data.append(go.Frame(name=str(frame_index), data=traces))

    fig = go.Figure(data=start, frames=frame_data)
    fig.update_layout(
        title="Baseline → Monitor · visual interpolation · display-downsampled",
        xaxis_title="Crossline index",
        yaxis_title="Inline index",
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{i / (len(frames) - 1):.0%}",
                    }
                    for i in range(len(frames))
                ]
            }
        ],
    )
    return fig
