# Visualization

The active Streamlit/Plotly app reads the configured baseline and monitor SEG-Y files. It presents metadata/QC, paired volume views, selected-sample difference/NRMS maps, selected-sample MAD screening, connected components, an availability view for well support files, and an optional structured-evidence explanation.

Full volume displays are downsampled into derived copies by the Plotly helper; the stored input arrays are unchanged. Difference and NRMS displays calculate the selected sample window rather than generating several full attribute volumes. Large input loading is memory bounded, and an explicitly configured scratch cache is the only supported memmap output.

The endpoint animation uses actual baseline and monitor sections. Intermediate frames are linear display interpolation only, never measured observations or a verified trajectory. No well position is plotted until geometry, coordinate reference, and unit transformations have been verified from actual metadata.

**Execution status:** the source has not been launched in the current local environment because Streamlit and Plotly are unavailable and the dataset path is inaccessible. Dashboard startup with missing data is designed to show a clear unavailable state, but has not been run end to end here.
