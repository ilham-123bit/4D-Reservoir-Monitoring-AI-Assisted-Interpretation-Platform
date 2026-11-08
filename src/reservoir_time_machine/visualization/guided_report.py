"""Compact report-first panel for every existing page; local evidence only."""
import streamlit as st
from ..analysis.guided_report import page_report
from ..ai.grounding import finding_packet,explanation_identity
from ..settings import LLMSettings


def render_page_report(result, finding, wells, page, discovery, *, shared_interpretation=None):
    report=page_report(result,finding,wells,page,discovery,shared_interpretation=shared_interpretation)
    st.markdown('### Automated evidence report')
    if finding:
        m=finding['metrics'];i,x,s=finding['representative']
        st.markdown(f"**RESULT · {m['mean_nrms_percent']:.1f}% NRMS** · inline {i} · crossline {x} · sample {s}")
        st.markdown(f"**DISCUSSION ·** Mean signed change {m['mean_signed_difference']:.4g}; cause N/A. Registration N/A.")
        st.markdown('**ACTION ·** Verify repeatability; obtain independent well evidence.')
    else:
        st.markdown('**RESULT · N/A** · No eligible completed finding.')
        st.markdown('**DISCUSSION ·** Current analysis coverage is limited.')
        st.markdown('**ACTION ·** Extend analysis in an authorized allocation.')
    if finding:
        try:
            key=explanation_identity(finding_packet(result,finding,well_context=wells),LLMSettings.from_environment())
            cached=st.session_state.get('grounded_explanations',{}).get(key)
            if cached and cached['provenance']['source']=='validated_provider_selection':
                report['interpretation']=cached
                report['source']='Gemma · cached validated evidence selection; not geological validation'
        except ValueError:pass
        st.caption('Explanation source: '+report['source'])
        well_count=report['wells']['well_records']
        st.markdown(f"**Well evidence:** {well_count} workbook records · seismic tie N/A")
        if finding['touches_tile_boundary']:
            st.caption('Extent: N/A · finding touches analysis-tile boundary.')
    return report
