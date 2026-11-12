"""Small existing-page integration for bounded preliminary ML; no new page or external calls."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from ..analysis.model_catalog import model_catalog
from ..ai.grounding import finding_packet, interpret, GemmaFindingProvider, explanation_identity, transmission_allowed
from ..settings import LLMSettings
from .slices import display_coordinates, display_region


def select_finding():
    finding = st.session_state.get('preliminary_findings_by_id',{}).get(st.session_state.get('preliminary_choice'))
    if finding:
        st.session_state['selected_preliminary_finding'] = finding
        i,x,s = finding['representative']
        st.session_state['shared_inline'],st.session_state['shared_crossline'],st.session_state['shared_sample'] = i,x,s


def render_preliminary(result, *, allow_external=False, well_context=None):
    st.markdown('### Bounded preliminary ML evidence')
    st.caption(f"Tile {result['bounds']} · sample {result['sample']} · {int(result['valid'].sum()):,} eligible cells. "
               'Unlabeled reference; unverified registration. Results are investigation priorities, not probabilities.')
    with st.expander('Model availability and reproducibility'):
        st.dataframe([{'model':name,**value} for name,value in result['model_status'].items()],use_container_width=True)
        st.write(result['provenance'])
        st.dataframe(model_catalog(),use_container_width=True)
    findings = result['findings']
    if not findings:
        st.info('No coherent candidates meet this tile’s criteria. This does not establish absence of reservoir change.')
        return
    # Same IDs and global source indices feed Change Detection and Evidence Explanation.
    st.session_state['preliminary_findings_by_id'] = {f['id']:{**f,'dataset_id':result['provenance']['dataset_id']} for f in findings}
    ids = list(st.session_state['preliminary_findings_by_id'])
    if st.session_state.get('preliminary_choice') not in ids:
        st.session_state['preliminary_choice'] = ids[0]
    selected_id = st.selectbox('Preliminary finding',ids,key='preliminary_choice',on_change=select_finding)
    finding = next(f for f in findings if f['id']==selected_id)
    st.button('Inspect finding in linked sections',on_click=select_finding,key='inspect_preliminary_finding')
    st.session_state['selected_preliminary_finding'] = {**finding, 'dataset_id':result['provenance']['dataset_id']}
    st.dataframe([{'finding':f['id'],'priority':round(f['priority'],4),'cells':f['metrics']['cells'],
                   'representative':str(f['representative']),'tile_edge':f['touches_tile_boundary']} for f in findings[:20]],
                 use_container_width=True)
    st.write({'selected_finding':finding['id'],'source_indices':finding['representative'],
              'attributes':finding['metrics'],'support_and_disagreement':finding['method_evidence'],
              'priority_contributions':finding['priority_terms']})
    i0,i1,x0,x1 = result['bounds']
    xs,ys = display_coordinates(result['valid'].shape,128)
    fig = go.Figure(go.Heatmap(z=result['scores']['statistical'],x=xs+x0,y=ys+i0,
                             colorscale='Magma',colorbar={'title':'Absolute difference'}))
    fig.add_trace(go.Heatmap(z=np.where(display_region(result['labels'],finding['label'],128),1,np.nan),
                            x=xs+x0,y=ys+i0,showscale=False,colorscale=[[0,'#ff7047'],[1,'#ff7047']],opacity=.5))
    st.plotly_chart(fig.update_layout(title='Selected preliminary finding · source grid indices',
                    xaxis_title='Crossline source index',yaxis_title='Inline source index',height=400),
                    use_container_width=True,theme=None)
    if finding['touches_tile_boundary']:
        st.caption('Candidate intersects the tile boundary; extent may continue outside the analyzed area.')
    packet = finding_packet(result,finding,well_context=well_context)
    try:
        settings = LLMSettings.from_environment()
    except ValueError:
        settings = LLMSettings()
        st.warning('Invalid Gemma configuration; using the local grounded explanation.')
    explanation_key = explanation_identity(packet,settings)
    explanations = st.session_state.setdefault("grounded_explanations", {})
    cached = explanations.get(explanation_key)
    explanation = {**cached, 'provenance':{**cached['provenance'],'cache_hit':True}} if cached else interpret(packet)
    if allow_external:
        eligible = bool(settings.base_url and transmission_allowed(packet,settings,authorized=True))
        if not eligible:
            st.info('Gemma transmission for Challenge 3 evidence is disabled until policy confirmation, deployment authorization and endpoint configuration are recorded. Local explanations remain available.')
        authorized = st.checkbox('I authorize transmitting this bounded finding packet to the configured Gemma endpoint.',
                                 key='phase2_transmission', disabled=not eligible)
        if st.button('Request grounded Gemma explanation',disabled=not (authorized and eligible)):
            # Recheck here as well as in the provider; changing a widget never sends data.
            explanation = explanation if cached else interpret(packet,provider=GemmaFindingProvider(settings),authorized=authorized and eligible)
            if explanation['provenance']['source']=='validated_provider_selection':
                explanations[explanation_key] = explanation
                while len(explanations)>32:
                    explanations.pop(next(iter(explanations)))
    provenance = explanation['provenance']
    if provenance['source']=='validated_provider_selection':
        st.success('Explanation source: Gemma · references and measurements validated · not geological validation'+(' · cached' if provenance.get('cache_hit') else ''))
    else:
        st.info('Explanation source: deterministic local fallback' if provenance['source']=='deterministic_fallback' else 'Explanation source: deterministic local summary')
        if provenance.get('error_type'):
            st.caption('Provider unavailable or response rejected: '+provenance['error_type'])
    for claim in explanation['claims']:
        st.markdown(f"**{claim['category'].replace('_',' ').title()}:** {claim['text']}")
        st.caption('Evidence: '+', '.join(claim['evidence_ids']))
    with st.expander('Interpretation evidence and provenance'):
        st.write(explanation['provenance'])
        st.json(packet.to_dict())
        st.write(result['limitations'])
