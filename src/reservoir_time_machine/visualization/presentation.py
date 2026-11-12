"""Finding-driven, synchronized Streamlit presentation; no LLM request side effects."""
import time
import streamlit as st
import plotly.graph_objects as go
from ..analysis.technical_report import storyboard


def select_entry(entries,offset=0):
    ids=[e['id'] for e in entries]
    if not ids:return
    current=st.session_state.get('registry_selection',ids[0]);index=ids.index(current) if current in ids else 0
    selected=ids[(index+offset)%len(ids)];entry=next(e for e in entries if e['id']==selected)
    st.session_state['registry_selection']=selected;st.session_state['preliminary_choice']=selected
    st.session_state['selected_preliminary_finding']={**entry['finding'],'dataset_id':entry['result']['provenance']['dataset_id']}
    i,x,s=entry['finding']['representative']
    for k,v in zip(('shared_inline','shared_crossline','shared_sample'),(i,x,s)):st.session_state[k]=v
    st.session_state['presentation_scene']=0;st.session_state['presentation_last_tick']=time.monotonic()


def scene_context(registry,entry):
    i,x,s=entry['finding']['representative'];shape=registry['body']['shape']
    fig=go.Figure()
    fig.add_shape(type='rect',x0=0,y0=0,x1=shape[1]-1,y1=shape[0]-1,line_color='#718a85')
    for run in registry['body']['runs']:
        if run['sample']==s:
            a,b,c,d=run['bounds'];fig.add_shape(type='rect',x0=c,y0=a,x1=d-1,y1=b-1,fillcolor='rgba(49,237,194,.2)',line_color='#31edc2')
    fig.add_trace(go.Scatter(x=[x],y=[i],mode='markers',marker={'size':15,'color':'#31edc2'},name=entry['id']))
    return fig.update_layout(template='plotly_dark',paper_bgcolor='#101719',plot_bgcolor='#152125',font={'color':'#f3fffb','size':15},title='Survey grid context · completed tiles at selected sample only',height=260,
        xaxis={'title':'Crossline index','range':[0,max(1,shape[1]-1)]},yaxis={'title':'Inline index','range':[0,max(1,shape[0]-1)]},showlegend=False)


def render_presentation(registry,entry,figure):
    scenes=storyboard(registry,entry)
    st.session_state.setdefault('presentation_scene',0);st.session_state.setdefault('presentation_playing',True)
    st.session_state.setdefault('presentation_last_tick',time.monotonic())
    cols=st.columns(4)
    if cols[0].button('Pause presentation' if st.session_state['presentation_playing'] else 'Play presentation'):
        st.session_state['presentation_playing']=not st.session_state['presentation_playing'];st.session_state['presentation_last_tick']=time.monotonic()
    if cols[1].button('Previous scene'):
        st.session_state['presentation_scene']=(st.session_state['presentation_scene']-1)%len(scenes);st.session_state['presentation_last_tick']=time.monotonic()
    if cols[2].button('Next scene'):
        st.session_state['presentation_scene']=(st.session_state['presentation_scene']+1)%len(scenes);st.session_state['presentation_last_tick']=time.monotonic()
    speed=cols[3].selectbox('Presentation speed',['Readable','Slow','Fast'],index=0)
    duration={'Readable':9,'Slow':15,'Fast':5}[speed]
    @st.fragment(run_every=1)
    def clock():
        if st.session_state['presentation_playing'] and time.monotonic()-st.session_state['presentation_last_tick']>=duration:
            scene=st.session_state['presentation_scene']+1
            if scene>=len(scenes):
                ids=[v['id'] for v in registry['entries']]
                current=st.session_state.get('registry_selection',ids[0])
                st.session_state['pending_registry_selection']=ids[(ids.index(current)+1)%len(ids)];scene=0
            st.session_state['presentation_scene']=scene;st.session_state['presentation_last_tick']=time.monotonic()
            st.rerun(scope='app')
    clock()
    index=st.session_state['presentation_scene']%len(scenes);title,text=scenes[index]
    st.markdown(f'## {index+1} / {len(scenes)} · {title}')
    st.write(text)
    if index in (0,1):st.plotly_chart(scene_context(registry,entry),use_container_width=True,theme=None)
    # A fresh layout copy avoids mutating a cached figure as the storyboard advances.
    display=go.Figure(figure)
    endpoint=1 if index in (3,4) else 0
    frame=display.frames[endpoint]
    for target,data in zip(frame.traces,frame.data):display.data[target].update(z=data.z)
    display.update_layout(title_text=frame.layout.title.text)
    st.plotly_chart(display,use_container_width=True,theme=None)
    st.caption('Autoplay follows plan order and local tile rank, never a calibrated survey-wide ranking. The same registry selection updates report and section coordinates on every scene/finding change. Only measured endpoints are shown.')
