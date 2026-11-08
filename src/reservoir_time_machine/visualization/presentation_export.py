"""Self-contained offline storyboard: no CDN, remote API or invented vintages."""
import json,html
from plotly.utils import PlotlyJSONEncoder
from plotly.offline import get_plotlyjs
from .finding_animation import finding_animation
from .presentation import scene_context
from ..analysis.technical_report import storyboard,report_markdown


def export_presentation(registry,baseline,monitor,missing,path,limit=8):
    if not 1<=limit<=8:raise ValueError('Export budget is one to eight findings')
    items=[]
    for entry in registry['entries'][:limit]:
        fig=finding_animation(baseline,monitor,missing,entry['result'],entry['finding'])
        items.append({'id':entry['id'],'scenes':storyboard(registry,entry),'figure':fig.to_plotly_json(),
            'context':scene_context(registry,entry).to_plotly_json()})
    if not items:raise ValueError('No eligible finding: a presentation is not fabricated')
    payload=json.dumps(items,cls=PlotlyJSONEncoder,allow_nan=False).replace('</','<\\/')
    document='''<!doctype html><html><head><meta charset="utf-8"><title>Reservoir Time Machine · evidence presentation</title>
<style>body{background:#101719;color:#f3fffb;font:18px Arial,sans-serif;margin:0;padding:28px}h1,h2{color:#39f4c5}button,select{background:#203b34;color:white;border:1px solid #39f4c5;padding:12px;font-size:17px;border-radius:8px;margin-right:8px}#scene{line-height:1.6;max-width:1200px}pre{white-space:pre-wrap;line-height:1.5;font:16px Arial}#context{max-width:700px}summary{color:#39f4c5;cursor:pointer}</style></head><body>
<p>RESERVOIR TIME MACHINE · PRESENTATION MODE · PRELIMINARY SCREENING</p><h1>Finding-directed 4D interpretation</h1>
<p>Trace-key pairing is available on the Baseline grid. Physical 4D registration is not yet verified. Interpret paired attributes as preliminary screening only.</p>
<button id="toggle">Pause</button><button id="prev">Previous finding</button><button id="next">Next finding</button><select id="finding"></select><select id="speed"><option value="9000">Readable</option><option value="15000">Slow</option><option value="5000">Fast</option></select><button id="step">Next scene</button>
<h2 id="title"></h2><p id="scene"></p><div id="context"></div><div id="plot"></div>
<p>Presentation sequence follows completed plan groups and local rank. No survey-wide rank, verified well tie, unmeasured reservoir history or recovery conclusion is implied. Export includes the first eight findings at most; the full report lists completed findings.</p>
<details><summary>Complete technical report and evidence provenance</summary><pre>REPORT_CONTENT</pre></details>
<script>PLOTLY_LIBRARY</script><script>
const items=DATA_PAYLOAD; let f=0,s=0,playing=true,timer,drawn=-1;
const select=document.getElementById('finding');items.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v.id;select.appendChild(o)});
async function render(){const v=items[f]; select.value=f;document.getElementById('title').textContent=(s+1)+'/8 · '+v.scenes[s][0];document.getElementById('scene').textContent=v.scenes[s][1];
if(drawn!==f){await Plotly.react('plot',v.figure.data,v.figure.layout,{responsive:true,displaylogo:false});await Plotly.addFrames('plot',v.figure.frames);await Plotly.react('context',v.context.data,v.context.layout,{responsive:true,displaylogo:false});drawn=f;}
await Plotly.animate('plot',[s===3||s===4?'Monitor':'Baseline'],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}});}
function schedule(){clearTimeout(timer);if(playing)timer=setTimeout(async()=>{s++;if(s===8){s=0;f=(f+1)%items.length;}await render();schedule();},Number(document.getElementById('speed').value));}
document.getElementById('toggle').onclick=()=>{playing=!playing;document.getElementById('toggle').textContent=playing?'Pause':'Play';schedule()};
document.getElementById('prev').onclick=async()=>{f=(f+items.length-1)%items.length;s=0;await render();schedule()};
document.getElementById('next').onclick=async()=>{f=(f+1)%items.length;s=0;await render();schedule()};
select.onchange=async()=>{f=Number(select.value);s=0;await render();schedule()};document.getElementById('speed').onchange=schedule;
document.getElementById('step').onclick=async()=>{s=(s+1)%8;await render();schedule()};render().then(schedule);
</script></body></html>'''
    document=document.replace('REPORT_CONTENT',html.escape(report_markdown(registry))).replace('PLOTLY_LIBRARY',get_plotlyjs()).replace('DATA_PAYLOAD',payload)
    path.write_text(document);return path
