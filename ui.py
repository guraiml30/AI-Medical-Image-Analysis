"""ui.py - medical-themed web UI (navy + teal clinical palette, ECG pulse header)."""

# Theme: clinical navy + teal + soft mint, white cards, green = normal, coral = attention.
UI_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MedVision AI | Medical Image Analysis</title>
<style>
:root{--navy:#0B2545;--navy2:#13315C;--teal:#13A89E;--teal2:#0E8379;--mint:#E8F7F5;
--bg:#F4F8FB;--card:#fff;--ink:#1C2B3A;--mute:#6B7C8F;--ok:#1B9E6B;--bad:#D64545;--line:#DCE6EE}
*{box-sizing:border-box;margin:0}
body{font-family:Inter,Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(120deg,var(--navy),var(--navy2));color:#fff;padding:22px 20px 0;position:relative;overflow:hidden}
.brand{display:flex;align-items:center;gap:12px;max-width:1000px;margin:auto}
.logo{width:42px;height:42px;border-radius:12px;background:var(--teal);display:grid;place-items:center;font-size:24px;font-weight:700}
h1{font-size:1.25rem}.sub{color:#A9C7DC;font-size:.82rem;margin-top:2px}
.ecg{display:block;width:100%;height:46px;margin-top:10px}
.ecg path{fill:none;stroke:var(--teal);stroke-width:2.2;stroke-dasharray:900;stroke-dashoffset:900;animation:beat 3.2s linear infinite}
@keyframes beat{to{stroke-dashoffset:0}}
main{max-width:1000px;margin:20px auto;padding:0 16px;display:grid;gap:18px;grid-template-columns:1fr 1fr}
@media(max-width:760px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 4px 14px rgba(11,37,69,.05)}
.card h2{font-size:1rem;color:var(--navy);margin-bottom:12px;display:flex;gap:8px;align-items:center}
.drop{border:2px dashed var(--teal);background:var(--mint);border-radius:14px;padding:28px 12px;text-align:center;cursor:pointer;color:var(--teal2);transition:.2s}
.drop:hover,.drop.on{background:#D3F0EC;transform:scale(1.01)}
.drop b{display:block;font-size:1.05rem}.drop small{color:var(--mute)}
#prev{display:none;width:100%;max-height:260px;object-fit:contain;background:#000;border-radius:12px;margin-top:12px}
button{width:100%;margin-top:12px;padding:12px;border:0;border-radius:12px;background:var(--teal);color:#fff;font-weight:600;font-size:1rem;cursor:pointer}
button:hover{background:var(--teal2)}button:disabled{opacity:.5;cursor:not-allowed}
.res{display:none;margin-top:14px;padding:14px;border-radius:12px;border-left:6px solid var(--ok);background:#EAF8F2}
.res.bad{border-color:var(--bad);background:#FDEEEE}
.res .lab{font-size:1.3rem;font-weight:700}.res.bad .lab{color:var(--bad)}.res .lab{color:var(--ok)}
.bar{height:10px;background:#E3EAF0;border-radius:99px;margin:10px 0 4px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--teal);transition:width .8s}
.res.bad .bar i{background:var(--bad)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.stat{background:var(--mint);border-radius:12px;padding:12px}
.stat b{font-size:1.4rem;color:var(--navy);display:block}.stat span{font-size:.75rem;color:var(--mute)}
.gal{grid-column:1/-1}.gal .g{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.gal img{width:100%;border-radius:12px;border:1px solid var(--line)}.gal p{font-size:.78rem;color:var(--mute);margin-top:4px}
.warn{grid-column:1/-1;background:#FFF7E6;border:1px solid #F5D78E;color:#7A5A00;padding:12px 14px;border-radius:12px;font-size:.82rem}
footer{text-align:center;color:var(--mute);font-size:.75rem;padding:18px}
</style></head><body>
<header>
 <div class="brand"><div class="logo">&#10010;</div>
  <div><h1>MedVision AI</h1><div class="sub">AI-Powered Medical Image Analysis &middot; Chest X-ray Screening Assistant</div></div></div>
 <svg class="ecg" viewBox="0 0 1000 46" preserveAspectRatio="none"><path d="M0 28 H180 L200 28 L215 6 L235 42 L250 28 H430 L450 28 L465 8 L485 40 L500 28 H700 L720 28 L735 4 L755 42 L770 28 H1000"/></svg>
</header>
<main>
 <section class="card"><h2>&#129659; Upload Scan</h2>
  <div class="drop" id="drop"><b>Drop X-ray here or tap to browse</b><small>PNG / JPG &middot; chest radiograph</small></div>
  <input type="file" id="file" accept="image/*" hidden>
  <img id="prev" alt="preview">
  <button id="go" disabled>Analyze Image</button>
  <div class="res" id="res"><div class="lab" id="lab"></div><div class="bar"><i id="fill"></i></div>
   <small id="conf"></small></div>
 </section>
 <section class="card"><h2>&#128202; Model Performance</h2>
  <div class="stats" id="stats"><div class="stat"><b>&ndash;</b><span>Run the pipeline first</span></div></div>
 </section>
 <section class="card gal"><h2>&#128247; Pipeline Outputs</h2><div class="g" id="gal"></div></section>
 <div class="warn">&#9888; Educational demo built on public / simulated data. Output is an AI-assisted suggestion, not a medical diagnosis. Always consult a qualified radiologist.</div>
</main>
<footer>MedVision AI &middot; FastAPI + TensorFlow + OpenCV &middot; API docs at <a href="/docs">/docs</a></footer>
<script>
const $=id=>document.getElementById(id);let f=null;
const drop=$('drop'),inp=$('file');
drop.onclick=()=>inp.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.add('on')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,x=>{x.preventDefault();drop.classList.remove('on')}));
drop.addEventListener('drop',e=>pick(e.dataTransfer.files[0]));inp.onchange=()=>pick(inp.files[0]);
function pick(x){if(!x)return;f=x;$('prev').src=URL.createObjectURL(x);$('prev').style.display='block';$('go').disabled=false;$('res').style.display='none'}
$('go').onclick=async()=>{
 $('go').disabled=true;$('go').textContent='Analyzing...';
 const fd=new FormData();fd.append('file',f);
 try{const r=await fetch('/predict',{method:'POST',body:fd});const d=await r.json();
  if(!r.ok)throw new Error(d.detail||'error');
  const bad=d.label==='PNEUMONIA';$('res').className='res'+(bad?' bad':'');$('res').style.display='block';
  $('lab').textContent=bad?'Possible Pneumonia Detected':'No Pneumonia Pattern Detected';
  $('conf').textContent='Confidence: '+d.confidence_percent+'% ('+d.label+')';
  setTimeout(()=>$('fill').style.width=d.confidence_percent+'%',50);
 }catch(e){alert('Prediction failed: '+e.message)}
 $('go').disabled=false;$('go').textContent='Analyze Image'};
fetch('/metrics').then(r=>r.json()).then(m=>{if(m.accuracy===undefined)return;
 const s=[['Accuracy',(m.accuracy*100).toFixed(1)+'%'],['Recall',(m.recall_pneumonia*100).toFixed(1)+'%'],
 ['Precision',(m.precision_pneumonia*100).toFixed(1)+'%'],['F1-score',(m.f1_pneumonia*100).toFixed(1)+'%']];
 $('stats').innerHTML=s.map(x=>`<div class="stat"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('')});
[['preprocessing_preview.png','Preprocessing'],['accuracy_loss_graph.png','Training curves'],
 ['confusion_matrix.png','Confusion matrix'],['prediction_results.png','Predicted vs actual']].forEach(([n,t])=>{
 const i=new Image();i.src='/outputs/'+n;i.onload=()=>{const d=document.createElement('div');d.append(i);
 const p=document.createElement('p');p.textContent=t;d.append(p);$('gal').append(d)}});
</script></body></html>
"""
