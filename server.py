"""Tiny web interface: python server.py  ->  http://localhost:8000  (standard library only)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from chatbot import Bot, DISCLAIMER

bot = Bot()
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Breast Cancer Info Chatbot</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f4f7;color:#222}
header{background:#7a2e5b;color:#fff;padding:16px 20px}header h1{margin:0;font-size:18px}
header p{margin:4px 0 0;font-size:13px;opacity:.9}
main{max-width:760px;margin:0 auto;padding:16px}
#log{min-height:50vh;display:flex;flex-direction:column;gap:12px;padding-bottom:12px}
.msg{padding:12px 14px;border-radius:12px;max-width:88%;line-height:1.5;font-size:15px;white-space:pre-wrap}
.me{align-self:flex-end;background:#7a2e5b;color:#fff}.bot{align-self:flex-start;background:#fff;border:1px solid #e3dde6}
.typing{align-self:flex-start;background:#fff;border:1px solid #e3dde6;border-radius:12px;padding:12px 16px;display:flex;gap:5px}
.typing i{width:7px;height:7px;background:#b9a9c2;border-radius:50%;animation:b 1s infinite}.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}
@keyframes b{0%,60%,100%{transform:translateY(0);opacity:.5}30%{transform:translateY(-4px);opacity:1}}
.src{display:block;margin-top:8px;font-size:12px;color:#555}.src a{color:#7a2e5b}
form{display:flex;gap:8px;position:sticky;bottom:0;background:#f6f4f7;padding:10px 0}
input{flex:1;padding:12px;border:1px solid #cfc6d3;border-radius:10px;font-size:15px}
button{padding:12px 18px;background:#7a2e5b;color:#fff;border:0;border-radius:10px;font-size:15px;cursor:pointer}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}.chips button{background:#fff;color:#7a2e5b;border:1px solid #cfc6d3;font-size:13px;padding:7px 11px}
footer{font-size:12px;color:#666;text-align:center;padding:10px}
</style></head><body>
<header><h1>Breast Cancer Info Chatbot</h1><p>Answers come only from National Cancer Institute pages and cite the page they used.</p></header>
<main><div class="chips" id="chips"></div><div id="log"></div>
<form id="f"><input id="q" placeholder="Ask about screening, mammograms, risk, BRCA genes, treatment..." autocomplete="off"><button>Ask</button></form>
<footer>__DISCLAIMER__</footer></main>
<script>
const log=document.getElementById('log'),hist=[];
const ex=["How often should I get a mammogram?","What does BI-RADS 3 mean?","Who should consider BRCA testing?","What is triple negative breast cancer?"];
ex.forEach(t=>{const b=document.createElement('button');b.type='button';b.textContent=t;b.onclick=()=>send(t);chips.appendChild(b)});
function add(cls,text,sources){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;
 (sources||[]).forEach(s=>{const a=document.createElement('span');a.className='src';a.innerHTML='Source: <a target="_blank" rel="noopener"></a>';
  const l=a.querySelector('a');l.href=s.url;l.textContent=s.title;d.appendChild(a)});log.appendChild(d);d.scrollIntoView()}
add('bot',"Hi! I answer questions about breast cancer using National Cancer Institute pages and show you the page each answer came from. I'm an automated assistant, not a person, and I can't give personal medical advice. What would you like to know?");
async function send(t){add('me',t);const ty=document.createElement('div');ty.className='typing';ty.innerHTML='<i></i><i></i><i></i>';log.appendChild(ty);ty.scrollIntoView();
 try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q:t,history:hist})});
 const j=await r.json();await new Promise(z=>setTimeout(z,350));ty.remove();hist.push(t);add('bot',j.answer,j.sources)}
 catch(e){ty.remove();add('bot','Sorry, something went wrong. Please try again.')}}
document.getElementById('f').onsubmit=e=>{e.preventDefault();const i=document.getElementById('q');const t=i.value.trim();if(t){i.value='';send(t)}};
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.replace("__DISCLAIMER__", DISCLAIMER).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        out = json.dumps(bot.ask(str(data.get("q", ""))[:500], data.get("history", [])[-5:])).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a): pass


if __name__ == "__main__":
    print("Open http://localhost:8000"); HTTPServer(("127.0.0.1", 8000), H).serve_forever()
