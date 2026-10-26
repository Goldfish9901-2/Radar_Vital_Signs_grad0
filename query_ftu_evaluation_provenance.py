"""Fetch only documentation relevant to trigger/RR/ECG/export questions."""
import concurrent.futures
import hashlib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT=Path('experiment_runs/ftu_evaluation_controls/provenance');ROOT.mkdir(exist_ok=True,parents=True)
URLS={
    'dataset_paper_html':'https://arxiv.org/html/2405.12659v1',
    'polar_flow_export':'https://support.polar.com/en/support/how_do_i_export_individual_training_sessions_from_polar_flow_web_service',
    'polar_sdk_readme':'https://raw.githubusercontent.com/polarofficial/polar-ble-sdk/master/README.md',
    'polar_h10_product':'https://raw.githubusercontent.com/polarofficial/polar-ble-sdk/master/documentation/products/PolarH10.md',
}


def fetch(item):
    name,url=item;begin=time.perf_counter()
    old=ROOT/'requests.json'
    if old.exists():
        cached=next((x for x in json.loads(old.read_text()) if x['name']==name and x.get('url')==url and x.get('status')==200 and '/404/' not in x.get('final_url','')),None)
        if cached and (ROOT/(name+'.txt')).exists():return cached
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 scientific reference provenance audit'})
        with urllib.request.urlopen(req,timeout=20) as r:
            b=r.read(4_000_000);status=r.status;final=r.url
        path=ROOT/(name+'.txt');path.write_bytes(b)
        return dict(name=name,url=url,final_url=final,status=status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),elapsed_seconds=time.perf_counter()-begin)
    except Exception as e:return dict(name=name,url=url,error=str(e),elapsed_seconds=time.perf_counter()-begin)


if __name__=='__main__':
    if (ROOT/'requests.json').exists() and not (ROOT/'requests_initial.json').exists():
        (ROOT/'requests_initial.json').write_bytes((ROOT/'requests.json').read_bytes())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,URLS.items()))
    (ROOT/'requests.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    for name in ['dataset_paper_html','polar_flow_export']:
        source=ROOT/(name+'.txt')
        if not source.exists():continue
        text=source.read_text(encoding='utf-8')
        text=re.sub(r'<(script|style)\b[^>]*>.*?</\1>','',text,flags=re.I|re.S)
        text=re.sub(r'</(?:p|h[1-6]|div|li|tr)>','\n',text)
        text=html.unescape(re.sub('<[^>]+>',' ',text))
        text='\n'.join(re.sub(r'\s+',' ',s).strip() for s in text.splitlines() if s.strip())
        (ROOT/('dataset_paper_text.txt' if name=='dataset_paper_html' else 'polar_flow_export_text.txt')).write_text(text,encoding='utf-8')
