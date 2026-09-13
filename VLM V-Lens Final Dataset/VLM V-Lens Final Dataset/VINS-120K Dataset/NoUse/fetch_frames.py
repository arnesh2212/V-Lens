"""Extract selected input frames from VINS-120K UltraVideo tars on HuggingFace using HTTP range requests.
Walks tar headers (512 B each) until wanted members are found, downloads only those members, 
and shrinks them for efficient storage and fast VLM processing."""
import json, re, os, collections, time, threading, sys
import requests
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://huggingface.co/datasets/openvivo/VINS-120K/resolve/main/ultravideo/"
MAX_CLIPS = 60

# Ensure ultravideo.json is present
if not os.path.exists('ultravideo.json'):
    print("Error: 'ultravideo.json' not found in directory. Please place it here.")
    sys.exit(1)

d = json.load(open('ultravideo.json', encoding='utf-8'))

# Expanded good regex to capture traffic, school zones, intersections, vehicles, and urban elements
good = re.compile(
    r'\b(sign|signs|billboard|storefront|shop|store|sidewalk|crosswalk|pedestrian|tram|bus stop|awning|street lamp|market|cafe|restaurant|traffic light|crossing|alley|taxi|traffic|traffic jam|intersection|roundabout|congestion|street|road|highway|lane|school|school zone|school bus|playground|campus|vehicles|cars)\b', 
    re.I
)
bad = re.compile(r'\b(aerial|skyline|drone|from above|bird|hillside|canyon|river|coast|mountain|valley|rooftops|overhead|top-down|highway interchange|cityscape|wolf|animal)\b', re.I)

# Score clips: number of matching instructions
score = collections.Counter(); first = {}
for it in d:
    if good.search(it['instruction']) and not bad.search(it['instruction']):
        clip = it['input'].split('/')[2]; score[clip] += 1; first.setdefault(clip, it)

chosen = [c for c, _ in score.most_common(MAX_CLIPS)]
if os.path.exists('chosen_pass3.json'):
    _p = json.load(open('chosen_pass3.json', encoding='utf-8')); chosen = list(_p); first.update(_p)

want = collections.defaultdict(dict)   # tar -> {member_suffix: clip}
for c in chosen:
    it = first[c]; tar = it['input'].split('/')[0]
    want[tar][it['input']] = c

print(f"{len(chosen)} clips across {len(want)} tars", flush=True)

os.makedirs('frames', exist_ok=True); os.makedirs('preview', exist_ok=True)
log = open('fetch_log.txt', 'a'); lock = threading.Lock()
manifest = json.load(open('manifest.json', encoding='utf-8')) if os.path.exists('manifest.json') else {}

def walk(tar):
    s = requests.Session()
    url = BASE + tar + ".tar"
    try:
        r = s.head(url, allow_redirects=True, timeout=60); final = r.url; total = int(r.headers['Content-Length'])
    except Exception as e:
        with lock: log.write(f"{tar} head failed: {e}\n"); log.flush()
        return

    def rng(a, b):
        for k in range(5):
            try: return s.get(final, headers={'Range': f'bytes={a}-{b}'}, timeout=120).content
            except Exception: time.sleep(2 + 3 * k)
        raise RuntimeError("range failed")

    off = 0; todo = dict(want[tar]); found = 0; n = 0
    while off < total and todo:
        h = rng(off, off + 511); n += 1
        if h[:1] == b'\0': break
        name = h[0:100].rstrip(b'\0').decode('utf-8', 'replace'); size = int((h[124:136].rstrip(b'\0 ') or b'0'), 8); typ = h[156:157]
        prefix = h[345:500].rstrip(b'\0').decode('utf-8', 'replace')
        if prefix: name = prefix + '/' + name
        if typ == b'L':   # GNU long name
            name = rng(off + 512, off + 512 + size - 1).rstrip(b'\0').decode('utf-8', 'replace')
            off += 512 + ((size + 511) // 512) * 512
            h = rng(off, off + 511); size = int((h[124:136].rstrip(b'\0 ') or b'0'), 8); typ = h[156:157]
        data_off = off + 512
        hit = next((k for k in todo if name.endswith(k)), None)
        if hit and typ in (b'0', b''):
            clip = todo.pop(hit); dst = f"frames/{clip}.png"
            with open(dst, 'wb') as f:
                got = 0; CH = 8 * 1024 * 1024
                while got < size:
                    chunk = rng(data_off + got, data_off + min(got + CH, size) - 1); f.write(chunk); got += len(chunk)
            
            # --- SHRINK & OPTIMIZE FRAME SIZE ---
            try:
                with Image.open(dst) as im:
                    im = im.convert('RGB')
                    # Cap max dimension to 1280px to save storage space and speed up VLM inference
                    im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    optimized_dst = f"frames/{clip}.jpg"
                    im.save(optimized_dst, "JPEG", quality=85)
                    if optimized_dst != dst and os.path.exists(dst):
                        os.remove(dst)
                    dst = optimized_dst
                    w, h_ = im.size
            except Exception as e: 
                w = h_ = None

            # --- GENERATE PREVIEW ---
            try:
                with Image.open(dst) as preview_im:
                    preview_im.thumbnail((800, 800))
                    preview_im.save(f"preview/{clip}.jpg", quality=80)
            except Exception:
                pass

            with lock:
                manifest[clip] = {
                    "tar": tar, "member": name, "bytes": size, "size": [w, h_], 
                    "frame": dst, "preview": f"preview/{clip}.jpg",
                    "instruction_example": first[clip]['instruction']
                }
                json.dump(manifest, open('manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
                log.write(f"{tar} got {clip} {size/1e6:.1f}MB after {n} headers\n"); log.flush()
            found += 1
        off = data_off + ((size + 511) // 512) * 512
        
    with lock: log.write(f"{tar} done: {found}/{len(want[tar])} after {n} headers, {off/1e9:.1f} GB walked\n"); log.flush()

threads = [threading.Thread(target=walk, args=(t,), daemon=True) for t in want]
for t in threads: t.start()
for t in threads: t.join()
print("ALL DONE", len(manifest), "frames", flush=True)