"""
Build ONE self-contained HTML dashboard from the pipeline outputs.

Shows: satellite basemap, predicted road mask, extracted road graph, intersections
(coloured by degree), OSM validation points (if validation.geojson exists), network
statistics, click-two-points shortest-path routing over the extracted graph, and
GeoJSON download buttons.

Open outputs/gandhinagar/dashboard.html in Chrome/Edge (double-click). The map library is built
into the file; only the satellite basemap needs internet. Keep mask_overlay.png in the same folder.

Run from the repository root:
    python geospatial/make_dashboard.py
"""
import argparse
import json
import os

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Road Network Extraction Dashboard</title>
<style>__LEAFLET_CSS__</style>
<style>
 :root{--bg:#fff;--fg:#1c2330;--muted:#5b6678;--line:#dfe3ea;--accent:#1f6feb}
 @media (prefers-color-scheme:dark){:root{--bg:#161b22;--fg:#e6edf3;--muted:#9aa7b8;--line:#2d333b;--accent:#58a6ff}}
 *{box-sizing:border-box} body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);display:flex;height:100vh}
 #side{width:340px;max-width:42vw;overflow:auto;padding:16px;border-right:1px solid var(--line)}
 #map{flex:1} h1{font-size:17px;margin:0 0 2px} .sub{color:var(--muted);font-size:12px;margin-bottom:12px}
 h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:16px 0 6px}
 table{width:100%;border-collapse:collapse} td{padding:3px 0;border-bottom:1px solid var(--line)} td:last-child{text-align:right;font-variant-numeric:tabular-nums}
 button{font:inherit;padding:6px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer;margin:2px 4px 2px 0}
 button:hover{border-color:var(--accent);color:var(--accent)} .note{font-size:12px;color:var(--muted)}
 #hint{position:fixed;left:calc(340px + (100vw - 340px)/2);bottom:22px;transform:translateX(-50%);z-index:1000;background:var(--bg);color:var(--fg);border:1px solid var(--accent);border-radius:8px;padding:8px 14px;box-shadow:0 2px 10px rgba(0,0,0,.25);max-width:60vw;text-align:center;font-size:13px}
 #route{padding:8px;border:1px solid var(--line);border-radius:6px;min-height:40px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
 @media(max-width:700px){body{flex-direction:column}#side{width:100%;max-width:none;height:45vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<div id="side">
 <h1>Road network from satellite imagery</h1>
 <div class="sub">U-Net road segmentation &rarr; skeleton &rarr; graph &rarr; junctions &middot; Gandhinagar study area</div>
 <h2>Network statistics</h2><table id="stats"></table>
 <h2>Intersections by type</h2><table id="types"></table>
 <div id="valbox"></div>
 <h2>Shortest-path demo</h2>
 <div id="route" class="note">Click two points on the map. Each snaps to the nearest graph node and the shortest route over the extracted road graph is drawn.</div>
 <button onclick="clearRoute()">Clear route</button>
 <h2>Export</h2>
 <button onclick="dl(ROADS,'roads.geojson')">roads.geojson</button><button onclick="dl(INTER,'intersections.geojson')">intersections.geojson</button>
 <h2>Legend</h2>
 <div class="note"><span class="dot" style="background:#ffd400"></span>road segment &nbsp; <span class="dot" style="background:#ff3b30"></span>degree 3 &nbsp; <span class="dot" style="background:#ff9500"></span>degree 4 &nbsp; <span class="dot" style="background:#af52de"></span>degree 5+</div>
 <p class="note">Predictions are model output, not surveyed truth. OSM reference points have their own errors and omissions. Imagery &copy; Esri; road reference &copy; OpenStreetMap contributors (ODbL).</p>
</div>
<div id="map"></div>
<div id="hint">Click a start point on the map, then a destination: the shortest route over the extracted road graph will be drawn.</div>
<script>__LEAFLET_JS__</script>
<script>
if(typeof L==='undefined'){document.getElementById('hint').innerHTML='<b>Map library did not load.</b> Re-run make_dashboard.py (it needs geospatial/vendor/leaflet.js).';throw new Error('Leaflet missing');}
const ROADS=__ROADS__, INTER=__INTER__, STATS=__STATS__, VAL=__VAL__, VALRES=__VALRES__, BOUNDS=__BOUNDS__;
const map=L.map('map').setView([(BOUNDS[1]+BOUNDS[3])/2,(BOUNDS[0]+BOUNDS[2])/2],16);
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics'}).addTo(map);
const bb=[[BOUNDS[1],BOUNDS[0]],[BOUNDS[3],BOUNDS[2]]]; map.fitBounds(bb);
const maskLayer=L.imageOverlay('mask_overlay.png',bb,{opacity:.65});
const roads=L.geoJSON(ROADS,{style:{color:'#ffd400',weight:3},onEachFeature:(f,l)=>l.bindTooltip('Road segment<br>length '+f.properties.length_m+' m<br>est. width '+f.properties.width_m+' m')}).addTo(map);
const col=d=>d>=5?'#af52de':d==4?'#ff9500':'#ff3b30';
const inter=L.geoJSON(INTER,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:6,color:'#fff',weight:1.5,fillColor:col(f.properties.degree),fillOpacity:1}),
 onEachFeature:(f,l)=>l.bindTooltip('<b>'+f.properties.type+'</b>'+(f.properties.roundabout?'<br>collapsed roundabout':''))}).addTo(map);
const overlays={'Road mask (model output)':maskLayer,'Road graph':roads,'Intersections':inter};
if(VAL){const vc={TP:'#34c759',FP:'#ff3b30',FN:'#0a84ff'};
 overlays['OSM check: green=match, red=extra, blue=missed']=L.geoJSON(VAL,{pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:5,color:'#000',weight:1,fillColor:vc[f.properties.status],fillOpacity:.95}),
 onEachFeature:(f,l)=>l.bindTooltip({TP:'Matches an OSM junction',FP:'Predicted, no OSM junction nearby (may be an OSM omission)',FN:'OSM junction not predicted'}[f.properties.status])});}
L.control.layers({'Satellite':sat},overlays,{collapsed:false}).addTo(map);

// ---- stats
const fmt=(k,v)=>'<tr><td>'+k+'</td><td>'+v+'</td></tr>';
document.getElementById('stats').innerHTML=
 fmt('Area',STATS.mosaic_area_km2+' km²')+fmt('Total road length',STATS.total_road_length_km+' km')+fmt('Road density',STATS.road_density_km_per_km2+' km/km²')+
 fmt('Graph nodes / edges',STATS.nodes+' / '+STATS.edges)+fmt('Intersections (degree ≥ 3)',STATS.intersections_degree_ge3)+fmt('Dead ends',STATS.dead_ends)+
 fmt('Mean node degree',STATS.mean_node_degree)+fmt('Connected components',STATS.connected_components)+fmt('Largest component (nodes)',(STATS.largest_component_share_of_nodes*100).toFixed(0)+'%');
document.getElementById('types').innerHTML=Object.entries(STATS.intersections_by_type).map(([k,v])=>fmt(k,v)).join('')+fmt('Roundabouts collapsed',STATS.roundabouts_collapsed);
if(VALRES){const t=VALRES.tolerances;const keys=Object.keys(t);let h='<h2>OSM validation</h2><table>';
 keys.forEach(k=>{const r=t[k];h+=fmt('within '+parseFloat(k)+' m: P / R / F1',r.precision.toFixed(2)+' / '+r.recall.toFixed(2)+' / '+r.f1.toFixed(2));});
 if(VALRES.median_offset_m)h+=fmt('median imagery–OSM offset',VALRES.median_offset_m.magnitude.toFixed(1)+' m');
 h+='</table><div class="note">'+VALRES.n_predicted+' predicted vs '+VALRES.n_reference+' OSM junctions. OSM is incomplete, so precision is a lower bound.</div>';
 document.getElementById('valbox').innerHTML=h;}

// ---- routing over the extracted graph
const nodes=new Map(), adj=new Map();
ROADS.features.forEach((f,i)=>{const p=f.properties,c=f.geometry.coordinates;
 nodes.set(p.u,c[0]);nodes.set(p.v,c[c.length-1]);
 if(!adj.has(p.u))adj.set(p.u,[]);if(!adj.has(p.v))adj.set(p.v,[]);
 adj.get(p.u).push([p.v,p.length_m,i,false]);adj.get(p.v).push([p.u,p.length_m,i,true]);});
function nearest(ll){let best=null,bd=1e18;const k=Math.cos(ll.lat*Math.PI/180);
 nodes.forEach((c,id)=>{const dx=(c[0]-ll.lng)*k,dy=c[1]-ll.lat,d=dx*dx+dy*dy;if(d<bd){bd=d;best=id;}});return best;}
function dijkstra(s,t){const dist=new Map([[s,0]]),prev=new Map(),done=new Set();
 while(true){let u=null,ud=1e18;dist.forEach((d,n)=>{if(!done.has(n)&&d<ud){ud=d;u=n;}});
  if(u===null)return null;if(u===t)break;done.add(u);
  (adj.get(u)||[]).forEach(([v,w,ei,rev])=>{const nd=ud+w;if(!dist.has(v)||nd<dist.get(v)){dist.set(v,nd);prev.set(v,[u,ei,rev]);}});}
 const path=[];let cur=t;while(cur!==s){const [p,ei,rev]=prev.get(cur);path.push([ei,rev]);cur=p;}return {len:dist.get(t),edges:path.reverse()};}
let rl=L.layerGroup().addTo(map),clicks=[];
const HINT=document.getElementById('hint');
function clearRoute(){rl.clearLayers();clicks=[];document.getElementById('route').innerHTML='Click two points on the map.';HINT.innerHTML='Click a start point on the map, then a destination.';}
map.on('click',e=>{if(clicks.length==2){clearRoute();}
 const id=nearest(e.latlng),c=nodes.get(id);clicks.push(id);
 L.circleMarker([c[1],c[0]],{radius:8,color:'#fff',weight:2,fillColor:clicks.length==1?'#34c759':'#0a84ff',fillOpacity:1}).addTo(rl);
 if(clicks.length==1)HINT.innerHTML='Start set (green). Now click the destination.';
 if(clicks.length==2){const r=dijkstra(clicks[0],clicks[1]);const box=document.getElementById('route');
  if(!r){box.innerHTML=HINT.innerHTML='<b>No route.</b> These two points are in different connected parts of the extracted network (a break in the road mask). Click again to retry.';return;}
  r.edges.forEach(([ei,rev])=>{let cs=ROADS.features[ei].geometry.coordinates.map(c=>[c[1],c[0]]);L.polyline(cs,{color:'#0a84ff',weight:6,opacity:.9}).addTo(rl);});
  box.innerHTML='<b>Route found</b><br>'+(r.len/1000).toFixed(2)+' km over '+r.edges.length+' road segments';HINT.innerHTML='<b>Route found: '+(r.len/1000).toFixed(2)+' km</b> over '+r.edges.length+' road segments. Click anywhere to start a new route.';}});
function dl(obj,name){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(obj)],{type:'application/json'}));a.download=name;a.click();}
</script></body></html>
"""


def lambda_free(x):
    return x


def load(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/gandhinagar")
    args = ap.parse_args()
    d = args.dir
    roads = load(os.path.join(d, "roads.geojson"))
    inter = load(os.path.join(d, "intersections.geojson"))
    stats = load(os.path.join(d, "stats.json"))
    if not (roads and inter and stats):
        raise SystemExit("Run graph_extraction/build_network.py first (roads/intersections/stats missing).")
    val = load(os.path.join(d, "validation.geojson"))
    valres = load(os.path.join(d, "validation_results.json"))
    vend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    leaflet_js = open(os.path.join(vend, "leaflet.js"), encoding="utf-8").read()
    leaflet_css = open(os.path.join(vend, "leaflet.css"), encoding="utf-8").read()
    html = (TEMPLATE
            .replace("__ROADS__", json.dumps(roads, separators=(",", ":")))
            .replace("__INTER__", json.dumps(inter, separators=(",", ":")))
            .replace("__STATS__", json.dumps(stats))
            .replace("__VALRES__", json.dumps(valres))
            .replace("__VAL__", json.dumps(val))
            .replace("__BOUNDS__", json.dumps(stats["bounds_west_south_east_north"]))
            .replace("__LEAFLET_CSS__", lambda_free(leaflet_css))
            .replace("__LEAFLET_JS__", lambda_free(leaflet_js)))
    out = os.path.join(d, "dashboard.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB). Open it in Chrome (keep mask_overlay.png next to it).")
    if val is None:
        print("Note: validation.geojson not found - run geospatial/validate_osm.py, then this script again, to add the OSM layer.")


if __name__ == "__main__":
    main()
