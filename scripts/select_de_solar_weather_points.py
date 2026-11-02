"""Capacity-based solar weather points from the full public MaStR solar inventory.

Downloads only solar ZIP entries using byte ranges. Rooftop installations with
private coordinates are placed at public postcode centroids (GeoNames). No
forecast outcomes enter selection. Raw inventory stays outside the repository.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, math, re, struct, subprocess, zipfile, zlib
from datetime import datetime, UTC
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
from lxml import etree
from sklearn.cluster import KMeans

REGISTRY_URL='https://download.marktstammdatenregister.de/Gesamtdatenexport_20261102_26.2.zip'
ZIP_SIZE=3267222849


def registry_directory(root, url):
    tail_path=root/'zip-tail.bin';meta_path=root/'registry_download.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['url']!=url:raise ValueError('Use a fresh cache directory for another registry snapshot')
        return entries(tail_path.read_bytes(),meta['size'])
    if tail_path.exists() and url==REGISTRY_URL:
        size=ZIP_SIZE  # Existing validated cache from this dated study.
    else:
        with requests.Session().get(url,headers={'Range':'bytes=-131072','Accept-Encoding':'identity'},timeout=30,stream=True) as r:
            if r.status_code!=206:raise ValueError('Registry server must support byte ranges')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            if not match or int(match[2])-int(match[1])+1>131072:raise ValueError('Unexpected ZIP directory range')
            body=r.content
            if len(body)!=int(match[2])-int(match[1])+1:raise ValueError('Truncated ZIP directory')
            tail_path.write_bytes(body);size=int(match[3])
    meta_path.write_text(json.dumps(dict(url=url,size=size,tail_sha256=hashlib.sha256(tail_path.read_bytes()).hexdigest(),
                                         frozen_at_utc=datetime.now(UTC).isoformat()),indent=2))
    return entries(tail_path.read_bytes(),size)


def entries(tail: bytes, size: int):
    z=zipfile.ZipFile(io.BytesIO(tail))
    return [dict(name=i.filename,offset=i.header_offset+size-len(tail),compressed=i.compress_size,
                 uncompressed=i.file_size,crc=i.CRC) for i in z.infolist() if i.filename.startswith('EinheitenSolar_')]


def download_entry(info,root,url):
    path=root/'compressed'/info['name'].replace('.xml','.bin');path.parent.mkdir(parents=True,exist_ok=True)
    required=info['compressed']+512
    for attempt in range(4):
        have=path.stat().st_size if path.exists() else 0
        if have>=required:return path
        part=path.with_suffix('.part')
        cmd=['curl','--http1.1','-fsSL','-H','Accept-Encoding: identity','--range',
             f"{info['offset']+have}-{info['offset']+required-1}",'--max-time','240',url,'-o',str(part)]
        result=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        if part.exists():
            with path.open('ab') as out:out.write(part.read_bytes())
            part.unlink()
        if path.exists() and path.stat().st_size>=required:
            if path.read_bytes()[:4]!=b'PK\x03\x04':raise ValueError('Unexpected ZIP range response')
            return path
        print(info['name'],'download retry',attempt+1,'bytes',path.stat().st_size if path.exists() else 0,flush=True)
    raise RuntimeError(f"Cannot finish registry entry {info['name']}")


def postcodes(path):
    with zipfile.ZipFile(path) as z:
        frame=pd.read_csv(z.open('DE.txt'),sep='\t',header=None,dtype={1:str})
    result={}
    for code,g in frame.groupby(1):
        result[str(code).zfill(5)]=(float(g[9].mean()),float(g[10].mean()))
    return result


def unit_record(values,centroids):
    """Return coarse public geography and capacity; never expose operator addresses."""
    if values.get('Land')!='84' or values.get('EinheitSystemstatus')!='472':return None
    try:
        gross=float(values.get('Bruttoleistung') or 0)/1000
        net=float(values.get('Nettonennleistung') or 0)/1000
        weight=net if net>0 else gross
    except ValueError:return None
    if not math.isfinite(weight) or weight<=0:return None
    coord='reported'
    try:lat,lon=float(values.get('Breitengrad') or 0),float(values.get('Laengengrad') or 0)
    except ValueError:lat=lon=0
    if not (47<=lat<=55.5 and 5<=lon<=16):
        location=centroids.get((values.get('Postleitzahl') or '').zfill(5))
        if location is None:return dict(unlocated=True,net_mw=weight,gross_mwp=gross)
        lat,lon=location;coord='postcode'
    return dict(lat=lat,lon=lon,weight=weight,gross=gross,coordinate_source=coord,
                commission=(values.get('Inbetriebnahmedatum') or '')[:10],
                registered=(values.get('Registrierungsdatum') or '')[:10],
                retirement=(values.get('DatumEndgueltigeStilllegung') or values.get('DatumEndgueltigeStilllegung_nv') or '')[:10],
                status=values.get('EinheitBetriebsstatus'),state=values.get('Bundesland','unknown'))


def parse_entry(info,root,url,centroids,as_of):
    out=root/'aggregated'/info['name'].replace('.xml','.json');out.parent.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(dict(as_of=as_of,crc=info['crc'],geocoding=centroids),sort_keys=True).encode()).hexdigest()
    if out.exists():
        cached=json.loads(out.read_text())
        if cached.get('cache_signature')==signature:return cached
    path=download_entry(info,root,url)
    # Cell values preserve capacity-weighted coordinates; the 0.02-degree bins
    # reduce millions of private units to coarse public geography for clustering.
    grids={k:collections.defaultdict(lambda:[0.,0.,0.,0.,0.]) for k in ['current','preperiod']}
    stats=collections.Counter();state=collections.defaultdict(float)
    parser=etree.XMLPullParser(events=('end',),tag='EinheitSolar',huge_tree=True)
    crc=0;length=0;decoder=zlib.decompressobj(-15)
    with path.open('rb') as handle:
        header=handle.read(30);fields=struct.unpack('<4s5H3I2H',header)
        handle.seek(30+fields[-2]+fields[-1]);remaining=info['compressed']
        while remaining:
            block=handle.read(min(65536,remaining));remaining-=len(block)
            if not block:raise ValueError('Truncated compressed inventory')
            xml=decoder.decompress(block);crc=zlib.crc32(xml,crc);length+=len(xml);parser.feed(xml)
            for _,element in parser.read_events():
                values={child.tag:child.text for child in element}
                row=unit_record(values,centroids);stats['xml_units']+=1
                if row:
                    if row.get('unlocated'):
                        stats['unlocated_net_mw']+=row['net_mw'];stats['unlocated_units']+=1
                    else:
                        is_current=row['status'] in {'35','37'}
                        is_prior=(row['commission'] and row['commission']<=as_of and row['registered'] and row['registered']<=as_of
                                  and (row['status'] in {'35','37'} or (row['retirement'] and row['retirement']>as_of)))
                        for group,eligible in [('current',is_current),('preperiod',is_prior)]:
                            if eligible:
                                key=f"{round(row['lat']/0.02)},{round(row['lon']/0.02)}"
                                cell=grids[group][key];w=row['weight']
                                cell[0]+=w;cell[1]+=w*row['lat'];cell[2]+=w*row['lon'];cell[3]+=1;cell[4]+=row['gross']
                                stats[group+'_units']+=1;stats[group+'_net_mw']+=w
                                stats[group+'_'+row['coordinate_source']+'_net_mw']+=w
                        if is_current:state[row['state']]+=row['gross']
                element.clear()
                while element.getprevious() is not None:del element.getparent()[0]
    tail=decoder.flush();crc=zlib.crc32(tail,crc);length+=len(tail)
    if crc!=info['crc'] or length!=info['uncompressed']:raise ValueError('Registry ZIP CRC/length mismatch')
    result=dict(file=info['name'],cache_signature=signature,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stats=dict(stats),
                states=dict(state),grids={g:dict(v) for g,v in grids.items()})
    out.write_text(json.dumps(result));print('parsed',info['name'],'units',stats['xml_units'],flush=True)
    return result


def select_points(grid,n=4):
    weights=grid.net_mw.to_numpy();lat=grid.latitude.to_numpy();lon=grid.longitude.to_numpy()
    scale=np.array([111.32*math.cos(math.radians(51)),111.32])
    xy=np.column_stack([lon-10,lat-51])*scale
    model=KMeans(n_clusters=n,n_init=40,random_state=42).fit(xy,sample_weight=weights)
    rows=[]
    for cluster in range(n):
        mask=model.labels_==cluster;w=weights[mask];la=np.average(lat[mask],weights=w);lo=np.average(lon[mask],weights=w)
        era_lat,era_lon=round(la*4)/4,round(lo*4)/4
        # Same ERA5-reference -> nominal GFS-grid convention as the original selection.
        gfs_lat,gfs_lon=round(era_lat/.1171875)*.1171875,round(era_lon/.1171875)*.1171875
        rows.append(dict(country='DE',type='solar',reference_lat=la,reference_lon=lo,era5_lat=era_lat,era5_lon=era_lon,
                         gfs_lat=gfs_lat,gfs_lon=gfs_lon,cluster_capacity_mw=w.sum(),cluster_share=w.sum()/weights.sum(),
                         cluster_assets=int(grid.loc[mask,'units'].sum()),selection_method='full_mastr_capacity_weighted_kmeans_k4'))
    frame=pd.DataFrame(rows).sort_values(['reference_lat','reference_lon']).reset_index(drop=True)
    frame.insert(2,'point',np.arange(1,n+1));return frame


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--url',default=REGISTRY_URL);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--as-of',default='2026-07-23');args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    infos=registry_directory(root,args.url)
    postcode_path=root/'DE-postcodes.zip'
    if not postcode_path.exists():
        r=requests.Session().get('https://download.geonames.org/export/zip/DE.zip',timeout=30)
        r.raise_for_status();postcode_path.write_bytes(r.content)
    centroids=postcodes(postcode_path)
    stats=collections.Counter();states=collections.Counter();grids={g:collections.defaultdict(lambda:np.zeros(5)) for g in ['current','preperiod']};audits=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(parse_entry,i,root,args.url,centroids,args.as_of) for i in infos]
        for future in as_completed(futures):
            r=future.result();stats.update(r['stats']);states.update(r['states']);audits.append({k:r[k] for k in ['file','sha256','stats']})
            for group,cells in r['grids'].items():
                for key,values in cells.items():grids[group][key]+=values
            print('completed',len(audits),'/',len(infos),'current_MW',round(stats['current_net_mw']),flush=True)
            (root/'selection_progress.json').write_text(json.dumps(dict(completed=len(audits),files=len(infos),stats=dict(stats))))
    for group,cells in grids.items():
        frame=pd.DataFrame([dict(net_mw=v[0],latitude=v[1]/v[0],longitude=v[2]/v[0],units=int(v[3]),gross_mwp=v[4]) for v in cells.values()])
        frame.to_parquet(root/f'{group}_capacity_grid.parquet',index=False)
        select_points(frame).to_csv(root/f'{group}_points.csv',index=False)
    (root/'selection_audit.json').write_text(json.dumps(dict(registry_url=args.url,registry_date=re.search(r'_(\d{8})_',args.url).group(1) if re.search(r'_(\d{8})_',args.url) else 'see URL',as_of=args.as_of,
        postcode_sha256=hashlib.sha256(postcode_path.read_bytes()).hexdigest(),
        note='Current export filtered by commission/registration/retirement dates; not a certified as-issued July inventory. Coordinates use public postcode centroids where exact locations are private.',
        geocoding='https://download.geonames.org/export/zip/DE.zip (GeoNames, CC BY 4.0)',stats=dict(stats),states_gross_mwp=dict(states),entries=audits),indent=2))

if __name__=='__main__':main()
