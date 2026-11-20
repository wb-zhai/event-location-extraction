#!/usr/bin/env python3
"""Read-only, resumable audit of the global V4 extraction run.

Uses only Python's standard library and the authenticated gcloud CLI.
Downloads one complete shard at a time with checksum-verified gcloud storage cp.
Never modifies GCS, the production database, or source prediction files.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

VERSION = '1.0'
FIELDS = ('event_type','grounding_quote','event_location','event_location_admin_level','event_time','time_status','severity')
ENUMS = {
    'event_location_admin_level': {'country','state','county','city','district','not_stated'},
    'time_status': {'past','ongoing','forecast','not_stated'},
    'severity': {'low','medium','high','extreme','not_stated'},
}
MISSING = {'', 'not_stated'}


def run(*args):
    p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{p.stderr[-4000:]}")
    return p.stdout


def inventory(prefix, kind, expected):
    prefix = prefix.rstrip('/')
    if prefix.startswith('gs://'):
        names = run('gcloud','storage','ls','--recursive',prefix).splitlines()
    else:
        names = [str(p) for p in Path(prefix).rglob('*.jsonl')]
    pattern = re.compile(r'^(manifest|shard)-(\d+)\.jsonl$')
    found = {}
    for name in names:
        m = pattern.match(name.rsplit('/',1)[-1])
        if m and m.group(1) == kind:
            i = int(m.group(2))
            if i in found:
                raise RuntimeError(f'Duplicate {kind} shard index {i}: {name}')
            found[i] = name
    missing = sorted(set(range(expected)) - set(found))
    extra = sorted(set(found) - set(range(expected)))
    if missing or extra:
        raise RuntimeError(f'{kind} inventory mismatch. Missing: {missing}; extra: {extra}')
    return found


def obtain(uri, target):
    if not uri.startswith('gs://'):
        return Path(uri), False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()  # Only our own scratch download, never a source file.
    run('gcloud','storage','cp',uri,str(target))
    return target, True


def normalize_id(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('missing or invalid article ID')
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError('missing or invalid article ID')


def language(value):
    aliases = {'en':'eng','english':'eng','fr':'fra','fre':'fra','french':'fra'}
    if not isinstance(value, str) or not value.strip():
        return 'unknown'
    value = value.strip().lower()
    return aliases.get(value,value)


def load_labels(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(obj, dict):
        for key in ('event_types','labels','events','risk_factors'):
            if key in obj and isinstance(obj[key], (list,dict)):
                obj = obj[key]
                break
    if isinstance(obj, dict):
        # Label -> description/cluster mapping, as in bona.v4.clusters.json.
        labels = list(obj.keys())
    elif isinstance(obj, list):
        labels = []
        for item in obj:
            if isinstance(item,str):
                labels.append(item)
            elif isinstance(item,dict):
                label = next((item[k] for k in ('name','label','event_type') if isinstance(item.get(k),str)),None)
                if label is None:
                    raise ValueError('Unknown ontology entry structure; supply a flat label list.')
                labels.append(label)
            else:
                raise ValueError('Unknown ontology entry structure.')
    else:
        raise ValueError('Ontology must contain a label list or mapping.')
    labels = [x.strip() for x in labels]
    if len(labels) != len(set(labels)) or any(not x for x in labels):
        raise ValueError('Duplicate or empty ontology labels.')
    return set(labels)


def iso_time(value):
    if value == 'not_stated':
        return 'not_stated', None
    if re.fullmatch(r'\d{4}',value):
        year = int(value)
        return ('year',year) if 1 <= year <= 9999 else ('invalid',None)
    if re.fullmatch(r'\d{4}-\d{2}',value):
        try:
            dt.date.fromisoformat(value+'-01')
            return 'month',int(value[:4])
        except ValueError:
            return 'invalid',None
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        try:
            dt.date.fromisoformat(value)
            return 'day',int(value[:4])
        except ValueError:
            return 'invalid',None
    if re.match(r'^\d{4}-\d{2}-\d{2}[T ]',value):
        try:
            dt.datetime.fromisoformat(value.replace('Z','+00:00'))
            return 'datetime',int(value[:4])
        except ValueError:
            return 'invalid',None
    return 'other',None


class Stats:
    def __init__(self, limit=10):
        self.c = collections.Counter()
        self.dist = collections.defaultdict(collections.Counter)
        self.lang = collections.defaultdict(collections.Counter)
        self.samples = collections.defaultdict(list)
        self.limit = limit
    def issue(self, key, sample):
        self.c[key] += 1
        if len(self.samples[key]) < self.limit:
            self.samples[key].append(sample)
    def merge(self, other):
        for k,v in other['counts'].items():
            if k == 'max_events_per_article': self.c[k] = max(self.c[k],v)
            else: self.c[k] += v
        for k,values in other['distributions'].items(): self.dist[k].update(values)
        for k,values in other['languages'].items(): self.lang[k].update(values)
        for k,values in other['samples'].items():
            self.samples[k] = (self.samples[k]+values)[:self.limit]
    def dump(self):
        return {'counts':dict(self.c),'distributions':{k:dict(v) for k,v in self.dist.items()},
                'languages':{k:dict(v) for k,v in self.lang.items()},'samples':dict(self.samples)}


def inspect_event(ev, article_id, index, labels, s, lang):
    s.c['raw_event_mentions'] += 1
    s.lang[lang]['raw_event_mentions'] += 1
    if not isinstance(ev,dict):
        s.issue('non_object_event',{'id':article_id,'index':index})
        return False
    ok = True
    for field in FIELDS:
        value = ev.get(field)
        if field not in ev:
            s.issue('missing_field.'+field,{'id':article_id,'index':index}); ok = False
        if not isinstance(value,str):
            s.issue('non_string_field.'+field,{'id':article_id,'index':index}); ok = False
        elif not value.strip():
            s.issue('blank_field.'+field,{'id':article_id,'index':index}); ok = False
    if not all(isinstance(ev.get(f),str) for f in FIELDS):
        return False
    for field in FIELDS:
        value = ev[field]
        if field in ENUMS and field != 'event_location_admin_level':
            s.dist[field][value] += 1
            if value not in ENUMS[field]:
                s.issue('invalid_enum.'+field,{'id':article_id,'value':value[:160]}); ok = False
    label = ev['event_type']
    s.dist['event_types'][label] += 1
    s.dist['event_types_by_language'][lang+'\t'+label] += 1
    if label not in labels:
        s.issue('off_ontology',{'id':article_id,'label':label[:160]}); ok = False
    quote = ev['grounding_quote']
    if quote.strip() in MISSING:
        s.issue('missing_grounding_quote',{'id':article_id,'index':index}); ok = False
    qlen = len(quote)
    bucket = '0' if qlen == 0 else ('1-49' if qlen < 50 else '50-99' if qlen < 100 else '100-249' if qlen < 250 else '250-499' if qlen < 500 else '500-999' if qlen < 1000 else '1000+')
    s.dist['quote_length'][bucket] += 1
    loc = ev['event_location']
    if loc.strip() in MISSING:
        s.c['location_not_stated'] += 1
    else:
        s.c['location_stated'] += 1
        nloc = len(loc.split(';'))
        nlev = len(ev['event_location_admin_level'].split(';'))
        s.dist['location_count'][str(nloc)] += 1
        if nloc != nlev:
            s.issue('location_admin_count_mismatch',{'id':article_id,'locations':nloc,'levels':nlev}); ok = False
    levels = [x.strip() for x in ev['event_location_admin_level'].split(';')]
    for level in levels:
        s.dist['admin_levels'][level] += 1
        if level not in ENUMS['event_location_admin_level']:
            s.issue('invalid_admin_level',{'id':article_id,'value':level[:160]}); ok = False
    precision,year = iso_time(ev['event_time'].strip())
    s.dist['time_precision'][precision] += 1
    if year is not None: s.dist['event_year'][str(year)] += 1
    if precision in {'invalid','other'}:
        s.issue('time_format_review',{'id':article_id,'value':ev['event_time'][:160]})
        if precision == 'invalid': ok = False
    if ok:
        s.c['schema_valid_event_mentions'] += 1
        s.lang[lang]['schema_valid_event_mentions'] += 1
    return ok


def connect(path, config):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA busy_timeout=30000')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS manifest (id TEXT PRIMARY KEY, language TEXT NOT NULL, shard INTEGER NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS observed (id TEXT PRIMARY KEY, shard INTEGER NOT NULL, event_count INTEGER NOT NULL, valid_event_count INTEGER NOT NULL, schema_ok INTEGER NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS progress (stage TEXT NOT NULL, shard INTEGER NOT NULL, stats TEXT NOT NULL, PRIMARY KEY(stage,shard))')
    old = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
    text = json.dumps(config,sort_keys=True)
    if old and old[0] != text:
        raise RuntimeError('This workdir belongs to a different audit. Use a new --workdir.')
    if not old:
        db.execute("INSERT INTO meta VALUES ('config',?)",(text,)); db.commit()
    return db


def progress(db, stage, shard):
    row = db.execute('SELECT stats FROM progress WHERE stage=? AND shard=?',(stage,shard)).fetchone()
    return json.loads(row[0]) if row else None


def manifest_shard(db, path, shard, limit):
    s = Stats(limit)
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            s.c['physical_lines'] += 1
            if not line.strip():
                s.c['blank_lines'] += 1; continue
            s.c['manifest_rows'] += 1
            try:
                row = json.loads(line)
                if not isinstance(row,dict): raise ValueError('record is not an object')
                article_id = normalize_id(row.get('id',row.get('article_id')))
            except (ValueError,TypeError,json.JSONDecodeError) as e:
                s.issue('invalid_manifest_record',{'shard':shard,'line':line_no,'error':str(e)[:160]}); continue
            lang = language(row.get('language'))
            s.dist['manifest_languages'][lang] += 1
            if lang not in {'eng','fra'}:
                s.issue('unexpected_manifest_language',{'id':article_id,'language':lang})
            if not isinstance(row.get('gcs_path',row.get('cloud_uri')),str) or not row.get('gcs_path',row.get('cloud_uri')):
                s.issue('missing_manifest_gcs_path',{'id':article_id})
            cur = db.execute('INSERT OR IGNORE INTO manifest VALUES (?,?,?)',(article_id,lang,shard))
            if cur.rowcount == 0:
                s.issue('duplicate_manifest_id',{'id':article_id,'shard':shard})
            else:
                s.c['unique_manifest_ids_added'] += 1
    return s


def output_shard(db, path, shard, labels, limit):
    s = Stats(limit)
    local = {r[0]:r[1] for r in db.execute('SELECT id,language FROM manifest WHERE shard=?',(shard,))}
    with path.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            s.c['physical_lines'] += 1
            if not line.strip():
                s.c['blank_lines'] += 1; continue
            s.c['output_rows'] += 1
            try:
                row = json.loads(line)
                if not isinstance(row,dict): raise ValueError('record is not an object')
            except (ValueError,json.JSONDecodeError) as e:
                s.issue('invalid_output_json_or_object',{'shard':shard,'line':line_no,'error':str(e)[:160]}); continue
            try:
                article_id = normalize_id(row.get('id'))
            except ValueError:
                s.issue('invalid_output_id',{'shard':shard,'line':line_no}); continue
            # The first record for an ID is the canonical audit record; duplicates are never merged.
            cur = db.execute('INSERT OR IGNORE INTO observed VALUES (?,?,?,?,?)',(article_id,shard,0,0,0))
            if cur.rowcount == 0:
                s.issue('duplicate_output_id',{'id':article_id,'shard':shard,'line':line_no}); continue
            s.c['unique_output_ids_added'] += 1
            lang = local.get(article_id)
            if lang is None:
                expected = db.execute('SELECT language,shard FROM manifest WHERE id=?',(article_id,)).fetchone()
                if expected:
                    lang = expected[0]
                    s.issue('output_shard_mismatch',{'id':article_id,'expected':expected[1],'actual':shard})
                else:
                    lang = 'unknown'
                    s.issue('unexpected_output_id',{'id':article_id,'shard':shard})
            s.lang[lang]['articles'] += 1
            preds = row.get('predictions')
            if not isinstance(preds,list):
                s.issue('invalid_predictions',{'id':article_id,'type':type(preds).__name__}); continue
            s.c['schema_valid_prediction_lists'] += 1
            n = len(preds)
            s.c['raw_event_mentions_in_lists'] += n
            s.lang[lang]['event_mentions_in_lists'] += n
            if n == 0:
                s.c['no_event_articles'] += 1
                s.lang[lang]['no_event_articles'] += 1
            else:
                s.c['with_event_articles'] += 1
                s.lang[lang]['with_event_articles'] += 1
            bucket = str(n) if n <= 20 else ('21-50' if n <= 50 else '51+')
            s.dist['events_per_article'][bucket] += 1
            s.c['max_events_per_article'] = max(s.c['max_events_per_article'],n)
            valid = 0
            all_ok = True
            fingerprints = set()
            for i,ev in enumerate(preds):
                ev_ok = inspect_event(ev,article_id,i,labels,s,lang)
                valid += int(ev_ok)
                all_ok = all_ok and ev_ok
                if isinstance(ev,dict):
                    fingerprint = json.dumps(ev,sort_keys=True,ensure_ascii=False)
                    if fingerprint in fingerprints:
                        s.issue('duplicate_event_payload_within_article',{'id':article_id,'index':i})
                    fingerprints.add(fingerprint)
            if all_ok:
                s.c['schema_valid_articles'] += 1
                s.lang[lang]['schema_valid_articles'] += 1
            db.execute('UPDATE observed SET event_count=?,valid_event_count=?,schema_ok=? WHERE id=?',(n,valid,int(all_ok),article_id))
    return s


def write_csv(path, headers, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)


def reconcile(db, workdir, stats, expected, shards, all_labels):
    n_manifest = db.execute('SELECT COUNT(*) FROM manifest').fetchone()[0]
    n_observed = db.execute('SELECT COUNT(*) FROM observed').fetchone()[0]
    missing_sql = 'SELECT m.id,m.shard,m.language FROM manifest m LEFT JOIN observed o ON o.id=m.id WHERE o.id IS NULL'
    extra_sql = 'SELECT o.id,o.shard FROM observed o LEFT JOIN manifest m ON m.id=o.id WHERE m.id IS NULL'
    missing = db.execute('SELECT COUNT(*) FROM ('+missing_sql+')').fetchone()[0]
    extra = db.execute('SELECT COUNT(*) FROM ('+extra_sql+')').fetchone()[0]
    if missing:
        write_csv(workdir/'missing_article_ids.csv',['id','manifest_shard','language'],db.execute(missing_sql))
    if extra:
        write_csv(workdir/'unexpected_article_ids.csv',['id','output_shard'],db.execute(extra_sql))
    coverage = []
    for row in db.execute('''SELECT m.language,COUNT(*),COUNT(o.id),
        COALESCE(SUM(o.event_count),0),COALESCE(SUM(o.valid_event_count),0),
        COALESCE(SUM(CASE WHEN o.event_count=0 AND o.schema_ok=1 THEN 1 ELSE 0 END),0),
        COALESCE(SUM(o.schema_ok),0)
        FROM manifest m LEFT JOIN observed o ON m.id=o.id GROUP BY m.language'''):
        coverage.append(dict(zip(['language','expected_articles','observed_articles','raw_events','schema_valid_events','no_event_articles','schema_valid_articles'],row)))
    by_shard = []
    for i in range(shards):
        m = progress(db,'manifest',i) or {}
        o = progress(db,'output',i) or {}
        a,b = m.get('counts',{}),o.get('counts',{})
        by_shard.append({'shard':i,'manifest_rows':a.get('manifest_rows',0),'output_rows':b.get('output_rows',0),
            'unique_manifest_ids_added':a.get('unique_manifest_ids_added',0),'unique_output_ids_added':b.get('unique_output_ids_added',0),
            'malformed_output':b.get('invalid_output_json_or_object',0),'duplicate_output_ids':b.get('duplicate_output_id',0),
            'raw_events':b.get('raw_event_mentions',0),'no_event_articles':b.get('no_event_articles',0)})
    write_csv(workdir/'shard_stats.csv',list(by_shard[0]),([r[k] for k in by_shard[0]] for r in by_shard))
    label_rows = []
    label_counts = stats.dist.get('event_types',{})
    for label in sorted(set(all_labels) | set(label_counts)):
        label_rows.append([label,label_counts.get(label,0),int(label in all_labels)])
    label_rows.sort(key=lambda r:(-r[1],r[0]))
    write_csv(workdir/'event_types.csv',['event_type','count','in_v4_ontology'],label_rows)
    lang_labels = stats.dist.get('event_types_by_language',{})
    write_csv(workdir/'event_types_by_language.csv',['language','event_type','count'],
              ([key.split('\t',1)[0],key.split('\t',1)[1],count] for key,count in sorted(lang_labels.items())))
    failures = []
    if n_manifest != expected: failures.append('manifest unique-ID count differs from expected')
    if missing: failures.append('missing manifest article IDs')
    if extra: failures.append('unexpected output article IDs')
    for key in ('duplicate_manifest_id','duplicate_output_id','invalid_manifest_record','invalid_output_json_or_object','invalid_output_id','invalid_predictions','non_object_event','off_ontology','time_format_review','location_admin_count_mismatch','invalid_admin_level'):
        if stats.c.get(key): failures.append(key)
    if any(k.startswith(('missing_field.','non_string_field.','blank_field.','invalid_enum.')) and v for k,v in stats.c.items()):
        failures.append('schema field validation errors')
    if stats.c.get('missing_manifest_gcs_path') or stats.c.get('unexpected_manifest_language'):
        failures.append('manifest metadata validation errors')
    if stats.c.get('schema_valid_event_mentions',0) != stats.c.get('raw_event_mentions',0):
        failures.append('some event records fail schema validation')
    if stats.c.get('unexpected_output_id') or stats.c.get('output_shard_mismatch'):
        failures.append('manifest language or shard assignment anomalies')
    def ratio(a,b): return round(100*a/b,4) if b else None
    metrics = {
        'manifest_coverage_percent':ratio(n_manifest-missing,n_manifest),
        'expected_manifest_coverage_percent':ratio(n_manifest-missing,expected),
        'no_event_rate_percent':ratio(stats.c.get('no_event_articles',0),stats.c.get('schema_valid_prediction_lists',0)),
        'events_per_analyzed_article':round(stats.c.get('raw_event_mentions',0)/stats.c.get('schema_valid_prediction_lists',1),4) if stats.c.get('schema_valid_prediction_lists') else None,
        'schema_valid_event_percent':ratio(stats.c.get('schema_valid_event_mentions',0),stats.c.get('raw_event_mentions',0)),
        'location_stated_percent':ratio(stats.c.get('location_stated',0),stats.c.get('raw_event_mentions',0)),
    }
    for row in coverage:
        row['coverage_percent'] = ratio(row['observed_articles'],row['expected_articles'])
        row['no_event_rate_percent'] = ratio(row['no_event_articles'],row['observed_articles'])
    write_csv(workdir/'language_coverage.csv',list(coverage[0]) if coverage else ['language'],
              ([row[k] for k in coverage[0]] for row in coverage) if coverage else [])
    report = {'audit_version':VERSION,'expected_manifest_rows':expected,'expected_shards':shards,
              'unique_manifest_ids':n_manifest,'unique_output_ids':n_observed,'missing_article_ids':missing,
              'unexpected_article_ids':extra,'coverage_by_language':coverage,'metrics':metrics,'canonical_labels':sorted(all_labels),
              'schema_gate_passed':not failures,'schema_gate_failures':sorted(set(failures)),
              'note':'Schema/coverage gate only. Grounding accuracy, extraction F1, geocoding quality and database ingestion are not established.',
              **stats.dump()}
    (workdir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    def fmt(n): return f'{n:,}'
    lines = ['GLOBAL V4 EXTRACTION AUDIT','',f'Expected manifest rows: {fmt(expected)}',f'Unique manifest IDs: {fmt(n_manifest)}',
             f'Unique output IDs: {fmt(n_observed)}',f'Missing IDs: {fmt(missing)}',f'Unexpected IDs: {fmt(extra)}',
             f'Output rows: {fmt(stats.c.get("output_rows",0))}',f'Duplicate output IDs: {fmt(stats.c.get("duplicate_output_id",0))}',
             f'Raw event mentions: {fmt(stats.c.get("raw_event_mentions",0))}',f'Schema-valid event mentions: {fmt(stats.c.get("schema_valid_event_mentions",0))}',
             f'Articles with events: {fmt(stats.c.get("with_event_articles",0))}',f'No-event articles: {fmt(stats.c.get("no_event_articles",0))}',
             f'Schema-valid articles: {fmt(stats.c.get("schema_valid_articles",0))}',
             f'Manifest ID coverage: {metrics["manifest_coverage_percent"]}%',
             f'No-event rate (valid prediction lists): {metrics["no_event_rate_percent"]}%',
             f'Events per analyzed article: {metrics["events_per_analyzed_article"]}',
             f'Schema-valid event rate: {metrics["schema_valid_event_percent"]}%',
             f'Location stated rate: {metrics["location_stated_percent"]}%',
             '', 'LANGUAGE COVERAGE']
    for r in coverage:
        lines.append(f'{r["language"]}: expected={fmt(r["expected_articles"])}, observed={fmt(r["observed_articles"])}, events={fmt(r["raw_events"])}, no-event={fmt(r["no_event_articles"])}')
    lines += ['', 'SCHEMA / COVERAGE GATE: '+('PASS' if not failures else 'REVIEW REQUIRED')]
    lines += sorted(set(failures)) if failures else ['No blocking structural or coverage anomalies detected.']
    lines += ['','This is not an extraction-accuracy or geocoding-quality evaluation.','See report.json for all field, label, time, severity, and language distributions.']
    text = '\n'.join(lines)+'\n'
    (workdir/'summary.txt').write_text(text,encoding='utf-8')
    print('\n'+text)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='gs://zhai-risk-factor-extraction/extraction/manifests/global-v4-relevant-en-fr')
    p.add_argument('--output',default='gs://zhai-risk-factor-extraction/extraction/output/global-v4-relevant-en-fr')
    p.add_argument('--ontology',required=True,help='Canonical V4 label JSON or bona.v4.clusters.json')
    p.add_argument('--workdir',default='outputs/geography/global_v4_audit')
    p.add_argument('--expected',type=int,default=34390924)
    p.add_argument('--shards',type=int,default=200)
    p.add_argument('--expected-labels',type=int,default=48)
    p.add_argument('--sample-limit',type=int,default=10)
    p.add_argument('--keep-downloads',action='store_true')
    args = p.parse_args()
    workdir = Path(args.workdir).resolve(); workdir.mkdir(parents=True,exist_ok=True)
    labels = load_labels(args.ontology)
    if len(labels) != args.expected_labels:
        raise RuntimeError(f'Loaded {len(labels)} labels, expected {args.expected_labels}. Check the ontology before running.')
    print(f'Loaded {len(labels)} canonical labels.')
    manifests = inventory(args.manifest,'manifest',args.shards)
    outputs = inventory(args.output,'shard',args.shards)
    config = {'version':VERSION,'manifest':manifests,'output':outputs,'ontology_sha256':hashlib.sha256(Path(args.ontology).read_bytes()).hexdigest(),
              'expected':args.expected,'shards':args.shards,'expected_labels':args.expected_labels}
    db = connect(workdir/'audit.sqlite',config)
    try:
        for stage,sources in [('manifest',manifests),('output',outputs)]:
            print('\n'+stage.upper()+' AUDIT',flush=True)
            for i in range(args.shards):
                if progress(db,stage,i) is not None:
                    print(f'[{stage} {i+1}/{args.shards}] already complete',flush=True); continue
                target = workdir/'scratch'/f'{stage}-{i:03d}.jsonl'
                print(f'[{stage} {i+1}/{args.shards}] downloading and analyzing',flush=True)
                path,owned = obtain(sources[i],target)
                try:
                    with db:
                        if stage == 'manifest': result = manifest_shard(db,path,i,args.sample_limit)
                        else: result = output_shard(db,path,i,labels,args.sample_limit)
                        db.execute('INSERT INTO progress VALUES (?,?,?)',(stage,i,json.dumps(result.dump(),ensure_ascii=False)))
                    print(f'  {dict(result.c)}',flush=True)
                except Exception:
                    print(f'Failed shard retained at {path} when available. This shard was rolled back and can be retried.',file=sys.stderr)
                    raise
                else:
                    if owned and not args.keep_downloads: path.unlink()
                if (i+1)%10 == 0:
                    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        stats = Stats(args.sample_limit)
        for row in db.execute('SELECT stats FROM progress'):
            stats.merge(json.loads(row[0]))
        report = reconcile(db,workdir,stats,args.expected,args.shards,labels)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        print('Reports:',workdir)
        return 0 if report['schema_gate_passed'] else 2
    finally:
        db.close()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError,ValueError,sqlite3.Error,OSError) as e:
        print('ERROR:',e,file=sys.stderr)
        raise SystemExit(1)
