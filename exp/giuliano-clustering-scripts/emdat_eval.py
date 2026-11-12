from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import date
import pandas as pd
import matplotlib

def plot_metrics(metrics, out):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics).copy()
    df = df.set_index("class")

    # 1. Precision, recall and F1 by class
    score_cols = ["precision", "recall", "f1"]
    ax = df[score_cols].plot(
        kind="bar",
        figsize=(10, 5),
        ylim=(0, 1),
        width=0.8
    )
    ax.set_title("EM-DAT event detection performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Event class")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "classification_metrics.png", dpi=200)
    plt.close()

    # 2. Average temporal overlap by class
    overlap = pd.to_numeric(
        df["average_temporal_overlap_iou"], errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(overlap.index, overlap.values)
    ax.set_ylim(0, 1)
    ax.set_title("Average temporal overlap by class")
    ax.set_ylabel("Temporal IoU")
    ax.set_xlabel("Event class")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, overlap.values):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.savefig(out / "temporal_overlap.png", dpi=200)
    plt.close()

    # 3. Number of predictions, reference events and matched events
    count_cols = ["predictions", "emdat_events", ]
    ax = df[count_cols].plot(
        kind="bar",
        figsize=(10, 5),
        width=0.8
    )
    ax.set_title("Predicted and EM-DAT event counts")
    ax.set_ylabel("Number of events")
    ax.set_xlabel("Event class")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Predictions", "EM-DAT events", "Matched events"])
    plt.tight_layout()
    plt.savefig(out / "event_counts.png", dpi=200)
    plt.close()

    print(f"Plots saved in: {out.resolve()}")
    
    
EMDAT_CLASSES = {"flooding": "Flood",
                "drought": "Drought",
                "earthquake": "Earthquake",
                "fire": "Wildfire",
                "cyclone": "Storm",
                "abnormal rainfall": "Storm"}

def date_bound(row, prefix, is_end):
    y, m, d = (row.get(f"{prefix} Year"), row.get(f"{prefix} Month"), row.get(f"{prefix} Day"))
    if pd.isna(y): return pd.NaT
    y = int(y)
    if pd.isna(m): return pd.Timestamp(y, 12, 31) if is_end else pd.Timestamp(y, 1, 1)
    m = int(m)
    if pd.isna(d):
        return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) if is_end else pd.Timestamp(y, m, 1)
    return pd.Timestamp(y, m, int(d))

def pred_class(value):
    return EMDAT_CLASSES.get(str(value))

# def max3(preds, refs):
#     edges = []
#     for p in preds:
#         pstart, pend = p["start"], p["end"]
#         edges.append([j for j, r in enumerate(refs)
#                     if max(pstart, r["start"]) <= min(pend, r["end"])])
          
#     ref_to_pred = {}
#     def visit(i, seen):
#         for j in edges[i]:
#             if j in seen: continue
#             seen.add(j)
#             if j not in ref_to_pred or visit(ref_to_pred[j], seen):
#                 ref_to_pred[j] = i
#                 return True
#         return False
#     for i in range(len(preds)): visit(i, set())
    
#     total_match = set([x for xx in edges for x in xx])
#     return [(i, j, len(total_match)/len(ref_to_pred)) for j, i in ref_to_pred.items()]


def max3(preds, refs):
    matches = []
    matched_preds = set()
    matched_refs = set()

    for i, pred in enumerate(preds):
        for j, ref in enumerate(refs):

            # Inclusive interval overlap
            overlap_start = max(pred["start"], ref["start"])
            overlap_end = min(pred["end"], ref["end"])

            if overlap_start > overlap_end:
                continue

            intersection = (overlap_end - overlap_start).days + 1

            pred_duration = (
                pred["end"] - pred["start"]
            ).days + 1

            ref_duration = (
                ref["end"] - ref["start"]
            ).days + 1

            union = pred_duration + ref_duration - intersection
            iou = intersection / union if union > 0 else 0

            matches.append((i, j, iou))
            matched_preds.add(i)
            matched_refs.add(j)
    
    tp = len(matched_preds)
    detected_refs = len(matched_refs)

    return matches, tp, detected_refs

def prepare_excel(args): 
    e = pd.read_excel(args.emdat, sheet_name='EM-DAT Data')
    
    e['country'] = e['Country'].astype(str)
    e['code'] = e['ISO'].astype(str)
    e['class'] = e['Disaster Type'].astype(str)
    e['subclass'] = e['Disaster Subtype'].astype(str)
    e['start'] = e.apply(lambda r: date_bound(r, 'Start', False), axis=1)
    e['end'] = e.apply(lambda r: date_bound(r, 'End', True), axis=1)
    e['start'] = pd.to_datetime(e['start'], errors='coerce').dt.normalize()
    e['end'] = pd.to_datetime(e['end'], errors='coerce').dt.normalize()
        
    return e
def prepare_parquet(args):
    p = pd.read_parquet(args.predictions)
    required = [args.prediction_country_col, args.prediction_class_col,
                    args.prediction_start_col, args.prediction_end_col]
    missing = [c for c in required if c not in p.columns]
    if missing: raise ValueError(f"Prediction parquet missing columns: {missing}. Available: {list(p.columns)}")
    p=p.copy()
    p['country'] = p[args.prediction_country_col].astype(str)
    p['code'] = p['adm_code'].astype(str)
    p['class'] = p[args.prediction_class_col].astype(str)
    p['adm_level'] = p['adm_level'].astype(str)
    p['start'] = pd.to_datetime(p[args.prediction_start_col], errors='coerce').dt.normalize()
    p['end'] = pd.to_datetime(p[args.prediction_end_col], errors='coerce').dt.normalize()
    return p.copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True, help='Cluster parquet from the clustering script')
    ap.add_argument('--emdat', required=True, help='EM-DAT Excel workbook')
    ap.add_argument('--output-dir', default='data/evaluation_emdat')
    ap.add_argument('--start-year', type=int, default=2016)
    ap.add_argument('--end-year', type=int, default=2026)
    ap.add_argument('--prediction-country-col', default='country')
    ap.add_argument('--prediction-class-col', default='event_type')
    ap.add_argument('--prediction-start-col', default='cluster_start_date')
    ap.add_argument('--prediction-end-col', default='cluster_end_date')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = prepare_parquet(args)
    gold = prepare_excel(args)
    pairs=[]; metrics=[]
    classes=['flooding','drought','earthquake','cyclone']
    
    codes = gold['code'].to_list()
    matching_codes= [code for code in codes if sum(pred['code'].str.startswith(code)) != 0]
    gold = gold[gold['code'].isin(matching_codes)]
    pred = pred[pred['code'].isin(matching_codes)]
            
    # dictionary = pd.read_excel("/Users/giulianomartinelli/Downloads/zhai_annotation/data/adm_city_dictionary.csv", sheet_name='dd')
        
    total_fp = 0
    total_fn = 0
    total_detected_refs = 0
    for cls in classes:
        precision, recall, f1 = [],[],[] 
        overlaps=[]
        total_pred = 0
        tps = 0
        pred_countries=set(pred.loc[pred['class'].eq(cls),'code'])
        gold_countries=set(gold.loc[gold['class'].eq(pred_class(cls)),'code'])
        
        new_pred_countries = []
        preds = 0
        for code in codes:
            for pred_country in pred_countries:
                if pred_country.startswith(code):
                    new_pred_countries.append(code)
                    continue
        pred_countries = set(new_pred_countries)
        countries = pred_countries | gold_countries
        for code in countries:
            predicted_for_country_class=pred[(pred['class']==cls)&(pred['code'].str.startswith(code))].to_dict('records')
            gold_for_country_class=gold[(gold['class']==pred_class(cls))&(gold['code']==code)].to_dict('records')
                       
            if cls == 'cyclone':
                gold_for_country_class=gold[(gold['subclass'].str.contains(cls))&(gold['code']==code)].to_dict('records')
            # if cls == 'flooding' and predicted_for_country_class and False:
            #     print(len(predicted_for_country_class), len(gold_for_country_class))
            

            total_pred += len(predicted_for_country_class)
            matches, positives, detected_refs = max3(predicted_for_country_class, gold_for_country_class)
            # print([(c['class'], c['code'], c['start']) for c in list(set(gold_for_country_class) - set([j for (i,j,k) in matches]))])
            fp = len(predicted_for_country_class) - positives
            fn = len(gold_for_country_class) - detected_refs
            
            total_fp += fp
            total_fn += fn
            total_detected_refs += detected_refs

            tps += positives
            prec = positives / (positives + fp) if (positives + fp) else 0
            rec = detected_refs / len(gold_for_country_class) if gold_for_country_class else 0
            
            if len(gold_for_country_class) != 0:
                precision.append(prec )
                recall.append(rec)
                f1.append(
                    2 * prec * rec / (prec + rec)
                    if prec + rec > 0 else 0
                )

                overlaps.append(
                    max(iou for _, _, iou in matches) 
                    if matches else 0
                )
            
        total_ref=int((gold['class']==pred_class(cls)).sum())
        total_precision=sum(precision)/len(precision) if precision else 0.0
        total_recall=sum(recall)/len(recall) if recall else 0.0
        overlaps = [elem for elem in overlaps if elem != 0]
        tot_f1=sum(f1)/len(f1) if f1 else 0.0
        metrics.append({'class':cls,'predictions':total_pred,'emdat_events':total_ref, 'tp': tps,
                        'precision':total_precision,'recall':total_recall,'f1':tot_f1,
                        'average_temporal_overlap_iou': sum(overlaps)/len(overlaps) if overlaps else 0})
    import pprint
    pprint.pprint(metrics)
    pd.DataFrame(metrics).to_csv(out/'metrics_by_class.csv',index=False)
    
    plot_metrics(metrics, out)
    # # Overall micro scores across all five classes.
    tp=sum(x['tp'] for x in metrics)
    npred=sum(x['predictions'] for x in metrics)
    nref=sum(x['emdat_events'] for x in metrics)
    
    precision = tp / npred if npred else 0
    recall = total_detected_refs / nref if nref else 0
    overlap = sum(x['average_temporal_overlap_iou'] for x in metrics) / len( metrics)
    overall={'start_year':args.start_year,'end_year':args.end_year,'africa_only':True,
      'temporal_overlap':'intersection days / union days (IoU)', 'predictions':npred,'emdat_events':nref,
      'true_positives': tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
      'precision':precision,'recall':recall,'macro_f1':2*precision*recall/(precision+recall) if precision+recall else 0,
    #   'micro_f1':2*tp/(2*tp + total_fn + total_fn),
      'average_temporal_overlap_iou':overlap}
    #   'note':'EM-DAT partial dates are expanded to their calendar month/year bounds. Storm cccccccps to EM-DAT Disaster Type=Storm; cyclone subtypes are included under Storm.'}
    (out/'overall_metrics.json').write_text(json.dumps(overall,indent=2,default=str))
    print(json.dumps(overall,indent=2,default=str)); print(f'\nOutputs written to: {out.resolve()}')
    
if __name__=='__main__': 
    main()
