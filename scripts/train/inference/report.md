Qwen 3.5 4B

Checpoint step 200

Metric                            ── RELAXED ──────────    ── EXACT ────────────
                                    Prec    Rec     F1     Prec    Rec     F1
--------------------------------------------------------------------------------
Event extraction                   49.6   46.3   47.9     28.9   27.0   27.9   TP=1466  FP=1491  FN=1700
Event type (multiset)              62.6   58.4   60.4     62.6   58.4   60.4   TP=1850  FP=1107  FN=1316
Event type (doc-level set)         76.8   60.7   67.8     76.8   60.7   67.8   TP=1554  FP= 469  FN=1006
Cluster event extraction           54.5   50.9   52.7     31.6   29.5   30.5   TP=1612  FP=1345  FN=1554
Cluster type (multiset)            72.5   67.8   70.1     72.5   67.8   70.1   TP=2145  FP= 812  FN=1021
Cluster type (doc-level set)       89.7   75.1   81.7     89.7   75.1   81.7   TP=1179  FP= 136  FN= 391
Location extraction                70.9   63.2   66.8     69.0   61.5   65.0   TP=2031  FP= 834  FN=1185
Event-location pairing             40.4   37.7   39.0     23.1   21.6   22.3   TP=1194  FP=1763  FN=1972
Cluster event-location pairing     44.5   41.5   43.0     25.3   23.6   24.4   TP=1315  FP=1642  FN=1851
Event-time pairing                 43.9   41.0   42.4     25.0   23.3   24.1   TP=1298  FP=1659  FN=1868

Checkpoint step 400

Metric                            ── RELAXED ──────────    ── EXACT ────────────
                                    Prec    Rec     F1     Prec    Rec     F1
--------------------------------------------------------------------------------
Event extraction                   58.8   45.6   51.3     35.8   27.8   31.3   TP=1443  FP=1013  FN=1723
Event type (multiset)              71.0   55.1   62.0     71.0   55.1   62.0   TP=1744  FP= 712  FN=1422
Event type (doc-level set)         81.5   57.9   67.7     81.5   57.9   67.7   TP=1482  FP= 337  FN=1078
Cluster event extraction           63.2   49.0   55.2     38.1   29.5   33.3   TP=1551  FP= 905  FN=1615
Cluster type (multiset)            80.3   62.3   70.2     80.3   62.3   70.2   TP=1972  FP= 484  FN=1194
Cluster type (doc-level set)       92.7   71.8   80.9     92.7   71.8   80.9   TP=1128  FP=  89  FN= 442
Location extraction                73.0   65.9   69.3     71.3   64.3   67.6   TP=2118  FP= 782  FN=1098
Event-location pairing             48.9   37.9   42.7     29.9   23.2   26.1   TP=1200  FP=1256  FN=1966
Cluster event-location pairing     52.6   40.8   45.9     31.8   24.6   27.7   TP=1291  FP=1165  FN=1875
Event-time pairing                 52.1   40.4   45.5     31.4   24.3   27.4   TP=1279  FP=1177  FN=1887

Checkpoint step 600

Metric                            ── RELAXED ──────────    ── EXACT ────────────
                                    Prec    Rec     F1     Prec    Rec     F1
--------------------------------------------------------------------------------
Event extraction                   56.7   55.0   55.8     35.9   34.8   35.3   TP=1740  FP=1328  FN=1426
Event type (multiset)              68.0   65.9   66.9     68.0   65.9   66.9   TP=2085  FP= 983  FN=1081
Event type (doc-level set)         77.7   68.1   72.6     77.7   68.1   72.6   TP=1743  FP= 501  FN= 817
Cluster event extraction           60.6   58.7   59.7     38.1   36.9   37.5   TP=1860  FP=1208  FN=1306
Cluster type (multiset)            75.9   73.5   74.7     75.9   73.5   74.7   TP=2328  FP= 740  FN= 838
Cluster type (doc-level set)       89.1   81.9   85.3     89.1   81.9   85.3   TP=1286  FP= 158  FN= 284
Location extraction                73.4   72.0   72.7     71.9   70.6   71.2   TP=2317  FP= 841  FN= 899
Event-location pairing             47.3   45.9   46.6     29.9   29.0   29.4   TP=1452  FP=1616  FN=1714
Cluster event-location pairing     50.6   49.0   49.8     31.7   30.7   31.2   TP=1552  FP=1516  FN=1614
Event-time pairing                 50.6   49.0   49.8     31.6   30.7   31.2   TP=1552  FP=1516  FN=1614