Base LLM with reasoning capability for vertical LLMs

Post training pipeline

How to run)

Dataset

Total row = n * 4 (dataset list)

100 rows per each dataset, parallel processing

% python data/collect_math.py -l error.log -n 100 -p -m gemini | tee collect.tee.log

10 rows per each dataset, serial processing

% python data/collect_math.py -l error.log -n 10 -m gemini | tee collect.tee.log

Cluster 

% python data/cluster_results.py output/train_ds_math_update.csv
