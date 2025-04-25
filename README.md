A1 Post training
Total row = n * 4 (dataset list)

python data/collect_math.py -l error.log -n 100 -p -m gemini | tee collect.tee.log
python data/collect_math.py -l error.log -n 20 -p -m gemini | tee collect.tee.log

python data/collect_math.py -l error.log -n 10 -p -m gemini | tee collect.tee.log
python data/collect_math.py -l error.log -n 10 -m gemini | tee collect.tee.log

python data/collect_math.py -l error.log -n 5 -p -m gemini | tee collect.tee.log

python data/collect_math.py -n 1 -p -m gemini | tee collect.tee.log
python data/collect_math.py -n 2 -p -m gemini | tee collect.tee.log
python data/collect_math.py -n 1 -m gemini | tee collect.tee.log