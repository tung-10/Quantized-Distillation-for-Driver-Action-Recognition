import pickle

with open("/mnt/data/quangtungbk/qd-driver-activity-reco/output/logits_sequence.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
for key, value in data.items():
    print(key, type(value))