import json
#COPA
answer = []
for i in range(500):
    answer.append({"idx": 1 + i, "label": 1})
with open('copa.json', 'w') as f:
    json.dump(answer, f, indent=0)

#COLA
answer = []
for i in range(1060):
    answer.append({"idx": i, "label": 1})
with open('cola.json', 'w') as f:
    json.dump(answer, f, indent=0)

#WIC
answer = []
for i in range(1246):
    answer.append({"idx": 1 + i, "label": True})
with open('wic.json', 'w') as f:
    json.dump(answer, f, indent=0)


