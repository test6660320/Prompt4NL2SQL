import json

with open('./data/sparc_data_uncased_with_relavant_column2_with_final/dev.json') as f:
    all = json.load(f)
with open('./data/turn_switch_aux_data/dev.json') as f:
    twa = json.load(f)

print(all[0])
indext = 0
for index, cont in enumerate(all):
    for indexb, contb in enumerate(all[index]['dialogue_session'][:-1]):
        tmp = twa[indext]
        all[index]['dialogue_session'][indexb]['turn_change_index'] = tmp['turn_change_index']
        indext += 1
import os
# os.mkdir('./data/sparc_data_uncased_with_rc_tw_with_final')
with open('./data/sparc_data_uncased_with_rc_tw_with_final/dev.json', 'w') as f:
    json.dump(all, f)


