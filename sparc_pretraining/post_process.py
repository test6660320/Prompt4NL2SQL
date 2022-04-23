
with open('dev_gold.txt', 'r') as f:
    gold = f.readlines()

with open('sparc_predict/test.txt', 'r') as f1:
    predict = f1.readlines()

# for idx, i in enumerate(gold):
#     if i == '\n':
#         print(idx)
#         predict[idx] = '\n'
for idx, i in enumerate(gold):
    if i == '\n':
        print(idx)
        predict = predict[:idx] + ['\n'] + predict[idx:]

with open('test_post_process.txt', 'w') as fw:
    for i in predict:
        fw.write(i)

with open('test_post_process.txt', 'r') as f1:
    predict = f1.readlines()

new = []
for line in predict:
    line = line.replace(' . ', '.')
    line = line.replace(' , ', ', ')
    line = line.replace('<unk>', '<')
    new.append(line)

with open('sparc_predict/test.txt', 'w') as f:
    for line in new:
        f.write(line)