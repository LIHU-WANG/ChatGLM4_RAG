# import csv
#
# word_lists = []
# with open('data/risk/BIO_DATA_NSBD_TWO.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     word_list = []
#     building_list = []
#     location_list = []
#     ff_list = []
#     fm = []
#     for line in lines:
#         if line != '\n':
#             word, tag = line.strip('\n').split(' ')
#             word_list.append(word)
#             if tag.endswith('Building'):
#                 building_list.append(word)
#             if tag.endswith('Location'):
#                 location_list.append(word)
#             if tag.endswith('Failure_facility'):
#                 ff_list.append(word)
#             if tag.endswith('Failure_mode'):
#                 fm.append(word)
#         else:
#             word_lists.append({
#                 'words': ''.join(word_list),
#                 'buildings': ''.join(building_list),
#                 'locations': ''.join(location_list),
#                 'Failure_facility': ''.join(ff_list),
#                 'Failure_mode': ''.join(fm)
#             })
#             word_list = []
#             building_list = []
#             location_list = []
#             ff_list = []
#             fm = []
#
# with open('data/risk/BIO_DATA_NSBD_TWO.csv', 'a', newline='', encoding='utf-8') as csv_file:
#     fieldnames = ['words', 'buildings', 'locations', 'Failure_facility', 'Failure_mode']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     # 追加吸入新数据
#     for row in word_lists:
#         writer.writerow(row)
#

import json

word_lists = []
with open('data/AdvertiseGen/output_train.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        jd = json.loads(line)['messages']
        word_lists.append({
            "instruction": jd[0]['content'],
            "input": "",
            "output": jd[1]['content']
        })

with open('data/AdvertiseGen/chatglm_risk.json', 'a+', encoding='utf-8') as fw:
    json.dump(word_lists, fw, ensure_ascii=False, indent=1, separators=(',', ':'))
    fw.close()

