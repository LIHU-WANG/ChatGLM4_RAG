#
# # 打开第一个文件并读取内容
# with open('data/AdvertiseGen/output_line.jsonl', 'r', encoding='utf-8') as file1:
#     content1 = file1.read()
#
# # 打开第二个文件并读取内容
# with open('data/AdvertiseGen/train.jsonl', 'r', encoding='utf-8') as file2:
#     content2 = file2.read()
#
# # 打开（或创建）一个新的文件并写入合并后的内容
# with open('data/AdvertiseGen/merged_file.jsonl', 'w', encoding='utf-8') as merged_file:
#     merged_file.write(content1)
#     merged_file.write('\n')  # 可选：在两个文件内容之间添加一个换行符
#     merged_file.write(content2)
#
# print("文件已成功合并！")
#
#