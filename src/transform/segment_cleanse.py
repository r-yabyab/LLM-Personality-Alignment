import jsonlines
import os
"""
filters out freshly segmented conversations

TODO:
score each conversation by similarity
filter by char num
"""


data_root = "../../data"
# input = "./data/pairs_grouped_topics.jsonl" #segments
input = f"{data_root}/plain_pairs_combined.jsonl" #segments
# output = "./data/pairs_grouped_topics_filtered.jsonl"
output = f"{data_root}/plain_pairs_combined_filtered.jsonl"

def scan_files():
    file_path = os.listdir("./data/transformed/plain/plain_pairs")
    for file in file_path:
        segment_cleanse(file)
        

def segment_cleanse(file):
    
    data = []
    
    # with jsonlines.open(input, "r") as reader, \
    # jsonlines.open(output, "w") as writer:
    base_name = os.path.splitext(os.path.basename(file))[0]
    with jsonlines.open(file, "r") as reader, \
    jsonlines.open(f"{data_root}/{base_name}_filtered.jsonl", "w") as writer:
        for i, point in enumerate(reader):
            if i > 1317:
                break
            print(point)
            conversation_turns = len(point["messages"])
            print(conversation_turns)
            if conversation_turns <= 6:
                writer.write(point)
                
def readlines():
    with jsonlines.open(input, "r") as reader:
        for i,point in enumerate(reader):
            if i>1317:
                break
            conversations = point["messages"]
            if len(conversations) <= 6:
                print(conversations)
                print(len(conversations))

segment_cleanse(input) # for single
# readlines()