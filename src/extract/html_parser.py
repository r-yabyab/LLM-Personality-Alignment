"""
Pulls conversation from .html, appends consecutive messages into new line into one message into person 1,
repeat for person 2 and write as a json into a jsonl file

TODO:
# - incorporate time for topic detection
- order incorrect in some, first conversation starts with me, needs to be other person
"""


import os
import json
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import jsonlines
from transformers import BertTokenizer

data_root = "../../data"
root = f"{data_root}/localex/msg/dms"
message_selector = "span.chatlog__markdown-preserve"
short_timestamp = "chatlog__short-timestamp" #div
# sample = "380890688247693314_short.html"
sample = "380890688247693314.html"
message_author = "chatlog__message-group"
    # from message_selector... parent, parent --> message author


# puts into pairs.jsonl
def main():

    # yes, no, hi (maybe keep hi)
    MIN_FIRST_MESSAGE_LEN = 3

    model = SentenceTransformer("all-MiniLM-L6-v2")

    user_message = ""
    print(len(os.listdir(root)))
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
        open(f"{data_root}/pairs.jsonl", "w", encoding="utf-8") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages  = soup.select(message_selector)
        # for message in messages:
        # for i in range(len(messages) - 1):
        conversation_length = len(messages)
        start = 0
        # end = 25
        end = conversation_length -1

        # get starting person not me
        # cosine_sim with other person
        person_one = { "name": None, "content": ""}
        person_two = { "name": "net", "content": ""}
        counter = 0


        for i in range(start, end):
            # user_message = ""
            next_message = messages[i+1].find_parent("div", class_=message_author)
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            next_author = next_message.select_one("span", class_="chatlog__author").get_text()
            # author = author_tag.get_text()
            timestamp = parent.select_one(".chatlog__timestamp")["title"]
            dt = datetime.strptime(timestamp, "%A, %B %d, %Y %H:%M")

            if author != "net":
                person_one["name"] = author

            
            if author == next_author:
                user_message += f"{messages[i].get_text()}\n"
                # person_one["content"] += f"{messages[i].get_text()}\n"
            if author != next_author or i == end - 1 :
                if author == "net":
                    user_message += messages[i].get_text()
                    person_two["content"] = user_message
                    print(person_one)
                    print(person_two)

                    encode_person_one = model.encode(person_one["content"])
                    encode_person_two = model.encode(person_two["content"])
                    score = cos_sim(encode_person_one, encode_person_two)
                    print(score)
                    writer.write(json.dumps(
                        {"1": person_one, 
                         "2": person_two, 
                         "cos_sim": float(score[0][0])}
                    ) + "\n")

                    # clear
                    person_one["content"] = ""
                    person_two["content"] = ""
                    user_message = ""

                else:
                    print("--------")
                    user_message += messages[i].get_text()
                    person_one["content"] = user_message
                    # print(author)
                    # print(timestamp)
                    # print(dt)
                    # print(user_message)
                    # print(person_one)
                    # print(person_two)
                    #person_one.clear()
                    user_message = ""
                    
def group_text_single():
    # puts into pairs_plan.json
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # text = "your text here"
    # tokens = tokenizer.tokenize(text)
    # print(len(tokens))

    MIN_FIRST_MESSAGE_LEN = 3
    message_list = []
    conversation_list = []
    running_tokens = 0

    user_message = ""
    # print(len(os.listdir(root)))
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
        open(f"{data_root}/pairs_plain.jsonl", "w", encoding="utf-8") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages  = soup.select(message_selector)
        conversation_length = len(messages)
        start = 0
        end = conversation_length -1

        for i in range(start, end):
            # user_message = ""
            next_message = messages[i+1].find_parent("div", class_=message_author)
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            next_author = next_message.select_one("span", class_="chatlog__author").get_text()
            timestamp = parent.select_one(".chatlog__timestamp")["title"]
            dt = datetime.strptime(timestamp, "%A, %B %d, %Y %H:%M")
            
            if author == next_author:
                user_message += messages[i].get_text() + "\n"
            else:
                user_message += messages[i].get_text()

                msg_tokens = len(tokenizer.encode(user_message, add_special_tokens=False, truncation=True, max_length=510))

                if running_tokens + msg_tokens >= 300:
                    # store current conversation
                    conversation_list.append(message_list)

                    # start new one
                    message_list = [user_message]
                    running_tokens = msg_tokens
                else:
                    message_list.append(user_message)
                    running_tokens += msg_tokens

                user_message = ""

        # add last conversation if not empty
        if message_list:
            conversation_list.append(message_list)

        writer.write(json.dumps(conversation_list, ensure_ascii=False))

def get_message_text(tag):
    """Extract text from a message span, replacing chatlog__emoji images with their title."""
    result = []
    for child in tag.children:
        name = getattr(child, 'name', None)
        if name == 'img':
            classes = child.get('class', [])
            if any('chatlog__emoji' in c for c in classes):
                title = child.get('title', '')
                result.append(f':{title}:' if title else '')
        elif name is not None:
            result.append(get_message_text(child))
        else:
            result.append(str(child))
    return ''.join(result)

def scan_messages():
    files = os.listdir(f"{root}")
    with open("../../logs/empty_log.txt", "w", encoding="utf-8") as log_writer, \
        open("../../logs/small_files_log.txt", "w", encoding="utf-8") as size_log_writer:
        for i, file in enumerate(files):
            with open(f"{root}/{file}", "r", encoding="utf-8") as reader:
                soup = BeautifulSoup(reader, "html.parser")
                messages  = soup.select(message_selector)
                # if i <= 0:
                #     print(f"skipping {file}")
                #     continue
                if not messages:
                    print(f"skipping {file}")
                    log_writer.write(f"{file}" + "\n")
                elif len(messages) < 10:
                    print(f"small file: {file}")
                    size_log_writer.write(str(file) + "\n")
                else:
                    print(len(messages))
                    # group_text_jsonl(root, file)
                    group_text(root, file)
                    
def group_text(root, sample):
    # puts into pairs_plain.json
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    message_list = []
    conversation_list = []
    running_tokens = 0

    user_message = ""
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
        open(f"{data_root}/transformed/plain/{sample}-pairs_plain.json", "w", encoding="utf-8") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages  = soup.select(message_selector)
        conversation_length = len(messages)
        start = 0
        end = conversation_length -1

        for i in range(start, end):
            # user_message = ""
            next_message = messages[i+1].find_parent("div", class_=message_author)
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            next_author = next_message.select_one("span", class_="chatlog__author").get_text()
            timestamp = parent.select_one(".chatlog__timestamp")["title"]
            dt = datetime.strptime(timestamp, "%A, %B %d, %Y %H:%M")
            
            first_message_net = False
            
            if i == 0 and author == "net":
                first_message_net = True
            
            if author == next_author:
                user_message += get_message_text(messages[i]) + "\n"
            else:
                user_message += get_message_text(messages[i])

                msg_tokens = len(tokenizer.encode(user_message, add_special_tokens=False, truncation=True, max_length=510))

                if first_message_net:
                    pass
                else:
                    if running_tokens + msg_tokens >= 300 and not author == "net":
                        # store current conversation
                        conversation_list.append(message_list)

                        # start new one
                        message_list = [user_message]
                        running_tokens = msg_tokens
                    else:
                        message_list.append(user_message)
                        running_tokens += msg_tokens

                user_message = ""

        # add last conversation if not empty
        if message_list:
            conversation_list.append(message_list)

        writer.write(json.dumps(conversation_list, ensure_ascii=False))

def group_text_jsonl(root, sample):
    user_message = ""
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
        open(f"{data_root}/transformed/{sample}-pairs_grouped.jsonl", "w", encoding="utf-8") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages = soup.select(message_selector)
        conversation_length = len(messages)
        start = 0
        end = conversation_length - 1
        current_author = None

        for i in range(start, end):
            next_message = messages[i + 1].find_parent("div", class_=message_author)
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            next_author = next_message.select_one("span", class_="chatlog__author").get_text()
            current_author = author
            first_message_net = False

            if i == 0 and author == "net":
                first_message_net = True
                
            if author == next_author:
                user_message += messages[i].get_text() + "\n"
            else:
                if first_message_net:
                    pass
                else:
                    user_message += messages[i].get_text()
                    writer.write(json.dumps({"author": author, "content": user_message}, ensure_ascii=False) + "\n")
                    user_message = ""
                    current_author = None

        # flush last message
        if user_message and current_author:
            writer.write(json.dumps({"author": current_author, "content": user_message}, ensure_ascii=False) + "\n")

        print("finished")
        
def group_text_jsonl_single():
    user_message = ""
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
        open(f"{data_root}/pairs_grouped.jsonl", "w", encoding="utf-8") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages = soup.select(message_selector)
        conversation_length = len(messages)
        start = 0
        end = conversation_length - 1
        current_author = None

        for i in range(start, end):
            next_message = messages[i + 1].find_parent("div", class_=message_author)
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            next_author = next_message.select_one("span", class_="chatlog__author").get_text()
            current_author = author

            if author == next_author:
                user_message += messages[i].get_text() + "\n"
            else:
                user_message += messages[i].get_text()
                writer.write(json.dumps({"author": author, "content": user_message}, ensure_ascii=False) + "\n")
                user_message = ""
                current_author = None

        # flush last message
        if user_message and current_author:
            writer.write(json.dumps({"author": current_author, "content": user_message}, ensure_ascii=False) + "\n")

        print("finished")

# extracts tag text into 1 line each
def raw_text():
    with open(f"{root}/{sample}", "r", encoding="utf-8") as reader, \
    jsonlines.open(f"{data_root}/rawtext.jsonl", "w") as writer:
        soup = BeautifulSoup(reader, "html.parser")
        messages  = soup.select(message_selector)
        conversation_length = len(messages)
        start = 0
        end = conversation_length -1

        for i in range(start, end):
            parent = messages[i].find_parent("div", class_=message_author)
            author = parent.select_one("span", class_="chatlog__author").get_text()
            writer.write({
                "message": messages[i].get_text(),
                "author": author
            }
                )
            print(messages[i].get_text())
            
        # for i in range(start, end):
        #     parent = messages[i].find_parent("div", class_=message_author)
        #     author = parent.select_one("span", class_="chatlog__author").get_text()
        #     writer.write(messages[i].get_text())
        #     print(messages[i].get_text())
        
# main()
# raw_text()
# group_text_single() # this one works for text-segmentation
# group_text_jsonl_single() #this one works
scan_messages()