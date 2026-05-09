import jsonlines

# puts into QA pairs for PEFT

            # conversation = [
            #     {
            #         "role": "user",
            #         "content": item['question']
            #     },
            #     {
            #         "role": "assistant", 
            #         "content": item['answer']
            #     }
            # ]
          
# input_data = "./data/pairs_grouped_topics_filtered.jsonl"
data_root = "../../data"
input_data = f"{data_root}/plain_pairs_combined_filtered.jsonl"
output = f"{data_root}/qa_pairs.jsonl"
            
ROLE_MAP = {
    "Person_1": "user",
    "Person_2": "assistant",
}

def main():
    with jsonlines.open(input_data, "r") as reader, \
        jsonlines.open(output, "w") as writer:
        for point in reader:
            messages = [
                {"role": ROLE_MAP.get(msg["role"], msg["role"]), "content": msg["content"]}
                for msg in point["messages"]
            ]
            pairs = [messages[i:i+2] for i in range(0, len(messages) - 1, 2)]
            writer.write({"conversations": pairs})


main()