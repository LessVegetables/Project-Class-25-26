import json


# "from_id": "user521051027",

files_to_parse = ['/Users/danielgehrman/Documents/Programming/trainData/katie/result.json',
                  '/Users/danielgehrman/Documents/Programming/trainData/dad/result.json',
                  '/Users/danielgehrman/Documents/Programming/trainData/mom/result.json'
                  ]

my_messages_data = {"messages": []}

for file in files_to_parse:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Loop through every message in the export
            for msg in data['messages']:

                # We only want actual text messages, not "Call started" etc.
                if msg['type'] == 'message':
                    # We only want my messages
                    if msg["from_id"] != "user521051027":
                        continue

                    text_content = msg.get('text', '')
                    
                    # --- Telegram Quirk Handling ---
                    # Sometimes, if a message has a link or bold text, Telegram 
                    # stores it as a list instead of a simple string.
                    # This quick check handles that.
                    if isinstance(text_content, list):
                        full_text = ""
                        for item in text_content:
                            if isinstance(item, str):
                                full_text += item
                            elif isinstance(item, dict):
                                # Grab the text part of the link/formatting
                                full_text += item.get('text', '')
                        text_content = full_text
                    # -------------------------------

                    # If there is text, add it to our list
                    if text_content:
                        my_messages_data['messages'].append(text_content)
                        
    except FileNotFoundError:
        print(f"ERROR: Could not find {file}.")
        print("Make sure the JSON file is in the same folder as this script.")
        print("Skipping it for now")
        continue

with open("my-messages.json", "w", encoding='utf-8') as f:
    json.dump(my_messages_data, f, ensure_ascii=False, indent=2)