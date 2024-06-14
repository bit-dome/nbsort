import json
import re
from bs4 import BeautifulSoup
from pathlib import Path



def extract_hidden_divs(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    hidden_divs = soup.find_all('div', style='display:none')

    div_data_dict = {}

    for div in hidden_divs:
        div_id = div.get('id', 'unknown_id')
        protobuf_string = div.get_text()

        # Clean up the string by removing \n and adding colons

        protobuf_string = re.sub(r'(\w+)\s*:', r'"\1":', protobuf_string)
        protobuf_string = re.sub(r'(\w+)\s*{', r'"\1": {', protobuf_string)
        protobuf_string = re.sub(r'\t', '', protobuf_string)
        protobuf_string = re.sub(r'\n', ',', protobuf_string)

        protobuf_string = protobuf_string.replace(" ", "")
        protobuf_string = protobuf_string.replace(",}", "}")
        protobuf_string = protobuf_string.replace("{,", "{")
        protobuf_string = protobuf_string.replace(",,", "")

        protobuf_string = protobuf_string.replace('"set_display_name",' , '[')
        protobuf_string = protobuf_string.replace('"object_annotation":' , '')
        protobuf_string = "{"+protobuf_string[1:]+"}"
        protobuf_string = protobuf_string.replace('}}}' , '}]}}')

        jj = json.loads(protobuf_string)

        image_name = Path(jj["image"]["image_source"]["image_source_id"]).stem



        div_data_dict[image_name] = jj['object_annotations']['group_name']

    return div_data_dict




def save_to_json(data_dict, output_folder):
    for div_id, div_data in data_dict.items():
        output_file_path = f"{output_folder}/{str(div_id).zfill(6)}.json"
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(json.dumps(div_data))

# Replace 'your_file.html' with the path to your HTML file
html_file_path = 'raid_results.html'
output_data = extract_hidden_divs(html_file_path)

# Specify the output folder (create it if it doesn't exist)
output_folder = 'raid_results'
save_to_json(output_data, output_folder)
