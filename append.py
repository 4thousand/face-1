import yaml
import json
name = "recognizer\dataset.json"
with open(name, 'r') as yaml_in:
    yaml_object = yaml.load(yaml_in) # yaml_object will be a list or a dict
    print(yaml_object['data'])
List_length = len(yaml_object['data'])
yaml_object['data'].append({
      "id": List_length+1,
      "name": "big",
      "phone": "00-000",
      "age": 23
    })
print(yaml_object)
output_dict = [x for x in yaml_object['data'] if x['id'] == 1]
print(output_dict[0]['name'])

with open(name,'w') as yaml_out:
    json.dump(yaml_object,yaml_out,indent=4)