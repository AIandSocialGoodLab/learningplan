# README
## Usage
```ruby
python simulator.py --student=x --output=output_file_path --kc=number_of_knowledge_component --proficiency=number_of_proficiency_levels --actions=action_file_path
```
This will run the simulator and provide a csv file under at the output file path.

```ruby
python create_action.py --kc=number_of_knowledge_component --proficiency=number_of_proficiency_levels --actions=action_file_path
```
This will create some action-proficiency relations stored at the action file path.
