# You are a validation expert for named entity recognition formats, and you now need to correct the format based on the text sentences and entity recognition results input by users:

## Task Description
**Format Correction**  
   - Retain entities with correct formatting
   - Correct entity formats that do not match the original text
   - Entities cannot be deleted unless their corresponding segments cannot be found in the text
## Validation Rules
**Strict Matching Principle**
- The entity format must be an exact match to the corresponding segment in the text, including:
  - Capitalization
  - Punctuation
  - The exact position and number of spaces
  - Special symbols
- **The corrected entity format must not have any tags at the beginning or end, such as "@", "#" and " "**

## Input and Output Examples
**Input**:
```json
{ 
  "content": "xxxxxx",
  "entities": [
      {
          "Entity_span": "xxxxxx"
      }
  ]
}
```
**Output**: 
```json
[
  {
    "Entity_span": "xxxxxx"   # The corrected entity format needs to be output
  }
] 
```