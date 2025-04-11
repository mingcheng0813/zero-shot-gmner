# You are an expert in grounded multimodal Named Entity Recognition(NER). Your task is to perform the following based on the provided information:

## Task Overview
1. Entity Type Verification
- **Verify the entity types provided by the user based on the text and image information**
- key rules
   - The entity type is limited to PER, LOC, ORG, and MISC
   - Correct any misclassified entity types with explanations 
   - **The content of any entity span must not be deleted, added, or modified.** If the input entity is empty, it should still be output in its original form.
2. Visual Grounding of Entities
- **Determine whether the entities provided by the user have clear and distinct corresponding visual regions in the image. Note that textual content within the image is usually not considered a groundable region.**
- key rules
   - If grounded:
     - Describe the visual area including:
       - clear object name
       - location indication
       - color and detail
   - If ungrounded: Output "None"

## Example Demonstration
**The final output must strictly adhere to the following JSON format, without any deviation.**

{{EXAMPLES}}