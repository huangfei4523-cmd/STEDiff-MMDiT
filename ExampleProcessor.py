import utils

def process_example(instance_image, instance_prompt, image_transforms, tokenizer, tokenizer_max_length=77):
    example = {}
    example["instance_images"] = image_transforms(utils.image_convert_RGB(instance_image))
    text_inputs = utils.tokenize_prompt(tokenizer, instance_prompt, tokenizer_max_length=tokenizer_max_length)
    example["text_inputs"] = text_inputs
    example["instance_prompt_ids"] = text_inputs.input_ids
    example["instance_attention_mask"] = text_inputs.attention_mask
    return example