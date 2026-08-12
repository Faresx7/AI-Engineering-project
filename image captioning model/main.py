# the library the makes the photo coming from request looks like a normal photo file 
from io import BytesIO
# the library that deals with images
from PIL import Image
import requests
from transformers import BlipForConditionalGeneration, BlipProcessor #type:ignore

# processor takes the raw image and convert it to matrix of numbers that model can understand
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
# model is the brain that analyze the data coming from processor
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


image_url = "image-url"


headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(image_url, headers=headers)


raw_image = Image.open(BytesIO(response.content)).convert("RGB")



inputs = processor(images=raw_image, return_tensors="pt") # type: ignore

out = model.generate(
    **inputs,
    max_new_tokens=100,
    min_length=30, 
    num_beams=5,
    repetition_penalty=1.2,
)


res = processor.decode(out[0], skip_special_tokens=True)
print("Detailed Caption:\n", res)
