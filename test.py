#load chat dataset
from datasets import load_from_disk
chat_dataset = load_from_disk("coco_chat_dataset")

#visualise the dataset randomly
import random
random_index = random.randint(0, len(chat_dataset)-1)
print(chat_dataset[random_index])

#display image
from PIL import Image
import matplotlib.pyplot as plt
image_path = chat_dataset[random_index]['image_path']
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()