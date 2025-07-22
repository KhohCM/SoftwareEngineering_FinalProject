# Project Introduction
Project Overview
In this project we have decided to developed an AI chatbot that mimics Gordon Ramsay’s tone and personality, which uses a custom language model combined with the Groq API and responds to user queries about, recipes, cooking methods, ingredients, dish recommendations and Addtionally it also provides culinary guidance in Gordon Ramsay’s style which aims to make cooking interactive and educational for users. Moving on the scope of this project includes apply NLP to interpret user cooking-related questions, and support upload of PDF/DOCX recipe documents for enhanced analysis (RAG: Retrieval Augmented Generation), currently limitations only supports english, no multilingual functionality. Furthermore note that the responses may not fully reflect Gordon Ramsay’s real culinary preferences and no voice interaction which the system lacks text-to-speech capabilities.


# User Manual
To run the system the first step will ensure that Python (version 3.8 or later) is installed on your computer.

First, install all of the dependencies that are required:
```bash
pip install -r requirements.txt
```

Next once all of the libraries are installed drive the google drive link below and download the .bin file and place it under folder named "gpt2-ramsay-finetuned2", which will be resulted in "your_path/gpt2-ramsay-finetuned2/pytorch_model.bin"
```bash
drive_link : https://drive.google.com/drive/folders/16f1-3VKEIbSRZlygmE0JUx23hnHPzbSQ?usp=sharing
```
Once all of the libraries and the .bin file is installed run the app.py:
```bash
python app.py
```

After clicking the link to the website, the chatbot should show up on the screen


