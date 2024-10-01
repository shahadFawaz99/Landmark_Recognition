# Landmark Recognition and Description Project

## Project Objective
The objective of this project is to build a bilingual (Arabic and English) AI application that recognizes landmarks in images and provides descriptive information using Hugging Face pipelines.

## Project Scope
This project incorporates various AI tasks, including:
- **Text-to-Audio**: Generating audio from text descriptions using Google Text-to-Speech (gTTS).
- **Zero-Shot Image Classification**: Using CLIP to classify images without task-specific training.
- **Image Captioning**: Generating captions for images using BLIP.
- **Text Summarization**: Summarizing long texts in both Arabic and English.

## Model Choices
1. **BLIP**: For generating captions from images. It provides descriptive text that reflects the content of the image.
   - [BLIP Model on Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-large)
2. **CLIP**: For zero-shot image classification, which allows the model to classify images based on textual descriptions without any additional training.
   - [CLIP Model on Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14)
3. **PEGASUS**: For summarizing English texts, offering concise and meaningful summaries.
   - [PEGASUS Model on Hugging Face](https://huggingface.co/google/pegasus-xsum)
4. **Arabic Summarization Model**: For summarizing Arabic texts, ensuring the information is relevant for Arabic-speaking users.
   - [Arabic Summarization Model](https://huggingface.co/abdalrahmanshahrour/auto-arabic-summarization)
5. **Translation Model**: For translating between English and Arabic.
   - [Translation Model on Hugging Face](https://huggingface.co/facebook/nllb-200-distilled-600M)

## Libraries Used
- **gTTS (Google Text-to-Speech)**: This library converts text into spoken audio using Google's Text-to-Speech API, enabling the application to provide audio summaries for users.
   - [gTTS Documentation](https://gtts.readthedocs.io/en/latest/)
  
- **Beautiful Soup**: A Python library for parsing HTML and XML documents. It is used in this project to scrape text data from Wikipedia pages, allowing the application to retrieve accurate and relevant descriptions for landmarks.
   - [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## Usage Instructions
- **Upload an Image**: Use the provided interface to upload an image of a landmark.
- **View Outputs**: The system will generate:
  - A descriptive caption.
  - A classification result.
  - A summarized description from Wikipedia.
  - An audio file of the summary.
- **Language Selection**: The interface supports both Arabic and English. Select your preferred language for appropriate outputs.

### Special Measures for Arabic Language Support
The project incorporates a dedicated Arabic summarization model and translation capabilities to ensure users receive accurate and relevant information in Arabic. 

## Challenges Faced
During the development of this project, several challenges were encountered:
- **Model Selection**: Choosing the right models that could perform efficiently in both Arabic and English was complex.
- **Web Scraping**: Initially, we aimed to use generated text, but we shifted to web scraping for more accurate information retrieval from Wikipedia.
- **Bilingual Support**: Ensuring seamless functionality across two languages required careful consideration of translation and summarization quality.

## Deliverables
1. Slides outlining the project's objectives and results.
2. Python Notebook containing the actual code.
3. Hugging Face Space for live demonstration.
4. Video Recording of a walkthrough of the project.

## Links
- [Hugging Face Space](HUGGING_FACE_SPACE_LINK)
- [Video Walkthrough](VIDEO_WALKTHROUGH_LINK)
- [Presentation Slides](PRESENTATION_LINK)
