import gradio as gr  # Import Gradio for creating web interfaces
import torch  # Import PyTorch for deep learning
from PIL import Image  # Import PIL for image processing
from transformers import pipeline, CLIPProcessor, CLIPModel  # Import necessary classes from Hugging Face Transformers
import requests  # Import requests for making HTTP requests
from bs4 import BeautifulSoup  # Import BeautifulSoup for web scraping
from gtts import gTTS  # Import gTTS for text-to-speech conversion

# Define the device to use (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BLIP model for image captioning
caption_image = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)

# Load CLIP model for image classification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the English summarization model
summarization_pipeline = pipeline("summarization", model="google/pegasus-xsum")

# Load the Arabic summarization model
arabic_summarization_pipeline = pipeline("summarization", model="abdalrahmanshahrour/auto-arabic-summarization")

# Load the translation model
translation_pipeline = pipeline("translation", model="facebook/nllb-200-distilled-600M")

# Function to fetch long texts from Wikipedia
def get_wikipedia_summary(landmark_name, language='en'):
    url = f"https://{language}.wikipedia.org/wiki/{landmark_name.replace(' ', '_')}"  # Construct the URL
    response = requests.get(url)  # Make an HTTP GET request to fetch the page
    soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content with BeautifulSoup

    paragraphs = soup.find_all('p')  # Extract all paragraph elements
    summary_text = ' '.join([para.get_text() for para in paragraphs if para.get_text()])  # Join text from all paragraphs

    return summary_text[:2000]  # Return the first 2000 characters of the summary

# Function to load landmarks from an external file
def load_landmarks(filename):
    landmarks = {}
    with open(filename, 'r', encoding='utf-8') as file:  # Open the file in read mode
        for line in file:
            if line.strip():
                english_name, arabic_name = line.strip().split('|')  # Split by the delimiter
                landmarks[english_name] = arabic_name  # Add to the dictionary
    return landmarks  # Return the dictionary of landmarks

# Load landmarks from the file
landmarks_dict = load_landmarks("landmarks.txt")

# Function to convert text to speech
def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)  # Create a gTTS object for text-to-speech
    audio_file = "summary.mp3"  # Define the audio file name
    tts.save(audio_file)  # Save the audio file
    return audio_file  # Return the path to the audio file

# Function to generate a caption for the image
def generate_caption(image):
    return caption_image(image)[0]['generated_text']  # Get generated caption from the model

# Function to classify the image using the CLIP model
def classify_image(image, labels):
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)  # Prepare inputs for CLIP model
    outputs = clip_model(**inputs)  # Get model outputs
    logits_per_image = outputs.logits_per_image  # Get logits for images
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]  # Compute probabilities
    top_label = labels[probs.argmax()]  # Get the label with the highest probability
    top_prob = probs.max()  # Get the highest probability value
    return top_label, top_prob  # Return top label and probability

# Function to summarize the description
def summarize_description(full_description, language):
    if language == 'ar':
        return arabic_summarization_pipeline(full_description, max_length=150, min_length=50, do_sample=False)[0]['summary_text']  # Summarize in Arabic
    else:
        return summarization_pipeline(full_description, max_length=150, min_length=50, do_sample=False)[0]['summary_text']  # Summarize in English

# Function to translate the caption and classification result
def translate_results(caption, top_label, top_prob, landmarks_dict, language):
    if language == 'ar':
        caption_translated = translation_pipeline(caption, src_lang='eng_Latn', tgt_lang='arb_Arab')[0]['translation_text']  # Translate caption to Arabic
        classification_result = translation_pipeline(f"أفضل مطابقة: {landmarks_dict[top_label]} باحتمالية {top_prob:.4f}", src_lang='eng_Latn', tgt_lang='arb_Arab')[0]['translation_text']  # Translate classification result
    else:
        caption_translated = caption  # Keep caption in English
        classification_result = f"Best match: {top_label} with probability {top_prob:.4f}"  # Create English classification result

    return caption_translated, classification_result  # Return translated results

# Function to process the image and generate results
def process_image(image, language='en'):
    try:
        # Generate caption for the image
        caption = generate_caption(image)  # Call the caption generation function

        # Classify the image
        top_label, top_prob = classify_image(image, list(landmarks_dict.keys()))  # Use keys for classification

        # Determine the appropriate name to use based on the language
        landmark_name = top_label if language == 'en' else landmarks_dict[top_label]
        full_description = get_wikipedia_summary(landmark_name, language)  # Get the Wikipedia summary for the top label

        # Summarize the full description
        summarized_description = summarize_description(full_description, language)  # Call the summarization function

        # Translate caption and classification result
        caption_translated, classification_result = translate_results(caption, top_label, top_prob, landmarks_dict, language)  # Call the translation function

        # Convert the summarized description to speech
        audio_file = text_to_speech(summarized_description, language)  # Convert summary to audio

        # Return results formatted for Arabic
        if language == 'ar':
            return f"<div style='text-align: right;'>{caption_translated}</div>", \
                   f"<div style='text-align: right;'>{classification_result}</div>", \
                   f"<div style='text-align: right;'>{summarized_description}</div>", \
                   audio_file  # Return formatted results for Arabic
        else:
            return caption_translated, classification_result, summarized_description, audio_file  # Return results for English
    except Exception as e:
        return "Error processing the image.", str(e), "", ""  # Return error message if any exception occurs

# Create Gradio interface for English
english_interface = gr.Interface(
    fn=lambda image: process_image(image, language='en'),  # Function to call on image upload
    inputs=gr.Image(type="pil", label="Upload Image"),  # Input field for image upload
    outputs=[  # Define output fields
        gr.Textbox(label="Generated Caption"),  # Output for generated caption
        gr.Textbox(label="Classification Result"),  # Output for classification result
        gr.Textbox(label="Summarized Description", lines=10),  # Output for summarized description
        gr.Audio(label="Summary Audio", type="filepath")  # Output for audio summary
    ],
    title="Landmark Recognition",  # Title of the interface
    description="Upload an image of a landmark, and we will generate a description, classify it, and provide simple information.",  # Description of the tool
    examples=[  # Examples for user
        ["SOL.jfif"],
        ["OIP.jfif"]
    ]
)

# Create Gradio interface for Arabic
arabic_interface = gr.Interface(
    fn=lambda image: process_image(image, language='ar'),  # Function to call on image upload
    inputs=gr.Image(type="pil", label="تحميل صورة"),  # Input field for image upload in Arabic
    outputs=[  # Define output fields
        gr.HTML(label="التعليق المولد"),  # Output for generated caption in Arabic
        gr.HTML(label="نتيجة التصنيف"),  # Output for classification result in Arabic
        gr.HTML(label="الوصف الملخص"),  # Output for summarized description in Arabic
        gr.Audio(label="صوت الملخص", type="filepath")  # Output for audio summary in Arabic
    ],
    title="التعرف على المعالم",  # Title of the interface in Arabic
    description="قم بتحميل صورة لمعلم، وسنعمل على إنشاء وصف له وتصنيفه وتوفير معلومات بسيطة",  # Description of the tool in Arabic
    examples=[  # Examples for user
        ["SOL.jfif"],
        ["OIP.jfif"]
    ]
)

# Merge all interfaces into a tabbed interface
demo = gr.TabbedInterface(
    [english_interface, arabic_interface],  # List of interfaces to include
    ["English", "العربية"]  # Names of the tabs
)

# Launch the interface
demo.launch()  # Start the Gradio application.
