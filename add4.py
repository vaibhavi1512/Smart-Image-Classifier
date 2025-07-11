from transformers import BlipProcessor, BlipForConditionalGeneration, ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import gradio as gr
from typing import List, Dict, Tuple
import logging
import datetime
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# History storage
HISTORY_FILE = "search_history.json"

class ImageAnalyzer:
    def __init__(self):
        """Initialize models with error handling and device awareness."""
        try:
            # Load models
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            self.vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
            
            # Initialize history
            self.history = self._load_history()
            logger.info("Models and history loaded successfully")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def _load_history(self) -> List[Dict]:
        """Load search history from file."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading history: {str(e)}")
        return []

    def _save_history(self):
        """Save search history to file."""
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")

    def add_to_history(self, image_path: str, caption: str, question: str, answer: str):
        """Add a new entry to the search history."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image": image_path,
            "caption": caption,
            "question": question,
            "answer": answer
        }
        self.history.insert(0, entry)  # Add newest first
        self._save_history()

    def generate_caption(self, image: Image.Image) -> str:
        """Generate image caption."""
        try:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.caption_model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=5,
                    early_stopping=True
                )
            return self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Caption generation failed: {str(e)}")
            return "Could not generate caption"

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer question about image with more detailed responses."""
        try:
            inputs = self.vqa_processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.vqa_model(**inputs)
            predicted_id = outputs.logits.argmax(-1).item()
            base_answer = self.vqa_model.config.id2label[predicted_id]
            
            # Enhanced answer formatting
            question_lower = question.lower()
            
            if base_answer in ["yes", "no"]:
                return f"{base_answer.capitalize()}, {question_lower.replace('?', '').replace('is there', 'there is').replace('are there', 'there are')}"
            elif "color" in question_lower:
                return f"The color appears to be {base_answer}"
            elif "how many" in question_lower:
                return f"There are {base_answer} visible in the image"
            elif "what is" in question_lower or "what are" in question_lower:
                return f"It appears to be {base_answer}"
            else:
                return f"The image suggests: {base_answer}"
                
        except Exception as e:
            logger.error(f"VQA failed: {str(e)}")
            return "Could not generate a proper answer"

    def analyze_image(self, image: Image.Image, question: str = "") -> Tuple[str, str]:
        """Analyze image and return caption and answer."""
        caption = self.generate_caption(image)
        answer = self.answer_question(image, question) if question else "No question was provided"
        return caption, answer

def create_interface():
    """Create Gradio interface with history tab."""
    analyzer = ImageAnalyzer()
    
    def process_inputs(image: Image.Image, question: str) -> Tuple[str, str]:
        # Save uploaded image temporarily
        image_path = f"temp_{datetime.datetime.now().timestamp()}.jpg"
        image.save(image_path)
        
        # Process image
        caption, answer = analyzer.analyze_image(image, question)
        
        # Add to history
        analyzer.add_to_history(image_path, caption, question, answer)
        
        # Format current response
        current_response = (
            f"📝 **Caption**: {caption}\n\n"
            f"💬 **Question**: {question if question else 'None'}\n"
            f"🔍 **Answer**: {answer}"
        )
        
        # Format history (last 5 entries)
        history_entries = analyzer.history[:5]
        history_response = "## Recent Searches\n\n" + "\n\n---\n\n".join(
            f"**{entry['timestamp']}**\n"
            f"📷 Image: {entry['image']}\n"
            f"📝 Caption: {entry['caption']}\n"
            f"💬 Question: {entry['question']}\n"
            f"🔍 Answer: {entry['answer']}"
            for entry in history_entries
        )
        
        return current_response, history_response

    with gr.Blocks(title="🖼️ Image Analyzer") as demo:
        gr.Markdown("# 🖼️✨ Advanced Image Understanding")
        gr.Markdown("Upload an image to get a description and ask questions about it.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="📷 Upload Image")
                question_input = gr.Textbox(
                    label="💬 Ask a Question (optional)",
                    placeholder="What is in this image? What color is...?"
                )
                submit_btn = gr.Button("Analyze")
            
            with gr.Column():
                current_output = gr.Textbox(label="🧠 Current Analysis", lines=5)
                history_output = gr.Markdown(label="⏳ Search History")
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[image_input, question_input],
            outputs=[current_output, history_output]
        )

    return demo

if __name__ == "__main__":
    try:
        interface = create_interface()
        interface.launch()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")