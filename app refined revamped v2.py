# Import libraries
from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, url_for
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import openai
import os
import fitz  # PyMuPDF
import re
from typing import List, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os

# Import classes
from interfaces.ai_interface import AIInterface

# Import services
from services import logging_service

# Import utils
from utils.U1_FilesUtils import load_prompt_from_txt

# Load environment variables
load_dotenv()
CHROMA_PATH = "chroma_Jabran"
PLANNER_PROMPT = load_prompt_from_txt('prompts/planner_prompt.txt')
SOLVER_PROMPT = load_prompt_from_txt('prompts/solver_prompt.txt')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

ai_interface = AIInterface(
    chroma_path=CHROMA_PATH,                       # Path to the Chroma database
    planner_prompt_txt=PLANNER_PROMPT,             # Custom planner prompt
    solver_prompt_txt=SOLVER_PROMPT,               # Custom solver prompt             
    planner_model_name="gpt-3.5-turbo-16k",        # Use GPT-4 instead of GPT-3.5
    solver_model_name="gpt-3.5-turbo-16k",         # Solver model set to GPT-3.5-turbo
    express_model_name="gpt-3.5-turbo",            # Express model set to GPT-3.5-turbo
    planner_temp=0.7,                              # Higher temperature for planner (creativity)
    solver_temp=0.2,                               # Lower temperature for solver (accuracy)
    express_temp=0.3,                              # Express model with moderate temperature
    planner_max_tokens=10000,                        # Increase token limit for planner model
    solver_max_tokens=15000,                        # High token limit for solver model
    express_max_tokens=500                         # Set token limit for express model
)

# Step 3: Create the /logs endpoint
@app.route('/logs')
def logs():
    try:
        return logging_service.get_logs()
    except Exception as e:
        logging_service.log_exception(e)
        return  "An error occurred while retrieving logs.", 500

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    """Clear the log file when the button is clicked."""
    try:
        logging_service.clean_logs()
        # Redirect back to the /logs page after clearing
        return redirect(url_for('logs'))
    except Exception as e:
        logging_service.log_exception(e)
        return "An error occurred while clearing logs.", 500
    
@app.route('/logs/data')
def get_log_data():
    try:
        # Just call the logging service to return raw log data
        return logging_service.get_logs_data()
    except Exception as e:
        logging_service.log_exception(e)
        return "An error occurred while retrieving logs.", 500
    


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    content = data.get('content', '')

    pdf_path = r"C:\Users\AyoubFrikhat\Downloads\Solvi.v2\Solvi Latex Jabran\res.pdf"

    try:
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        justified_style = ParagraphStyle(
            name='Justified',
            parent=styles['BodyText'],
            alignment=4  # Justify
        )

        subtitle_style = ParagraphStyle(
            name='Subtitle',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            textColor='#052179',
            fontName='Helvetica-Bold'
        )

        story = []
        story.append(Paragraph("Solvi Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Splitting the content into paragraphs
        paragraphs = content.split('\n\n')
        for i, part in enumerate(paragraphs):
            part = part.strip()
            if part.startswith("**") and part.endswith("**"):
                # Removing stars and making the subtitle bold
                subtitle = part.strip('**').strip()
                story.append(Paragraph(subtitle, subtitle_style))
                story.append(Spacer(1, 6))
            else:
                if i > 0:
                    story.append(Spacer(1, 6))  # Add space between paragraphs
                story.append(Paragraph(part, justified_style))
                story.append(Spacer(1, 12))

        doc.build(story)

        # Serve the PDF file
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        # If there is any error, return a 500 error with the message
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    try:
        logging_service.log_message("info", "Home route accesssed")
        return render_template('indexios.html')
    except Exception as e:
        logging_service.log_exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data['query']
    is_detailed = data['isDetailed']
    
    if is_detailed:
        response, title = ai_interface.generate_detailed_report(query_text)
    else:
        response, title = ai_interface.generate_express_info(query_text)
    
    return jsonify({'response': response, 'title': title})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        try:
            ai_interface.process_file(file_path)
            return jsonify({'message': f'File {filename} processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/clear', methods=['POST'])
def clear_file_data():
    ai_interface.clear_file_data()
    return jsonify({'message': 'File data cleared'})

if __name__ == '__main__':
    try:
        # Clean logs on startup
        logging_service.clean_logs()
        logging_service.log_message('info', "###########################################")     
        logging_service.log_message('info', "Starting Flask app with debug mode")
        app.run(debug=True)
    except Exception as e:
        logging_service.log_exception(e)