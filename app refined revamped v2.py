from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma_Jabran"

PLANNER_PROMPT = """As an expert chemical engineer, create a comprehensive plan to answer the following question. Your plan should cover all relevant aspects, including fundamental concepts, equations, practical applications, and interrelationships between different chemical engineering principles.

##Available Tools##
1. SearchDoc: Search internal documents for relevant information.
2. LLM: Use a language model for analysis or generation tasks.

##Output Format##
#Plan1: <describe your plan here>
#E1: <toolname>[<detailed query, specifying required information, equations, or concepts>]
#Plan2: <describe next plan>
#E2: <toolname>[<input here, you can use #E1 to represent its expected output>]
Continue until you have a comprehensive plan covering all aspects of the question.

##Your Task##
Create an extensive, detailed plan to answer the following question in the context of chemical engineering: {question}

Ensure your plan covers:
1. Fundamental concepts
2. Relevant equations and their derivations
3. Interplay between different concepts
6. Potential challenges or limitations

##Now Begin##
"""

SOLVER_PROMPT = """As an expert chemical engineer, your task is to generate a comprehensive, technically detailed report based on the provided plans and evidence. Your report should be extensive, reaching up to 4000 tokens or more if necessary.

##Plans and Evidences##
{plan_evidence}

##Output Format##
Your report should be structured as follows:
1. Introduction: Briefly introduce the topic and its importance in chemical engineering.
2. Main Body: Divided into relevant sections based on the plans. Each section should:
   - Explain fundamental concepts
   - Present and derive relevant equations
   - Discuss practical applications
   - Explore interrelationships between concepts
   - Include historical context or recent developments when relevant
3. Challenges and Limitations: Discuss any potential challenges or limitations related to the topic.
4. Conclusion: Summarize the key points and their significance in chemical engineering.

Throughout the report:
- Use LaTeX for all equations (enclosed in $$ signs for inline equations and $$ for display equations)
- Provide detailed explanations for complex concepts
- Use specific examples to illustrate points when possible

##Your Task##
Generate a comprehensive chemical engineering report answering the following question: {question}

Ensure your report is technically accurate, detailed, and extensive. Do not hesitate to go into depth on relevant topics.

##Now Begin##
"""

class AIInterface:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.planner_model = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo-16k",  # Using GPT-4 for more sophisticated planning
            max_tokens=500,
        )
        self.solver_model = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo-16k",  # Using GPT-4 for more detailed and accurate responses
            max_tokens=2000,  # Increased to allow for longer outputs
        )
        self.file_db = None
        self.active_db = self.db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def generate_plan(self, query_text: str) -> str:
        planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
        prompt = planner_prompt.format(question=query_text)
        return self.planner_model.predict(prompt)

    def prepare_evidence(self, query_text: str) -> str:
        results = self.active_db.similarity_search_with_relevance_scores(query_text, k=50)  # Increased from 30 to 50
        
        filtered_results = [r for r in results if r[1] >= 0.5]  # Lowered threshold to include more potentially relevant content
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        total_tokens = 0
        max_tokens = 4000  # Increased to allow for more comprehensive evidence
        
        for doc, score in sorted_results:
            chunk_tokens = len(doc.page_content.split())
            if total_tokens + chunk_tokens > max_tokens:
                break
            selected_chunks.append(doc.page_content)
            total_tokens += chunk_tokens
        
        if not selected_chunks:
            return None
        
        return "\n\n---\n\n".join(selected_chunks)

    def execute_plan(self, plan: str, query_text: str) -> str:
        steps = plan.split('\n')
        evidences = []
        for step in steps:
            if step.startswith('#E'):
                tool, query = step.split('[', 1)
                query = query.rstrip(']')
                if 'SearchDoc' in tool:
                    evidence = self.prepare_evidence(query)
                elif 'LLM' in tool:
                    evidence = self.use_llm(query)
                evidences.append(f"{step}\n{evidence}")
        
        plan_evidence = '\n\n'.join(evidences)
        solver_prompt = ChatPromptTemplate.from_template(SOLVER_PROMPT)
        prompt = solver_prompt.format(plan_evidence=plan_evidence, question=query_text)
        return self.solver_model.predict(prompt)

    def use_llm(self, query: str) -> str:
        return self.solver_model.predict(query)

    def generate_detailed_report(self, query_text: str) -> Tuple[str, str]:
        plan = self.generate_plan(query_text)
        print(plan)
        response = self.execute_plan(plan, query_text)
        processed_response = self.process_latex_equations(response)
        
        lines = processed_response.split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()
        
        word_count = len(content.split())
        if word_count < 500:  # Increased minimum word count
            return f"The generated report does not meet the minimum word count requirement. Current word count: {word_count}. Please try again or refine your query.", "Insufficient Content"
        
        return content, title

    def process_latex_equations(self, text: str) -> str:
        def replace_equation(match):
            latex_eq = match.group(1).strip()
            is_inline = not ('\n' in latex_eq or '\\begin{' in latex_eq or '\\end{' in latex_eq)
            
            if is_inline:
                return f'<span class="inline-equation"><img src="https://latex.codecogs.com/svg.latex?{latex_eq}" alt="{latex_eq}" style="vertical-align: middle;"></span>'
            else:
                return f'<div class="display-equation"><img src="https://latex.codecogs.com/svg.latex?{latex_eq}" alt="{latex_eq}"></div>'

        # Process all LaTeX equations (both inline and display)
        text = re.sub(r'\$\$(.*?)\$\$', replace_equation, text, flags=re.DOTALL)
        text = re.sub(r'\$(.*?)\$', replace_equation, text)

        # Wrap non-equation text in paragraph tags
        paragraphs = text.split('\n')
        processed_paragraphs = []
        for paragraph in paragraphs:
            if not paragraph.strip().startswith('<div class="display-equation">'):
                paragraph = f'<p>{paragraph}</p>'
            processed_paragraphs.append(paragraph)

        return '\n'.join(processed_paragraphs)

    def process_file(self, file_path: str):
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                text = file.read()
        else:
            raise ValueError("Unsupported file type")

        chunks = self.text_splitter.split_text(text)
        self.file_db = Chroma.from_texts(chunks, self.embedding_function)
        self.active_db = self.file_db

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        pdf_document = fitz.open(file_path)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text

    def clear_file_data(self):
        self.file_db = None
        self.active_db = self.db


ai_interface = AIInterface()

from flask import Flask, request, jsonify, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os

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
    return render_template('indexios.html')

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
    app.run(debug=True)