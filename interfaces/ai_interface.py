import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import fitz  # PyMuPDF for PDF handling
from typing import Tuple, Optional
import concurrent.futures


class AIInterface:
    def __init__(self, chroma_path: str, planner_prompt_txt: str, solver_prompt_txt: str, 
                 planner_model_name: str = "gpt-3.5-turbo-16k", solver_model_name: str = "gpt-3.5-turbo-16k", express_model_name: str = "gpt-3.5-turbo",
                 planner_temp: float = 0.7, solver_temp: float = 0.2, express_temp: float = 0.3,
                 planner_max_tokens: int = 500, solver_max_tokens: int = 3000, express_max_tokens: int = 450):
        """
        Constructor for AIInterface class. Takes parameters for model configurations and paths.
        """
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)

        # Store planner and solver prompts
        self.planner_prompt_txt = planner_prompt_txt
        self.solver_prompt_txt = solver_prompt_txt

        # Initialize models with passed parameters
        self.planner_model = ChatOpenAI(
            temperature=planner_temp,
            model=planner_model_name,
            max_tokens=planner_max_tokens,
        )
        self.solver_model = ChatOpenAI(
            temperature=solver_temp,
            model=solver_model_name,
            max_tokens=solver_max_tokens,
        )
        self.express_model = ChatOpenAI(
            temperature=express_temp,
            model=express_model_name,
            max_tokens=express_max_tokens,
        )

        self.file_db = None
        self.active_db = self.db
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def generate_express_info(self, query_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a concise express response based on the query and evidence.
        """
        # Prepare evidence from the Chroma database
        evidence = self.prepare_evidence_express(query_text)
        if not evidence:
            return None, "No relevant evidence found."

        express_prompt = f"""
        As an expert chemical engineer, provide a concise overview answering the following question in approximately 350 words:
        
        Question: {query_text}
        
        Use the following relevant information to support your answer:
        
        {evidence}
        
        Your response should:
        1. Briefly introduce the topic
        2. Highlight key concepts and their significance in chemical engineering
        3. Mention any relevant equations without going into derivations
        4. Conclude with the practical implications or applications
        - Use LaTeX for all equations (enclosed in $$ signs for inline equations and $$ for display equations)
        
        Ensure your answer is informative yet accessible to someone with a basic understanding of chemical engineering.
        """

        try:
            response = self.express_model.predict(express_prompt)
        except openai.APIError as e:
            return None, f"Error while generating express info: {str(e)}"

        # Process any LaTeX equations in the response
        processed_response = self.process_latex_equations(response)

        # Extract title (first line) and content (rest of the response)
        lines = processed_response.split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()

        return content, title

    def generate_plan(self, query_text: str) -> str:
        """
        Generates a plan based on the input query using the planner model.
        """
        planner_prompt = ChatPromptTemplate.from_template(self.planner_prompt_txt)
        prompt = planner_prompt.format(question=query_text)

        try:
            return self.planner_model.predict(prompt)
        except openai.APIError as e:
            return f"Error while generating the plan: {str(e)}"

    def prepare_evidence(self, query_text: str) -> Optional[str]:
        """
        Prepares evidence by fetching relevant documents from the database and processing them.
        """
        results = self.active_db.similarity_search_with_relevance_scores(query_text, k=20)
        filtered_results = [r for r in results if r[1] >= 0.5]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        selected_chunks = []
        total_tokens = 0
        max_tokens = 4000

        for doc, score in sorted_results:
            chunk_tokens = len(doc.page_content.split())
            if total_tokens + chunk_tokens > max_tokens:
                break
            selected_chunks.append(doc.page_content)
            total_tokens += chunk_tokens

        if not selected_chunks:
            return None
        
        full_evidence = "\n\n---\n\n".join(selected_chunks)
        return self.chunk_and_process(full_evidence, self.solver_model, 15000)

    

    def chunk_and_process(self, text: str, model, max_tokens_per_chunk: int, max_workers: int = 5) -> str:
        """
        Splits the input text into smaller chunks and processes them in parallel using the model.
        """
        chunks = self.text_splitter.split_text(text)
        
        # Define a worker function to process a single chunk
        def process_chunk(chunk):
            chunk_tokens = len(chunk.split())
            if chunk_tokens > max_tokens_per_chunk:
                chunk = chunk[:max_tokens_per_chunk]  # Truncate chunk if too large
            try:
                return model.predict(chunk)
            except openai.APIError as e:
                return f"Error processing chunk: {str(e)}"
        
        # Use ThreadPoolExecutor to process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return "\n\n".join(results)


    def execute_plan(self, plan: str, query_text: str) -> str:
        """
        Executes the plan by fetching and processing the evidence.
        """
        steps = plan.split('\n')
        evidences = []

        for step in steps:
            if step.startswith('#E'):
                tool, query = step.split('[', 1)
                query = query.rstrip(']')
                if 'SearchDoc' in tool:
                    evidence = self.prepare_evidence(query_text)
                elif 'LLM' in tool:
                    evidence = self.use_llm(query)
                evidences.append(f"{step}\n{evidence}")

        plan_evidence = '\n\n'.join(evidences)
        solver_prompt = ChatPromptTemplate.from_template(self.solver_prompt_txt)
        prompt = solver_prompt.format(plan_evidence=plan_evidence, question=query_text)

        try:
            return self.solver_model.predict(prompt)
        except openai.APIError as e:
            return f"Error executing plan: {str(e)}"

    def use_llm(self, query: str) -> str:
        """
        Uses the LLM model to generate a response based on the query.
        """
        try:
            return self.solver_model.predict(query)
        except openai.APIError as e:
            return f"Error in LLM model: {str(e)}"

    def generate_detailed_report(self, query_text: str) -> Tuple[str, str]:
        """
        Generates a detailed report based on the input query using the planner and solver models.
        """
        plan = self.generate_plan(query_text)
        response = self.execute_plan(plan, query_text)
        processed_response = self.process_latex_equations(response)

        lines = processed_response.split('\n')
        title = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()

        word_count = len(content.split())
        if word_count < 500:
            return f"The generated report does not meet the minimum word count requirement. Current word count: {word_count}. Please try again or refine your query.", "Insufficient Content"

        return content, title

    def process_latex_equations(self, text: str) -> str:
        """
        Processes LaTeX equations in the text and converts them to rendered images.
        """
        def replace_equation(match):
            latex_eq = match.group(1).strip()
            is_inline = not ('\n' in latex_eq or '\\begin{' in latex_eq or '\\end{')

            if is_inline:
                return f'<span class="inline-equation"><img src="https://latex.codecogs.com/svg.latex?{latex_eq}" alt="{latex_eq}" style="vertical-align: middle;"></span>'
            else:
                return f'<div class="display-equation"><img src="https://latex.codecogs.com/svg.latex?{latex_eq}" alt="{latex_eq}"></div>'

        text = re.sub(r'\$\$(.*?)\$\$', replace_equation, text, flags=re.DOTALL)
        text = re.sub(r'\$(.*?)\$', replace_equation, text)

        paragraphs = text.split('\n')
        processed_paragraphs = []
        for paragraph in paragraphs:
            if not paragraph.strip().startswith('<div class="display-equation">'):
                paragraph = f'<p>{paragraph}</p>'
            processed_paragraphs.append(paragraph)

        return '\n'.join(processed_paragraphs)

    def process_file(self, file_path: str):
        """
        Processes a file (PDF or TXT) by extracting its text and splitting it into chunks.
        """
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
        """
        Extracts text from a PDF file.
        """
        pdf_document = fitz.open(file_path)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text

    def clear_file_data(self):
        """
        Clears the file database and resets to the original active database.
        """
        self.file_db = None
        self.active_db = self.db
