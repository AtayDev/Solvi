def load_prompt_from_txt(file_path):
    """
    Reads and returns the content of the specified file.
    
    :param file_path: The path to the file that contains the prompt.
    :return: The content of the file as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()