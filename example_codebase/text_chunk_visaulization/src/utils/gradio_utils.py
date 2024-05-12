import gradio as gr
from transformers import AutoTokenizer
from config.constant import MODEL_NAME, LABEL_RECURSIVE, LABEL_TEXTSPLITTER
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from src.text_overlapper import unoverlap_list


bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def length_tokens(txt):
    return len(bert_tokenizer.tokenize(txt))

def extract_separators_from_string(separators_str):
    try:
        separators_str = separators_str.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\") # fix special characters
        separators = separators_str[1:-1].split(", ")
        return [separator.replace('"', "").replace("'", "") for separator in separators]
    except Exception as e:
        raise gr.Error(f"""
        Did not succeed in extracting seperators from string: {separators_str} due to: {str(e)}.
        Please type it in the correct format: "['separator_1', 'separator_2', ...]"
        """)

def change_split_selection(split_selection):
    return (
        gr.Textbox.change(visible=(split_selection==LABEL_RECURSIVE)),
        gr.Radio.change(visible=(split_selection==LABEL_RECURSIVE)),
    )

def chunk(text, length, splitter_selection, separators_str, length_unit_selection, chunk_overlap):
    separators = extract_separators_from_string(separators_str)
    length_function = (length_tokens if "token" in length_unit_selection.lower() else len)
    if splitter_selection == LABEL_TEXTSPLITTER:
        text_splitter = CharacterTextSplitter(
            chunk_size=length,
            chunk_overlap=int(chunk_overlap),
            length_function=length_function,
            strip_whitespace=False,
            is_separator_regex=False,
            separator=" ",
        )
    elif splitter_selection == LABEL_RECURSIVE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=length,
            chunk_overlap=int(chunk_overlap),
            length_function=length_function,
            strip_whitespace=False,
            separators=separators,
        )
    splits = text_splitter.create_documents([text])
    text_splits = [split.page_content for split in splits]
    unoverlapped_text_splits = unoverlap_list(text_splits)
    output = [((split[0], 'Overlap') if split[1] else (split[0], f"Chunk {str(i)}")) for i, split in enumerate(unoverlapped_text_splits)]
    return output

def change_preset_separators(choice):
    text_splitter = RecursiveCharacterTextSplitter()
    if choice == "Default":
        return ["\n\n", "\n", " ", ""]
    elif choice == "Markdown":
        return text_splitter.get_separators_for_language(Language.MARKDOWN)
    elif choice == "Python":
        return text_splitter.get_separators_for_language(Language.PYTHON)
    else:
        raise gr.Error("Choice of preset not recognized.")