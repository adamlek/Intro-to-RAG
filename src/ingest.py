import os
import pathlib

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SentenceSplitter, 
    MarkdownNodeParser
)

from itertools import chain

import pymupdf4llm
from docx2md import (
    DocxFile,
    DocxMedia,
    Converter
)

from IPython import embed

def docx_convert(
    docx_file: str, 
    target_dir: str = "", 
    use_md_table: bool = False
) -> str:
    """
    Convert a DOCX file to Markdown format.

    Args:
        docx_file (str): Path to the DOCX file.
        target_dir (str, optional): Directory to save media files. Defaults to "".
        use_md_table (bool, optional): Whether to use Markdown tables. Defaults to False.

    Returns:
        str: The converted Markdown text or an exception message.
    """
    try:
        docx = DocxFile(docx_file)
        media = DocxMedia(docx)
        if target_dir:
            media.save(target_dir)
        converter = Converter(docx.document(), media, use_md_table)
        return converter.convert()
    except Exception as e:
        return f"Exception: {e}"

def parse_documents_in_folder(
    folder: str
) -> list[tuple]:
    """
    Parse all documents in a folder.

    Args:
        folder (str): Path to the folder containing documents.

    Returns:
        list[tuple]: List of tuples with parsed document, status, and path.
    """
    return [parse_to_md(folder+file) 
            for file 
            in os.listdir(folder)]

def parse_to_md(path: str) -> tuple:
    """
    Parse a document to Markdown or text, based on file extension.

    Args:
        path (str): Path to the document.

    Returns:
        tuple: (Document or error message, status, path)
    """
    extractors = {
        'pdf': pymupdf4llm.to_markdown, 
        'docx': docx_convert, 
        'md': lambda x: open(x).read()
    }
    ext = path.split('.')[-1]
    if ext not in extractors:
        return ('extension not available', False, path)
    return (Document(text=extractors[ext](path)), True, path)

def chunk_docs(
    documents: list[str], 
    max_len: int = 256,
) -> list:
    """
    Chunk documents into smaller nodes for processing.

    Args:
        documents (list[str]): List of document texts.
        max_len (int, optional): Maximum chunk length. Defaults to 256.

    Returns:
        list: List of chunked nodes.
    """
    node_parser = MarkdownNodeParser()
    fallback_parser = SentenceSplitter(chunk_size=max_len, chunk_overlap=20)

    nodes = node_parser(documents)
    _nodes = []
    for n in nodes:
        if len(n.get_content()) > max_len:
            _nodes += fallback_parser([n])
        else:
            _nodes.append(n)
    return _nodes

def parse_and_get_nodes(folder: str) -> list:
    """
    Parse documents in a folder and return chunked nodes.

    Args:
        folder (str): Path to the folder.

    Returns:
        list: List of chunked nodes.
    """
    documents = parse_documents_in_folder(folder)
    documents, *_ = list(zip(*documents))
    return chunk_docs(documents)

def node_lens(nodes: list) -> None:
    """
    Print the length of content in each node.

    Args:
        nodes (list): List of nodes.
    """
    word_lens = [len(x.get_content()) for x in nodes]
    print('Number of "words" in chunks', word_lens)
        
if __name__ == '__main__':
    """
    Main execution block:
    - Parses documents in the 'data/' folder.
    - Chunks the documents into nodes.
    - Prints the length of each node.
    """
    folder = 'data/'
    nodes = parse_and_get_nodes(folder)
    node_lens(nodes)