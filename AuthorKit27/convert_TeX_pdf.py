# Python script for converting .TeX files to .pdf files using python libraries

# Implementation:
from asyncio import subprocess
import os


def convert_tex_to_pdf(tex_file):
    """
    Convert a .TeX file to a .pdf file using pdflatex command-line tool.
    """
    if not tex_file.endswith(".tex"):
        raise ValueError("Input file must be a .tex file")
    
    pdf_file = tex_file.replace(".tex", ".pdf")
    subprocess.run(["pdflatex", tex_file], check=True)
    return pdf_file


