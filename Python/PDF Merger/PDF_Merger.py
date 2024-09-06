import PyPDF2

pdf_files = ["1.pdf" , "2.pdf"]
merger = PyPDF2.PdfMerger()
for filename in pdf_files:
    pdfFile = open(filename,'rb')
    pdfReader = PyPDF2.PdfReader(pdfFile)
    merger.append(pdfReader)

pdfFile.close()
merger.write('New_Merged.pdf')
