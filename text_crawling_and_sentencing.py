# -*- coding: utf-8 -*-
"""
* [Link](https://www.analyticsvidhya.com/blog/2021/09/pypdf2-library-for-working-with-pdf-files-in-python/) including: text extraction, read page, merge, split pages
"""
#%% Slake
""
#Using slake to get whole text [m,[filename,text]]
#For one-column documents
import slate3k as slate
import os

texts=[]
folder = 'C:\\Users\\smrya\Desktop\\博士\\Journal article\\Journal article 2\\Data\\4. books\\'
N=0
for filename in os.listdir(folder):
    N+=1; print (N)
    path = os.path.join(folder, filename)
    with open(path,'rb') as f:
        text = slate.PDF(f)
        text=' '.join(text) #join the text into a whole string
    texts.append(text) #return [m,text]
    
texts[0][0:100]
#%% PyPDF2
''
# PyPDF2
# not that good
from PyPDF2 import PdfFileReader
texts=[]
folder = 'C:\\Users\\smrya\Desktop\\博士\\Journal article\\Journal article 2\\Data\\4. books\\'
N=0
for filename in os.listdir(folder):
    N+=1; print (N)
    if N==1:
        path = os.path.join(folder, filename)
        reader = PdfFileReader(path)
        number_of_pages = reader.numPages
        text=''
        for i in range (number_of_pages):
            page = reader.pages[i]
            page_text = page.extractText()
            text=text+' '+page_text
        texts.append(text)
text

#%% Write file to local
''
# for writing the file to local
textfile = open("image_just_text.txt", "w", encoding='utf-8')
for element in texts:
    textfile.write(element+"\n"+"Another")
textfile.close()

# for storing in pt file
import torch
torch.save(sentences, 'C:\\Users\\smrya\Desktop\\sentences.pt')

#%% Pytesseract
''
# Using pytesseract
# for 2-column pdf
# pdf-image-text
import pytesseract
import os
from pdf2image import convert_from_path

folder = 'C:\\Users\\smrya\Desktop\\博士\\Journal article\\Journal article 2\\Data\\6. reports\\'
#for setting the path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path=r'C:\Users\smrya\anaconda3\Lib\site-packages\poppler-0.68.0\poppler-0.68.0\bin'

texts=[]
N=0
for filename in os.listdir(folder):
    N+=1;print (N)
    path = os.path.join(folder, filename)
    images = convert_from_path(path,poppler_path=poppler_path)
    text = ''
    for i in range(len(images)):  
        page_content = pytesseract.image_to_string(images[i])
        text = text + ' ' + page_content
    texts.append(text) #return [m,text]]
    
len(texts)
#%% Read txtfile of reports
''
# for reports, I manually copy the text and save them into a folder, and then process them
N=0
folder = 'C:\\Users\\smrya\Desktop\\博士\\Journal article\\Journal article 2\\Data\\6. reports\\txtfile\\'
texts=[]
for filename in os.listdir(folder):
    N+=1
    print (N)
    path = os.path.join(folder, filename)
    with open(path,'r',encoding='utf-8') as f:
        text = f.read()
        text=text.replace('Ina', 'In a')
        text=text.replace('  ',' ')
        text=text.replace('  ',' ')     
        text=text.replace('\x02','')  
        text=text.replace('\x0c','')  
        text=text.replace('\uf0b7 ','')  

    texts.append(text)

texts[0][100:500]

#%% mysentences to split the text into sentence segmentation
''
#to get individual sentences [m,sentences]
import mysentences
sentences=[]
for text in texts:
    sentence=mysentences.split_into_sentences(text) #[1,sentence]
    #remove the sentences whose lengths are shorter than a certain number
    sentence=[i for i in sentence if len(i.split(' '))>6] #[1,sentence]
    sentences.append(sentence) #[m,sentences]
