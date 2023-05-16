#!/usr/bin/env python
# coding: utf-8


#milad
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import nltk
import string
import re
import spacy 
from spacy.matcher import Matcher





def pre_process(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 1)

    edge = cv2.Canny(blur, 135, 200)

    kernal = np.ones((5, 5))

    dilate = cv2.dilate(edge, kernal, iterations=2)

    threshold = cv2.erode(dilate, kernal, iterations=1)

    return threshold




def get_contours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    points = cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)
    return biggest





def reshape(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points





def fix_image(img, contour):
    contour = reshape(contour)

    pts1 = np.float32(contour)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # get the warp perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # apply to the image
    final_img = cv2.warpPerspective(img, matrix, (width, height))

    #crop the image
    cropped = final_img[20:final_img.shape[0] - 20, 20:final_img.shape[1] - 20]

    return cropped




file_name = 'Brilliant AC.jpg'

img = cv2.imread(file_name)
print(img.shape)
img_contour = img.copy()





scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)





threshold = pre_process(img)
get_contour = get_contours(threshold)
fixed_image = fix_image(img, get_contour)





text = pytesseract.image_to_string(img)
print('text detected: \n' + text)



nlp = spacy.load('en_core_web_sm')
doc = nlp(text)


words = ['work','done']
for ent in doc.ents:
    if ent.label_ == "GPE" and all(word in str(doc) for word in words):
        location = ent



words = ['work','guaranteed']
for ent in doc.ents:
    if ent.label_ == "DATE" and all(word in str(doc) for word in words):
        guaranteed = ent



sentences = [x for x in doc.sents]
words = ['work','carried out','between']
for sen in sentences:
    if all(word in str(sen) for word in words):
        text_1 = sen

date_1 = re.findall(r'\d{1,2} (?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2,4}',str(text_1))[0]
date_2 = re.findall(r'\d{2,4}[/-]\d{2}[/-]\d{2,4}',str(text_1))[0]


from dateutil.parser import parse
if parse(date_1).strftime('%Y/%m/%d')<date_2:
    dates_between = date_1+' and '+date_2
else:
    dates_between = date_2+' and '+date_1



dates_list = re.findall(r'\d{1,2} (?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2,4}',str(text))
dates_list.append(re.findall(r'\d{2,4}[/-]\d{2}[/-]\d{2,4}',str(text))[0])



d = []
for date in dates_list:
    d.append(parse(date).strftime('%Y/%m/%d'))




for date in dates_list:
    if parse(date).strftime('%Y/%m/%d')==min(d):
        document_date = date



matcher = Matcher(nlp.vocab)

matcher.add("ORG", [[{'POS': 'PROPN', 'OP':'+'}, {'TEXT': {'REGEX': '(?i)^(?:LTD)$'}}]])

doc = nlp(text)
matches = matcher(doc)
spans = [doc[start:end] for match_id, start, end in matches]
for span in spacy.util.filter_spans(spans):
    company_name = span.text



matcher = Matcher(nlp.vocab)

phone_pattern = [{"ORTH": "","OP":"?"}, {"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}]
matcher.add("PHONE_NUMBER", [phone_pattern])

doc = nlp(text)
matches = matcher(doc)
spans = [doc[start:end] for match_id, start, end in matches]
for span in spacy.util.filter_spans(spans):
    contact_number = span.text



matcher = Matcher(nlp.vocab)


email_pattern = [{'LIKE_EMAIL': True}]
matcher.add("email_pattern", [email_pattern])


doc = nlp(text)
matches = matcher(doc)
spans = [doc[start:end] for match_id, start, end in matches]
for span in spacy.util.filter_spans(spans):
    contact_email = span.text



nlp.add_pipe("merge_entities", after="ner") 
matcher = Matcher(nlp.vocab)
contact_pattern = [{"POS": "VERB","OP":"+"}, {"ENT_TYPE": "PERSON"}]
matcher.add("contact_pattern", [contact_pattern])

doc = nlp(text)
matches = matcher(doc)
spans = [doc[start+1:end] for match_id, start, end in matches]
for span in spacy.util.filter_spans(spans):
    contact_person = span.text



print(f"company_name: {company_name}")
print(f"document_date: {document_date}")
print(f"location: {location}")
print(f"dates_between: {dates_between}")
print(f"contact_person: {contact_person}")
print(f"contact_email: {contact_email}")
print(f"contact_number: {contact_number}")
print(f"guaranteed: {guaranteed}")







