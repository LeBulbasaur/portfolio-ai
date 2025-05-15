from ollama import chat, ChatResponse
import csv
import time
import random

with open('listings:27b2.csv', newline='') as listings:
    listing_reader = [row[1] for row in csv.reader(listings)]

with open('resumes:27b.csv', newline='') as resumes:
    resume_reader = [row[1] for row in csv.reader(resumes)]


def get_score(resume: str, listing: str) -> float:
    response: ChatResponse = chat(model='gemma3:12b', messages=[
        {
            'role': 'user',
            'content': f'I am applying for this job: {listing}. This is my resume: {resume}. Give me an overall score on a 1-100 scale and nothing more',
        },
    ])
    return response.message.content


def main(): 
    # print(resume_reader[2])
    total = len(resume_reader)*len(listing_reader)
    i = 1
    timestart = time.time()
    for resume in resume_reader:
        # print(resume)
        for listing in listing_reader:
            with open('dataset:12b.csv', 'a', newline='') as listings:
                writer = csv.writer(listings, delimiter=',')
                writer.writerow([listing, resume, get_score(resume, listing)])
            passed_secs = time.time()-timestart
            passed = time.strftime("%d:%H:%M:%S", time.gmtime(passed_secs))
            eta = time.strftime("%d:%H:%M:%S", time.gmtime(passed_secs*(total/i -1)))
            print(f'row: {i}/{total}, ({i/total*100:,.2f}%), time passed: {passed}, ETA: {eta}')
            i+=1
if __name__ == '__main__':
    main()


    