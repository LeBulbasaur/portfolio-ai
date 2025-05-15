#!/usr/bin/env python3
from ollama import chat, ChatResponse
import csv
import time
import random
import argparse

parser = argparse.ArgumentParser(description='Makes random dataset rows. Warning! May create duplicates!')
parser.add_argument('listings', help='csv file containing job listings')
parser.add_argument('resumes', help='csv file containing resumes')
parser.add_argument('output', help='output file')
parser.add_argument('model', help='used model')



def get_score(resume: str, listing: str, model: str) -> float:
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': f'I am applying for this job: {listing}. This is my resume: {resume}. Give me an overall score on a 1-100 scale and nothing more',
        },
    ])
    return response.message.content


def main(): 
    args = parser.parse_args()
    with open(args.listings, newline='') as listings:
        listings = [row[1] for row in csv.reader(listings)]

    with open(args.resumes, newline='') as resumes:
        resumes = [row[1] for row in csv.reader(resumes)]
    # print(resume_reader[2])
    i = 1
    timestart = time.time()
    last_passed = timestart
    while i:
        resume = random.choice(resumes)
        listing = random.choice(listings)
        with open(args.output, 'a', newline='') as dataset:
            writer = csv.writer(dataset, delimiter=',')
            writer.writerow([listing, resume, get_score(resume, listing, args.model)])

        now = time.time()
        passed_secs = now-timestart
        total_passed = time.strftime("%H:%M:%S", time.gmtime(passed_secs))
        passed = time.strftime("%H:%M:%S", time.gmtime(time.time()-last_passed))
        last_passed = now

        
        print(f'row: {i}, time spent: {passed} ({total_passed} in total)')
        i+=1
if __name__ == '__main__':
    main()


    