#!/usr/bin/env python3
from ollama import chat, ChatResponse
import csv
import time
import argparse

parser = argparse.ArgumentParser(description='Makes resumes for each category')
parser.add_argument('model', help='used model')
parser.add_argument('--output', help='output file (rows will be appended)', default='resumes.csv')
parser.add_argument('-n', type=int, help='number of resumes for each category', default=4)

categories = [
    'Advocate',
    'Arts',
    'Automation Testing',
    'Blockchain',
    'Business Analyst',
    'Civil Engineer',
    'Data Science',
    'Database',
    'DevOps Engineer',
    'DotNet Developer',
    'Electrical Engineering',
    'ETL Developer',
    'Hadoop',
    'Health and fitness',
    'HR',
    'Java Developer',
    'Mechanical Engineer',
    'Network Security Engineer',
    'Operations Manager',
    'PMO',
    'Python Developer',
    'Sales',
    'SAP Developer',
    'Testing',
    'Web Designing'
]

def get_listing(category: str, model: str, out: str) -> str:
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': f'write me an example CV for someone applying for a job as a {category}. max 150 words. it is very important you do not leave any blanks for me to fill.',
        },
    ])
    with open(out, 'a', newline='') as listings:
        writer = csv.writer(listings, delimiter=',')
        writer.writerow([category, response.message.content])


def main(): 
    args = parser.parse_args()
    n = args.n
    total = len(categories)*n
    i = 1
    timestart = time.time()
    for _ in range(n):
        for category in categories:
            passed = time.strftime("%H:%M:%S", time.gmtime(time.time()-timestart))
            print(f'({i}/{total}, {i/total*100:,.2f}%, {passed}) Creating resume for {category}')
            get_listing(category=category, model=args.model, out=args.output)
            i+=1

    print('Done!')
if __name__ == '__main__':
    main()


    