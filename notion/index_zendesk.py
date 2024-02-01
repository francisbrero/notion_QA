import os
import pandas as pd
import csv
import html2text
import sys
import requests
import uuid

from io import StringIO
from pprint import pprint
from bs4 import BeautifulSoup
import argparse
from transformers import GPT2TokenizerFast
from typing import Tuple
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add an argument with a flag and a name
parser.add_argument("--zendesk", nargs="*", default=["madkudusupport"], help="Specify the Zendesk domains you want to index")
parser.add_argument("--max_pages", default=1000, help="The maximum amount of Zendesk pages to index")
parser.add_argument("--out", default="./zendesk_data/contents.csv", help="Specify the filename to save the content")
parser.add_argument("--min_tokens", default=20, help="Remove content with less than this number of tokens")
parser.add_argument("--input", default="./input", help="Folder to ingest CSVs from. Rows should be in the format 'heading,answers,answers,...'")

args = parser.parse_args()
max_pages = int(args.max_pages)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


def extract_html_content(
  title_prefix: str,
  page_title: str,
  html: str,
  url: str
):
  nuuids, ncontents, nurls = [], [], []

  soup = BeautifulSoup(html, 'html.parser')
  headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

  prev_heading = []

  # Iterate through all headings and subheadings
  for h in headings:
    # Extract the heading text and remove HTML
    heading = html2text.html2text(str(h)).strip()

    # Initialize the content list
    content = []

    # Find the next heading or subheading
    next_h = h.find_next(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    actual_heading = heading.lstrip('#').lstrip(' ')

    # Iterate through all siblings until the next heading or subheading is reached
    for sibling in h.next_siblings:
      if sibling == next_h:
        break

      # If the sibling is a tag, extract the text and remove HTML
      if sibling.name:
        para = html2text.html2text(str(sibling)).strip()
        if len(para) > 0:
          content.append(para)

    # If there are content entries, join them all together, clean up for utf-8 and write the row
    if len(content) > 0:
      content = "".join(content).replace("\n", "").encode('utf-8').decode('utf-8')

      # If there are headings above this one without content, we concat them here
      if len(prev_heading) > 0:
        full_heading = " - ".join(prev_heading) + " - " + actual_heading
      else:
        full_heading = actual_heading

      title = f"{title_prefix} - {page_title}"
      # Store the extracted title, heading, content
      row_uuid = str(uuid.uuid4())
      nuuids.append(row_uuid)
      ncontents.append(f"{title} - {full_heading} - {content}")
      nurls.append(url)
      prev_heading = []
    else:
      # Otherwise, we store this heading to append to the next sibling with content
      prev_heading.append(actual_heading)
  
  # Return the 3 arrays of titles, headings and content
  return (nuuids, ncontents, nurls)

def count_content_tokens(
  nuuids:list,
  ncontents: list,
  nurls: list
):
  """This function takes a list of tuples (title, heading, content, tokens) and returns a list of tuples (title, heading, content, tokens)"""
  # count the tokens of each section
  ncontent_ntokens = [
      count_tokens(c) # Add the tokens from the content
      + 4
      + count_tokens(" ".join(id.split(" ")[1:-1])) # Add the tokens from the headings
      + count_tokens(" ".join(u.split(" ")[1:-1])) # Add the tokens from the url
      - (1 if len(c) == 0 else 0)
      for id, c, u in zip(nuuids, nurls, ncontents)
  ]
  # Create a tuple of (title, section_name, content, number of tokens)
  outputs = []
  outputs += [(id, u, c, tk) if tk<max_len 
              else (id, reduce_long(c, max_len), count_tokens(reduce_long(c,max_len))) 
                  for id, u, c, tk in zip(nuuids, nurls, ncontents, ncontent_ntokens)]
  return outputs

def extract_zendesk_domain(
  zendesk_domain: str,
  limit: int = max_pages
):
  """This function extracts the content from a Zendesk domain and returns a list of tuples (title, heading, content, tokens)"""
  nuuids, ncontents, nurls = [], [], []

  total_pages = 0;
  URL = f"https://{zendesk_domain}.zendesk.com/api/v2/help_center/en-us"
  
  print(f"Fetching up to {limit} pages from 'https://{zendesk_domain}.zendesk.com'...")

  # Fetch the Categories from Zendesk
  cat_response = requests.get(URL + '/categories.json')
  cat_data = cat_response.json()
  for category in cat_data['categories']:
    category_title = category['name']

    # Fetch the sections within the categories
    sections_response = requests.get(URL + '/categories/' + str(category['id']) + '/sections.json')
    sections_data = sections_response.json()
    for section in sections_data['sections']:
      page_title = section['name']
      
      # Fetch the articles within the section
      articles_response = requests.get(URL + '/sections/' + str(section['id']) + '/articles.json')
      articles_data = articles_response.json()

      for article in articles_data["articles"]:
        page_title += " - " + article['title']
        page_html = article['body']
        page_url = article['html_url']

        if (page_html is not None and total_pages < limit ):
          pageIds, pageContent, pageUrls = extract_html_content(category_title, page_title, page_html, page_url)
          nuuids += pageIds
          ncontents += pageContent
          nurls += pageUrls
          total_pages += 1
      
      if (articles_data['next_page'] is not None):
        pprint('TODO! But have not seen multiple pages yet at this level (due to using sections...)')
  
  return count_content_tokens(nuuids, ncontents, nurls)

def extract_csvfile(subdir, file):
    """This function takes our CSV file and returns a list of tuples (title, heading, content, tokens)"""
    nuuids, ncontents, nurls = [], [], []
    csv_filepath = os.path.join(subdir, file)
    subdir_name = os.path.basename(subdir)
    file_name = os.path.splitext(file)[0]
    
    print(f"Loading data from {csv_filepath}, subdir: {subdir_name}")
    
    title = f"{subdir_name} - {file_name}"

    with open(csv_filepath, 'r', encoding='utf-8') as csv_file:
      csv_reader = csv.reader(csv_file)
      for row in csv_reader:
        if row:
          row_uuid = str(uuid.uuid4())
          content = ""
          if row[0]:
            content += f"{row[0]} -"
          if len(row) > 1 and row[1]:
            content += row[1]
          for i in range(2, len(row)):
            if row[i]:
              content += ' ' + row[i]
          
          # If the content is empty, move on
          if (len(content) > 0):
            content = f"{title} - {content}"
          else:
            continue
          
          nuuids.append(row_uuid)
          ncontents.append(content)
          nurls.append(file)
    return count_content_tokens(nuuids, ncontents, nurls)


# Define the maximum number of tokens we allow per row
max_len = 1500

# For each Space, fetch the content and add to a list(title, heading, content, tokens)
res = []

for domain in args.zendesk:
  print(f"INDEXING CONTENT FROM ZENDESK: {domain}.zendesk.com")
  res += extract_zendesk_domain(domain)

if os.path.isdir(args.input):
  for subdir, dirs, files in os.walk(args.input):
    for file in files:
      if file.endswith(".csv"):
        res += extract_csvfile(subdir, file)
      elif file.endswith(".pdf"):
        res += index_pdf_content(subdir, file)

  
# Remove rows with less than 40 tokens
df = pd.DataFrame(res, columns=["id", "url", "content", "tokens"])
df = df[df.tokens > args.min_tokens]
df = df.drop_duplicates(['id'])
df = df.reset_index().drop('index',axis=1) # reset index
print(df.head())

# Store the content to a CSV
df.to_csv(args.out, index=False)
print(f"Done! File saved to {args.out}")