""" 
initial_retrieval.py
Goal: get urls from Google Sheet, save as HTML, and convert to Markdown
"""

# ------ Package installation ------
# Packages to handle pip installs
import os
import sys
import subprocess
import pkg_resources

# Import Statements for writing/reading with Google Sheets API
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from urllib.parse import urlparse, quote
import chardet
from bs4 import BeautifulSoup
import html2text
import PyPDF2
import time
import re

# Configuration
SPREADSHEET_ID = '19ROOIViqZbgc1127K6-QYFbqGNC8RXg0CxWQQBb0bh8'
RANGE_NAME = 'Sheet1!C:C'
SERVICE_ACCOUNT_FILE = os.path.join(os.path.expanduser('~'), 'VSCode', 'llm_fralitymodel_remote', 'llm_frality_model', 'frality-docs-c6f8f51d08f4.json')
OUTPUT_DIR = 'downloads'


# ------ Part 1: Check Package Requirements ------

def check_and_install_requirements():
    requirements_path = 'requirements.txt'
    with open(requirements_path, 'r') as f:
        required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = set(required_packages) - installed
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path], stdout=subprocess.DEVNULL)
        print("Installation complete.")

check_and_install_requirements()

# ------ Part 2: Functions Definitions ------

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/spreadsheets'])
service = build('sheets', 'v4', credentials=credentials)

def get_urls_from_sheet():
    try:
        result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        return [row[0] for row in result.get('values', []) if row and row[0].startswith('http')]
    except Exception as e:
        print(f"Error fetching URLs from Google Sheets: {e}")
        return []

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    path_parts = path.split('/')
    filename = path_parts[-1] if path_parts else parsed.netloc
    
    # Include part of the path in the filename
    if len(path_parts) > 1:
        filename = f"{path_parts[-2]}_{filename}"
    
    # Add timestamp to ensure uniqueness
    timestamp = int(time.time())
    
    # Remove any non-alphanumeric characters and replace spaces with underscores
    filename = re.sub(r'[^\w\-_\. ]', '', filename)
    filename = filename.replace(' ', '_')
    
    # Truncate filename if it's too long
    max_length = 100  # Adjust this value as needed
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    return f"{filename}_{timestamp}"

def html_to_markdown(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    h = html2text.HTML2Text()
    h.ignore_links = False
    return h.handle(str(soup.body) if soup.body else html_content)

def download_and_save(url, output_dir):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        safe_filename = sanitize_filename(url)
        
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            filename = os.path.join(output_dir, safe_filename)
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            source_url = record_filename_in_sheet(url, os.path.basename(filename))
            if source_url:
                update_pdf_metadata(filename, source_url)
        else:
            encoding = chardet.detect(response.content)['encoding'] or 'utf-8'
            html_content = response.content.decode(encoding)
            markdown_content = html_to_markdown(html_content)
            
            markdown_filename = os.path.join(output_dir, safe_filename + '.md')
            source_url = record_filename_in_sheet(url, os.path.basename(markdown_filename))
            if source_url:
                markdown_content = f"<!-- Source URL: {source_url} -->\n\n" + markdown_content
            
            with open(markdown_filename, 'w', encoding='utf-8', errors='replace') as f:
                f.write(markdown_content)
        
        return filename if 'filename' in locals() else markdown_filename
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def update_pdf_metadata(filename, source_url):
    try:
        with open(filename, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()
            
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            
            pdf_writer.add_metadata({'/SourceURL': source_url})
            
            with open(filename, 'wb') as output_file:
                pdf_writer.write(output_file)
    except Exception as pdf_error:
        print(f"Error updating PDF metadata: {pdf_error}")

def record_filename_in_sheet(url, filename):
    if not filename:
        print(f"No filename to record for URL: {url}")
        return None

    try:
        result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range='Sheet1!C:C').execute()
        values = result.get('values', [])
        row_index = next((i for i, row in enumerate(values) if row and row[0] == url), None)
        
        if row_index is not None:
            source_url = url
            range_name = f'Sheet1!H{row_index + 1}'
            body = {'values': [[filename]]}
            service.spreadsheets().values().update(spreadsheetId=SPREADSHEET_ID, range=range_name, valueInputOption='RAW', body=body).execute()
            return source_url
        else:
            print(f"URL not found in sheet: {url}")
            return None
    except Exception as e:
        print(f"Error recording filename in sheet: {e}")
        return None

def check_source_metadata(filename):
    try:
        if filename.endswith('.pdf'):
            with open(filename, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                if '/SourceURL' in metadata:
                    print(f"PDF file {filename} has source URL: {metadata['/SourceURL']}")
                else:
                    print(f"PDF file {filename} does not have a source URL in its metadata")
        elif filename.endswith('.md'):
            with open(filename, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()
                if first_line.startswith('<!-- Source URL:'):
                    print(f"Markdown file {filename} has source URL: {first_line}")
                else:
                    print(f"Markdown file {filename} does not have a source URL comment")
        else:
            print(f"Unsupported file type: {filename}")
    except Exception as e:
        print(f"Error checking metadata for {filename}: {e}")

def handle_existing_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

# Function to initialize the retrieval process
def initialize_retrieval():
    """
    Initializes the retrieval process by checking if the output directory exists,
    and if not, creating it. It then fetches URLs from the Google Sheet, downloads
    the files, handles existing files, checks source metadata, and prints the completion message.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    urls = get_urls_from_sheet()
    successful_downloads = 0
    failed_downloads = 0
    
    for url in urls:
        filename = download_and_save(url, OUTPUT_DIR)
        if filename:
            handle_existing_file(filename)
            new_filename = download_and_save(url, OUTPUT_DIR)
            if new_filename:
                successful_downloads += 1
            else:
                failed_downloads += 1
        else:
            failed_downloads += 1
    
    print("Download process completed.")
    print(f"Successfully downloaded: {successful_downloads} documents")
    print(f"Failed downloads: {failed_downloads} documents")
    print(f"Total processed: {len(urls)} documents")

# ------ Part 3: Main Function ------

if __name__ == "__main__":
    initialize_retrieval()
