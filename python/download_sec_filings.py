import csv
import os
import requests
import re
import time

import json
import markdownify

def sanitize_filename(name):
    # Replace all non-alphanumeric characters (including spaces) with underscores
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Collapse multiple consecutive underscores into a single one
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def extract_metadata(content):
    metadata = {
        "cik": None,
        "company_name": None,
        "filing_date": "12/23/2025",
        "period_end_date": None,
        "report_type": "N-CSR",
        "sector": None
    }

    # CIK - Try XBRL tag first, then legacy
    # <dei:EntityCentralIndexKey ...>0001643174</dei:EntityCentralIndexKey>
    cik_match = re.search(r'<dei:EntityCentralIndexKey[^>]*>(\d+)</dei:EntityCentralIndexKey>', content, re.IGNORECASE)
    if not cik_match:
        cik_match = re.search(r'CENTRAL INDEX KEY:\s*(\d+)', content, re.IGNORECASE)
    if not cik_match:
        cik_match = re.search(r'<CIK>(\d+)', content, re.IGNORECASE)
    if cik_match:
        metadata['cik'] = cik_match.group(1).strip()

    # Company Name - Try XBRL tag first, then legacy
    # <dei:EntityRegistrantName ...>Horizon Funds</dei:EntityRegistrantName>
    name_match = re.search(r'<dei:EntityRegistrantName[^>]*>(.*?)</dei:EntityRegistrantName>', content, re.IGNORECASE | re.DOTALL)
    if not name_match:
        name_match = re.search(r'COMPANY CONFORMED NAME:\s*(.+)', content, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r'Exact name of registrant as specified in charter:.*?<font[^>]*>(.*?)</font>', content, re.IGNORECASE | re.DOTALL)
    if name_match:
        metadata['company_name'] = name_match.group(1).strip()
    
    # Sector - Try oef:IndustrySectorAxis or similar
    # This might be tricky as it is often an Axis. We'll look for content if it exists as a tag or context content.
    # Searching for generic Sector tag if specific axis not found as content
    sector_match = re.search(r'<oef:IndustrySectorAxis[^>]*>(.*?)</oef:IndustrySectorAxis>', content, re.IGNORECASE)
    if not sector_match:
         # Sometimes it's a member value, try matching general sector text if plausible
         # or try finding a context member for the axis? Without a full xml parser, we guess.
         sector_match = re.search(r'Industry Sector \[Axis\]', content, re.IGNORECASE) # Plain text label?
    if sector_match:
        # If we matched tag content
        if sector_match.groups():
            metadata['sector'] = sector_match.group(1).strip()
        else:
             # Placeholder if we just found the axis label
             metadata['sector'] = None

    # Filing Date
    date_match = re.search(r'FILED AS OF DATE:\s*(\d+)', content, re.IGNORECASE)
    if not date_match:
         date_match = re.search(r'<FILING-DATE>(\d+)', content, re.IGNORECASE)
    if date_match:
        raw_date = date_match.group(1).strip()
        if len(raw_date) == 8:
            metadata['filing_date'] = f"{raw_date[4:6]}/{raw_date[6:8]}/{raw_date[0:4]}"
        else:
            metadata['filing_date'] = raw_date

    # Period End Date - Try XBRL first
    # <xbrli:period><xbrli:endDate>2025-09-30</xbrli:endDate></xbrli:period>
    # We just look for the first endDate tag usually
    period_match = re.search(r'<xbrli:endDate[^>]*>([\d-]+)</xbrli:endDate>', content, re.IGNORECASE)
    if not period_match:
        period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d+)', content, re.IGNORECASE)
    if not period_match:
        period_match = re.search(r'<PERIOD-OF-REPORT>(\d+)', content, re.IGNORECASE)
    if not period_match:
        period_match = re.search(r'Date of reporting period:.*?<font[^>]*>(.*?)</font>', content, re.IGNORECASE | re.DOTALL)
    
    if period_match:
        raw_period = period_match.group(1).strip()
        # Check if YYYY-MM-DD (XBRL)
        if re.match(r'\d{4}-\d{2}-\d{2}', raw_period):
            # Convert to MM/DD/YYYY for consistency with other formats if desired, or keep ISO. user prompt used slashed in default.
            y, m, d = raw_period.split('-')
            metadata['period_end_date'] = f"{m}/{d}/{y}"
        elif len(raw_period) == 8 and raw_period.isdigit():
             metadata['period_end_date'] = f"{raw_period[4:6]}/{raw_period[6:8]}/{raw_period[0:4]}"
        else:
             metadata['period_end_date'] = raw_period

    return metadata

def create_metadata_file(file_path):
    """
    Creates a corresponding .metadata.json file for the given file path.
    The metadata file contains extracted attributes.
    """
    try:
        filename = os.path.basename(file_path)
        metadata_filename = f"{filename}.metadata.json"
        metadata_path = os.path.join(os.path.dirname(file_path), metadata_filename)
        
        # Read content for extraction
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        extracted = extract_metadata(content)

        # Amazon Bedrock Knowledge Base expectation for metadata
        attributes = {
            "cik": extracted['cik'],
            "company_name": extracted['company_name'],
            "filing_date": extracted['filing_date'],
            "period_end_date": extracted['period_end_date'],
            "report_type": "N-CSR",
            "sector": extracted['sector']
        }
        
        # Remove empty values
        attributes = {k: v for k, v in attributes.items() if v}
        
        metadata_content = {
            "metadataAttributes": attributes
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_content, f, indent=2)
            
        print(f"    Created metadata: {metadata_filename}")
    except Exception as e:
        print(f"    Error creating metadata for {filename}: {e}")

def download_filings():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'sec.csv')
    output_dir = os.path.join(base_dir, 'data', 'filings')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Headers for SEC requests (required to avoid 403 Forbidden)
    # SEC requires a User-Agent in the format: Sample Company Name AdminContact@sample.com
    headers = {
        'User-Agent': 'MyCustomAgent/1.0 (test@example.com)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }



    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            
            for row_num, row in enumerate(reader, 1):
                if not row or len(row) < 2:
                    continue
                
                url = row[0].strip()
                name_col = row[1].strip()
                
                # specific clean up for the second column to make it a nice filename
                filename_base = sanitize_filename(name_col)
                if not filename_base.lower().endswith('.htm') and not filename_base.lower().endswith('.html'):
                    filename = f"{filename_base}.htm"
                else:
                    filename = filename_base
                

                file_path = os.path.join(output_dir, filename)
                
                print(f"[{row_num}] Downloading {url} -> {filename}...")
                
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    print(f"    Saved to {file_path}")
                    
                    # Be polite to SEC servers, limit request rate
                    # Rate limit is 10 requests per second, so a small sleep is safe
                    time.sleep(0.15) 
                    
                except requests.exceptions.RequestException as e:
                    print(f"    Error downloading {url}: {e}")
                except Exception as e:
                    print(f"    Error saving {filename}: {e}")
                
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def generate_metadata_files():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'filings')
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return

    print(f"\nGenerating metadata for files in {output_dir}...")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        
        # Skip directories and existing metadata files
        if os.path.isdir(file_path) or filename.endswith('.metadata.json'):
            continue
            
        create_metadata_file(file_path)

def render_and_save():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'filings')
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return

    print(f"\nConverting HTML to Markdown in {output_dir}...")
    for filename in os.listdir(output_dir):
        if not (filename.lower().endswith('.html') or filename.lower().endswith('.htm')):
            continue
            
        file_path = os.path.join(output_dir, filename)
        md_filename = os.path.splitext(filename)[0] + ".md"
        md_path = os.path.join(output_dir, md_filename)
        
        try:
             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
             
             # Convert to Markdown
             md_content = markdownify.markdownify(html_content, heading_style="ATX")
             
             with open(md_path, 'w', encoding='utf-8') as f:
                 f.write(md_content)
                 
             print(f"    Converted: {filename} -> {md_filename}")
        except Exception as e:
            print(f"    Error converting {filename}: {e}")
        

if __name__ == "__main__":
    # download_filings()
    # generate_metadata_files()
    render_and_save()
