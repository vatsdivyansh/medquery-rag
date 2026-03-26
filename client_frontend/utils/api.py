import requests
from config import API_URL

def upload_pdfs_api(files):
    files_payload = []

    for f in files:
        f.seek(0)  
        files_payload.append(
            ("files", (f.name, f, "application/pdf"))
        )

    response = requests.post(
        f"{API_URL}/upload_pdfs/",  
        files=files_payload
    )

    return response
     


def ask_question(question):
    return requests.post(f"{API_URL}/ask/",data={"question":question})

    