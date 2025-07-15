import os
import io
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from docx import Document
from config import Config

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveClient:
    def __init__(self):
        self.service = self._authenticate()
        self.folder_id = Config.GOOGLE_DRIVE_FOLDER_ID

    def _authenticate(self):
        """Authenticate and return Google Drive service"""
        creds = None
        
        # Check if token.json exists
        if os.path.exists(Config.GOOGLE_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(Config.GOOGLE_TOKEN_PATH, SCOPES)
        
        # If there are no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    Config.GOOGLE_CREDENTIALS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open(Config.GOOGLE_TOKEN_PATH, 'w') as token:
                token.write(creds.to_json())
        
        return build('drive', 'v3', credentials=creds)

    def list_files_in_folder(self, folder_id=None):
        """List all files in the specified folder"""
        if folder_id is None:
            folder_id = self.folder_id
        
        query = f"'{folder_id}' in parents and (mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
        
        results = self.service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, parents)"
        ).execute()
        
        return results.get('files', [])

    def get_folder_structure(self, folder_id=None):
        """Get the folder structure to identify modules"""
        if folder_id is None:
            folder_id = self.folder_id
        
        folders = {}
        
        # Get all folders
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
        results = self.service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        for folder in results.get('files', []):
            folders[folder['id']] = folder['name']
        
        return folders

    def download_document(self, file_id, file_name, mime_type):
        """Download a document from Google Drive"""
        try:
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Docs as Word document
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
            else:
                # Download Word document directly
                request = self.service.files().get_media(fileId=file_id)
            
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Save file temporarily
            temp_file_path = f"temp_{file_name}.docx"
            with open(temp_file_path, 'wb') as f:
                f.write(file_content.getvalue())
            
            # Extract text from Word document
            doc = Document(temp_file_path)
            text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            return text_content
        
        except Exception as e:
            print(f"Error downloading {file_name}: {str(e)}")
            return None

    def get_all_documents(self):
        """Get all documents with their module information"""
        documents = []
        
        # Get folder structure to identify modules
        folders = self.get_folder_structure()
        
        # Get documents from main folder
        main_files = self.list_files_in_folder()
        
        for file_info in main_files:
            content = self.download_document(
                file_info['id'], 
                file_info['name'], 
                file_info['mimeType']
            )
            
            if content:
                documents.append({
                    'id': file_info['id'],
                    'name': file_info['name'],
                    'content': content,
                    'module': 'Main',
                    'path': file_info['name']
                })
        
        # Get documents from subfolders (modules)
        for folder_id, folder_name in folders.items():
            module_files = self.list_files_in_folder(folder_id)
            
            for file_info in module_files:
                content = self.download_document(
                    file_info['id'], 
                    file_info['name'], 
                    file_info['mimeType']
                )
                
                if content:
                    documents.append({
                        'id': file_info['id'],
                        'name': file_info['name'],
                        'content': content,
                        'module': folder_name,
                        'path': f"{folder_name}/{file_info['name']}"
                    })
        
        return documents
