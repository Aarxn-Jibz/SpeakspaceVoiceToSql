
# Voice-to-SQL Workflow for SpeakSpace

Convert natural language voice notes into SAFE, structured SQL queries using a custom SpeakSpace Workflow.

## Overview  

This project enables SpeakSpace users to speak a query like:

> "Show me all interns in marketing"

…and instantly generate the SQL:
  
SELECT * FROM employees
WHERE employment_type='Intern'
AND department_id=(SELECT department_id FROM departments WHERE department_name='Marketing');

The workflow receives text from SpeakSpace, processes it using an LLM, and returns JSON-formatted SQL.

---
##  Core Logic

SpeakSpace sends:  
```
{
  "prompt": "The user's spoken instruction",

  "note_id": "id",

  "timestamp": "ISO string"

}
```

Your backend returns:  
{
  "status": "success",
  "sql": "SQL query here",
  "needs_clarification": false,
  "clarification_question": ""
}

---
# Deployment Guide

## Deploy on Render

1. Go to https://render.com
2. Create new Web Service
3. Connect GitHub repo
4. Go to https://huggingface.co/
5. Generate a user access token
6. Add user access token to the Environment Variable in render and keep the key as HF_API_KEY
7. Deploy
---
## API Endpoint
https://speakspacevoicetosql.onrender.com/process-voice

## POST
{
  "prompt": "The prompt goes here",
  "note_id": "The note_id goes here"
}

---
##  SpeakSpace Configuration

**Title:** Voice to SQL  
**Description:** Converts natural language into SQL queries for employee management system.  
**Prompt Template:**
```
{
  "prompt": "The prompt goes here",
  "note_id": "The note_id goes here"
}
```

**API URL:** https://speakspacevoicetosql.onrender.com/process-voice

**Authorization:**  Your generated user token from HuggingFace