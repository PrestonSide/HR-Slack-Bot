from flask import Flask, request, jsonify

## Slack Imports ##
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

## LLaMa 3 Imports ##
#from langchain.chains import RetrievalQA
#from langchain_community.vectorstores import FAISS
#from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
#import pdfplumber


import os
import requests
from dotenv import load_dotenv
import sys
import json
import threading
import re

load_dotenv()

slack_token = os.environ["SLACK_BOT_TOKEN"]
signing_secret = os.environ["SLACK_SIGNING_SECRET"]
openrouter_key = os.environ["OPENROUTER_API_KEY"]



app = Flask(__name__)
conversations = {}

client = WebClient(token=slack_token)
verifier = SignatureVerifier(signing_secret=signing_secret)

llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_key,
            model="mistralai/mistral-7b-instruct"
        )
## Doc Retrieval ##



# qa_chain = None

# def get_qa_chain():
#     global qa_chain
#     if qa_chain is None:
#         debug_log("⏳Building vectorstore")
        

#         loader1 = PyPDFLoader("handbook.pdf")
#         loader2 = PyPDFLoader("retirement.pdf")

#         docs = loader1.load() + loader2.load()
#         splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = splitter.split_documents(docs)

#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(texts, embeddings)

#         llm = ChatOpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             openai_api_key=openrouter_key,
#             model="mistralai/mistral-7b-instruct"
#         )

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=vectorstore.as_retriever(),
#             return_source_documents=False
#         )
#     debug_log("Done Building")
#     return qa_chain

# def extract_text_from_pdf(path):
#     try:
#         text = ""
#         with pdfplumber.open(path) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         print(f"Loaded In {path}")
#         return text
#     except Exception as e:
#         print(f"Error reading PDF {path}: {e}")
#         return ""
    


## LLama Post Method ##
@app.route("/ask_llama", methods=["POST"])
def ask_llama():
    global llm
    # pdf1 = extract_text_from_pdf("handbook.pdf")
    #pdf2 = extract_text_from_pdf("retirement.pdf")
    #pdf_text = pdf2
    user_input = request.json.get("question")
    debug_log(f"Ask AI: {user_input}")
    if not user_input:
        return jsonify({"error": "Missing 'question' in request"}), 400

    #ai_input = f"Using the following info: {pdf_text}. Answer this prompt: {user_input}"
    #debug_log(ai_input)
    answer = llm.invoke(user_input)
    debug_log(f"ai answer: {answer}")
    #debug_log(f"AI Answer: {answer}")
    return jsonify({"answer": str(answer.content if hasattr(answer, "content") else answer)})

## Post method responses ##
@app.route("/slack/events", methods=["POST"])
def slack_events():
    #debug_log("Starting Event")
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return "Invalid request", 403
    
    if "payload" in request.form:
        payload = json.loads(request.form["payload"])
    else:
        payload = request.json

    #debug_log(f"Incoming payload: {payload}")
    if payload.get("type") == "url_verification":
        return jsonify({"challenge": payload["challenge"]}), 200

    if payload.get("type") == "block_actions":
        user_id = payload["user"]["id"]
        action_id = payload["actions"][0]["action_id"]
        #debug_log(f"action_id: {action_id}")
        simulated_event = {
            "type": "message",
            "user": user_id,
            "text": "confirm" if action_id == "confirm_action" else "cancel",
            "channel": payload["channel"]["id"],
            "channel_type": "im"
        }
        threading.Thread(target=handle_event, args=(simulated_event,)).start()
        return "", 200
    
    if "event" in payload:
        event = payload["event"]
        threading.Thread(target=handle_event, args=(event,)).start()
        return "", 200

    return "", 200

def handle_event(event):
    global conversations
    ## Makes bot not respond to itself ##
    if "bot_id" in event:
        #debug_log(f"BOT EVENT: {event}")
        return "", 200
        
    #debug_log(f"Slack Event: {event}")
    ##If Bot is Opened ##
    # if event.get("type") == "app_home_opened":
    #     user_id = event.get("user")
    #     publish_home_tab(user_id)

    ## If bot gets a DM ##
    if event.get("type") == "message" and event.get("channel_type") == "im":
        text = event.get("text")
        channel = event.get("channel")
        user = event.get("user")

        #debug_log(f"DM: {text}")
        #Change
        response = requests.post("https://hr-slack-bot.onrender.com/ask_llama", json={"question": text})
        directline = get_directline_token()
        directline_token = directline["token"]
        conversation_id = directline["conversationId"]
        debug_log(f"Direct Line Token: {directline_token}")
        debug_log(f"Conversation Id: {conversation_id}")
        start_copilot_conversation(directline_token)
        debug_log(send_copilot_message(text, conversation_id, directline_token, user))
        debug_log(f"Copilot Message: {get_copilot_response(conversation_id, directline_token)}")
        llama_answer = response.json().get("answer", "Sorry, I couldn't find an answer.")
        
        send_slack_message(channel, llama_answer)
                
            
    return "", 200


## Debug Print for Render ##
def debug_log(message):
    print(message, file=sys.stdout, flush=True)



## Function for slack bot to send message to user ##
def send_slack_message(channel, text):
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json"
    }
    data = {
        "channel": channel,
        "text": text
    }
    requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=data)

def get_directline_token():
    token_url = "https://defaultceaebaaed1ed4b91aa8383683a9425.13.environment.api.powerplatform.com/powervirtualagents/botsbyschema/cr552_hrAssistantSlack/directline/token?api-version=2022-03-01-preview"
    
    response = requests.get(token_url)
    return response.json()

def start_copilot_conversation(token):
    url = "https://directline.botframework.com/v3/directline/conversations"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    #response = requests.post(url, headers=headers)
    #return response.json()
    requests.post(url, headers=headers)

def send_copilot_message(text, conversation_id, token, user):
    url = f"https://directline.botframework.com/v3/directline/conversations"
    headers = {
        "Authorization": f"Bearer 2QghxPX1vqvEGcQMHjy9H0iQUWlHxPwa3tCtu28nXGYmfYt98MWUJQQJ99BFACYeBjFAArohAAABAZBS2Bhw.B9xEX81xDLMgHUBYMyAuEAhMnQwgAF1Dj4UAUCJRxpGpDJaMqOBzJQQJ99BFACYeBjFAArohAAABAZBSqq6Z",
        "Content-Type": "application/json"
    }
    data = {
        "locale": "en-US",
        "type": "message",
        "from": {
            "id": user
        },
        "text": text
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def get_copilot_response(conversation_id, token):
    url = f"https://directline.botframework.com/v3/directline/conversations/{conversation_id}/activities"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    return response.json()





# ## Home Menu Display ##
# def publish_home_tab(user_id):
#     view = {
#         "type": "home",
#         "blocks": [
#             {
#                 "type": "header",
#                 "text": {
#                     "type": "plain_text",
#                     "text": "👋 Welcome to Help Desk Bot!",
#                     "emoji": True
#                 }
#             },
#             { "type": "divider" },
#             {
#                 "type": "section",
#                 "text": {
#                     "type": "mrkdwn",
#                     "text": (
#                         "Here's what I can help you with:\n\n"
#                         "`1` – Reset your password 🔒 \n"
#                         "`2` – Update your phone number 📱 \n"
#                         "`3` – View your user info 🧑‍💼 \n"
#                         "`4` – Add a user to a group 🏢 \n"
#                         "`5` – Email the help desk 📩 \n\n"

                        
#                         "_Just type the number above in messages to start your request._"
#                     )
#                 }
#             },
#             { "type": "divider" },
#             {
#                 "type": "section",
#                 "text": {
#                     "type": "mrkdwn",
#                     "text": (
#                         "*Need assistance with an IT issue?*\n\n"
#                         "I can help answer many frequently asked questions.\n\n"
#                         "_If you need further support, simply type `5` to contact the IT department._"
#                     )
#                 }
#             },
#             { "type": "divider" },
#             {
#                 "type": "context",
#                 "elements": [
#                     {
#                         "type": "mrkdwn",
#                         "text": "⚙️ If you need more help, contact IT or type `help`."
#                     }
#                 ]
#             }
#         ]
#     }

#     client.views_publish(user_id=user_id, view=view)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)