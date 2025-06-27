from flask import Flask, request, jsonify

## Slack Imports ##
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

## LLaMa 3 Imports ##
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

import os
import requests
from dotenv import load_dotenv
import sys
import json
import threading
import re

slack_token = os.environ["SLACK_BOT_TOKEN"]
signing_secret = os.environ["SLACK_SIGNING_SECRET"]
openrouter_key = os.environ["OPENROUTER_API_KEY"]

load_dotenv()

app = Flask(__name__)
conversations = {}

client = WebClient(token=slack_token)
verifier = SignatureVerifier(signing_secret=signing_secret)


## Doc Retrieval ##
# # docs = [Document(page_content=open("doc1.txt").read()),
# #         Document(page_content=open("doc2.txt").read())]

# splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# #texts = splitter.split_documents(docs)

# embeddings = OllamaEmbeddings(model="llama3")
# vectorstore = FAISS.from_documents(texts, embeddings)
# qa_chain = RetrievalQA.from_chain_type(llm=Ollama(model="llama3"), retriever=vectorstore.as_retriever())

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1", 
    openai_api_key= openrouter_key,  # from openrouter.ai/keys
    model="mistralai/mistral-7b-instruct" # or "meta-llama/llama-3-70b-instruct", etc.
)


## LLama Post Method ##
@app.route("/ask_llama", methods=["POST"])
def ask_llama():
    user_input = request.json.get("question")
    if not user_input:
        return jsonify({"error": "Missing 'question' in request"}), 400

    answer = llm.invoke(user_input)
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
    #debug_log(event)
    ##If Bot is Opened ##
    # if event.get("type") == "app_home_opened":
    #     user_id = event.get("user")
    #     publish_home_tab(user_id)

    ## If bot gets a DM ##
    if event.get("type") == "message" and event.get("channel_type") == "im":
        text = event.get("text")
        channel = event.get("channel")

        #debug_log(f"DM from {user} with Email {email}: {text}")
        #Change
        response = requests.post("https://slackbot-tbiv.onrender.com/ask_llama", json={"question": text})
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