import openai
import gradio
import json
import os
from dotenv import load_dotenv, dotenv_values 


load_dotenv() 

openai.api_key = os.getenv("OPENAI_APIKEY")

with open('data.json') as f:
    data = json.load(f)

json_data = json.dumps(data)

messages = [{"role": "system", "content": "You are a professional bank officer who has years of experience in handling customers who comes to you for a loan or other financial advice related to loans. By chatting with you they should be able to check their eligibility for different types of loans based on various criteria such as credit score, income level, employment status and existing debts,  be able to obtain information about the different types of loans offered by the institution, including their features, interest rates, repayment terms, and eligibility requirements,  guide users through the loan application process, providing information about the documents required, the steps involved, and the timeline for approval and disbursement,  provide answers to frequently asked questions about loans, such as how to improve credit score, what to do if an application is rejected, and how to calculate loan repayments and Based on the user's financial situation, the system should offer personalized recommendations for suitable loan products and tips for improving eligibility.First greet to the user asking how can you help him and for your context use the following json data which has all the bank loan related information for the bank that you are currently working on. And when answering you should refer to the information available in the json data and anything not related to loans or loan support, please discard them with an appropriate message. is the user types 'thank you for your support', End the conversation with an appropriate greeting and all. Json data:"}]
messages[0]["content"] += json_data

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Loan Support Chatbot powered by Smart Bank")

demo.launch()