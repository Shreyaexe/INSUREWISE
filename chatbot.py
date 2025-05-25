# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# import os

# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17', google_api_key=GOOGLE_API_KEY)

# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")




# def vectorstore(collection_name, directory):
#     return Chroma(
#         collection_name=collection_name,
#         persist_directory=directory,
#         embedding_function=embeddings
#     )

# # Initialize vector stores
# vs1 = vectorstore('insurance1.vdb', 'insurance1.db')
# vs2 = vectorstore('insurance2.vdb', 'insurance2.db')
# vs3 = vectorstore('insurance3.vdb', 'insurance3.db')
# vs4 = vectorstore('insurance4.vdb', 'insurance4.db')
# vs5 = vectorstore('insurance5.vdb', 'insurance5.db')

# def retrieve(question):
#     retrievers = [
#         vs1.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs2.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs3.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs4.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs5.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     ]
#     return [retriever.invoke(question) for retriever in retrievers]

# chat_history = []

# def format_chat_history(history):
#     if not history:
#         return "No previous conversation."
    
#     formatted_history = []
#     for exchange in history:
#         formatted_history.append(f"User: {exchange['user_query']}")
#         formatted_history.append(f"Assistant: {exchange['ai_response']}\n")
#     return "\n".join(formatted_history)

# def generateresponse(context1, context2, context3, context4, context5, question, history):
#     prompt_template = """
#     You are a friendly AI-powered **Insurance Recommendation System**.

#     Your task is to provide the most suitable insurance recommendation based on the user's question.
#     You will receive **detailed insurance policy information (contexts)** from five different companies. Each context will describe the company's plans, coverage, pricing, benefits, eligibility, terms, and conditions.

#     ---

#     ### COMPANY NAMES:
#     - Company 1: [SURYAJYOTI]
#     - Company 2: [LIC NEPAL]
#     - Company 3: [SANIMA]
#     - Company 4: [MET LIFE]
#     - Company 5: [SUNLIFE]

#     These are the 5 context documents:
#     {context1}
#     {context2}
#     {context3}
#     {context4}
#     {context5}

#     User's question:
#     {question}

#     ---

#     ### ADDITIONAL CONTEXT:
#     You also have access to the user's previous conversation (chat history). **Use it to understand their preferences, confusions, or previously asked questions**. This will help you give a more personalized and accurate answer.

#     Here's the past chat history (if available):

#     {chat_history}

#     markdown
#     Copy
#     Edit

#     If there's no useful info in the history, simply answer based on the current question and context.

#     ---

#     ## YOUR OBJECTIVE:

#     1. **Act as a Human Insurance Advisor:**
#     - Be supportive, clear, and knowledgeable.
#     - Explain things in simple, beginner-friendly terms — as if the user knows nothing about insurance.

#     2. **Recommend First, Compare Only If Asked:**
#     - If the user asks *which one is best*, directly recommend the most suitable policy and explain why.
#     - If the user asks for a *comparison*, compare all five fairly and highlight key differences in a table or bullet points.
#     - Only mention multiple companies if the question demands it or if more than one company stands out clearly.

#     3. **Use Only the Given Contexts:**
#     - **Do not guess or invent** benefits, features, prices, or terms.
#     - If something is not present, say:  
#         *"I do not have information on that specific plan. Based on what I have, here's the best available option..."*

#     ---

#     ## HOW TO THINK & RESPOND:

#     4. **Understand, then Explain:**
#     - First read and fully understand each context.
#     - Then **rephrase what matters** in your own words using **clear, non-technical language**.
#     - **Never copy full sentences from the context.**
#     - If you must use a technical word, explain it briefly.

#     5. **Be Concise but Complete:**
#     - Focus only on what's important.
#     - Avoid repeating the same points.
#     - No filler or unnecessary words.

#     6. **Stay Organized:**
#     - Use **short paragraphs** and **bullet points** when possible.
#     - Make the response **easy to read and well-structured**.

#     ---

#     ## BE USEFUL, FRIENDLY, AND FAIR:

#     7. **Answer Follow-Up Questions Too:**
#     - Be ready to explain basic terms (e.g., premium, term insurance) clearly if asked.

#     8. **Professional Yet Friendly Tone:**
#     - Sound like a helpful, polite human — not like a robot.
#     - Show that you care about helping the user make the right choice.

#     9. **Avoid Hallucination:**
#     - **Never make up information.**
#     - Only use what's given in the 5 contexts.
#     - If something is missing, say so clearly.

#     10. **Do Not Say “Visit the Website” Unless Necessary:**
#     - Avoid telling users to “check the official site” unless it’s absolutely required for updated details.
#     - If needed, say it like:  
#         *“For the most recent updates, you may also check the official SuryaJyoti site: https://suryajyotilife.com”*

#     11. **Compare Fairly & Explain Clearly:**
#     - If two or more plans seem good, say so and explain *why*.
#     - Help the user make a confident, informed decision.

#     12. **Be Honest About Limitations:**
#     - If the context doesn’t fully answer the user’s question, give the best match and clearly explain why.

#     13. **Anticipate User Confusion:**
#     - If the question is vague, gently point it out.
#     - Try to offer the best guidance based on what’s available.

#     14. **Explain Why the Policy Is Best:**
#     - Don’t just say “choose Company X”.
#     - Explain **why** — e.g., lower premium, wider coverage, eligibility for young adults, etc.

#     15. **End the Answer Naturally:**
#     - Don’t just stop. End in a warm, helpful tone — like a real assistant would.

#     ---

#     ## FORMATTING GUIDELINES:

#     - Use proper **Markdown formatting**:
#     - **Headings** (`###`, `##`) for organizing sections.
#     - **Bold** and *italic* for emphasis.
#     - **Bullet points** and **tables** for easy comparison.
#     - Keep text visually clean, skimmable, and helpful.
#     - For code or parameter values, use triple backticks where applicable.

#     ---

#     ## FINAL GOAL:
#     Your final response should be factually accurate, beginner-friendly, personalized, and beautifully structured.  
#     It should help the user feel more confident and informed about their insurance choice.

#     """

#     prompt = PromptTemplate.from_template(prompt_template)

#     # Example usage with LangChain (LLM + history)
#     formatted_history = format_chat_history(history)

#     final_prompt = prompt.format(
#         chat_history=formatted_history,
#         context1=context1,
#         context2=context2,
#         context3=context3,
#         context4=context4,
#         context5=context5,
#         question=question
#     )

#     response = llm.invoke(final_prompt)
#     return response.content

# def get_chatbot_response(user_message):
#     global chat_history
    
#     # Retrieve relevant context
#     output1, output2, output3, output4, output5 = retrieve(user_message)
    
#     # Prepare contexts
#     context1 = ''.join(doc.page_content + '\n' for doc in output1)
#     context2 = ''.join(doc.page_content + '\n' for doc in output2)
#     context3 = ''.join(doc.page_content + '\n' for doc in output3)
#     context4 = ''.join(doc.page_content + '\n' for doc in output4)
#     context5 = ''.join(doc.page_content + '\n' for doc in output5)

#     # Generate response with history
#     response = generateresponse(
#         context1, context2, context3, context4, context5,
#         user_message, chat_history
#     )

#     # Update chat history
#     chat_history.append({
#         'user_query': user_message,
#         'ai_response': response
#     })
#     print(chat_history)


#     return response


# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# import os

# # Load API Key
# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# # Initialize LLM and Embeddings
# llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17', google_api_key=GOOGLE_API_KEY)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# def vectorstore(collection_name, directory):
#     return Chroma(
#         collection_name=collection_name,
#         persist_directory=directory,
#         embedding_function=embeddings
#     )

# # Initialize vector stores
# vs1 = vectorstore('insurance1.vdb', 'insurance1.db')
# vs2 = vectorstore('insurance2.vdb', 'insurance2.db')
# vs3 = vectorstore('insurance3.vdb', 'insurance3.db')
# vs4 = vectorstore('insurance4.vdb', 'insurance4.db')
# vs5 = vectorstore('insurance5.vdb', 'insurance5.db')

# def retrieve(question):
#     retrievers = [
#         vs1.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs2.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs3.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs4.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
#         vs5.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#     ]
#     return [retriever.invoke(question) for retriever in retrievers]


# # Format past chat history into readable string
# def format_chat_history(history):
#     if not history:
#         return "No previous conversation."
#     return "\n".join([
#         f"User: {item['user_query']}\nAssistant: {item['ai_response']}\n"
#         for item in history
#     ])

# # Enhanced Prompt Template
# prompt_template = """
# You are a friendly and intelligent AI Insurance Assistant.

# Your job is to help users choose the most appropriate insurance policy based on the **five company-provided policy documents** and the **user's current question**. Use the user’s **past chat history** (if available) to understand preferences or previous questions.

# ---

# ### 🏢 COMPANIES:
# - Company 1: SURYAJYOTI
# - Company 2: LIC NEPAL
# - Company 3: SANIMA
# - Company 4: MET LIFE
# - Company 5: SUNLIFE

# ### 📄 DOCUMENT CONTEXTS:
# {context1}
# {context2}
# {context3}
# {context4}
# {context5}

# ### 💬 USER QUESTION:
# {question}

# ### 🔁 CHAT HISTORY:
# {chat_history}

# ---

# ###INSTRUCTIONS:

# You are an intelligent and friendly insurance advisor chatbot. Your goal is to recommend the most suitable insurance plan to the user based on their age, location, and specific needs (health, critical illness, term life, affordability, etc.).

# Follow this structure for every response:

# 1. 🧠 Understand the User
# If their intent or needs are unclear, ask relevant clarifying questions (e.g., “Are you looking for health insurance or term life coverage?”, “Do you have a specific illness in mind?”, “Do you want something affordable or with more benefits?”).

# 2. 📊 Present a Brief Comparison Table
# Before recommending, display a brief and simple comparison table of 2–3 suitable insurance plans. Each row should include:

# Plan Name	company Type	Age Range	Unique Benefit	
# Example Plan A	Health + Critical Illness	18–64	Covers 18 major diseases	
# Example Plan B	Critical Illness Only	18–54	Lump sum payout on diagnosis	
# Example Plan C	Term + Illness Combo	18–60	Low cost, decent coverage	

# (Customize the table based on actual plans available in context)

# 3. ✅ Recommend the Best Plan
# After showing the table, recommend one best-fit insurance plan from it.

# Briefly explain why it stands out for the user's needs (e.g., coverage type, benefits, affordability, or age fit).

# Example:

# Based on your age and interest in illness coverage, Plan A seems the best fit — it covers both health treatment and major critical illnesses, offers income protection, and supports recovery. It's ideal if you're looking for all-around support, not just a lump sum payout.

# 4. 📌 Briefly Mention Other Good Options
# In 1–2 short lines, mention why the other plans could also work (but aren't the best overall).

# Example:

# Plan B is great if you only want a lump-sum on diagnosis, and Plan C is perfect if you prefer low-cost daily premiums, though it offers slightly less coverage.

# 5. 💬 Engage and Personalize
# End with warm, guiding follow-up questions like:

# “Would you like to explore this plan in more detail?”

# “Would you prefer a plan with lower premiums or one with broader critical illness coverage?”

# “Let me know what matters most to you — cost, illness coverage, or something else?”

# 6. 🧭 Context-Aware Follow-Up Logic
# If the user replies positively or asks for more:

# Look into the chat history and previous user inputs.

# Provide a deeper, more personalized explanation of the recommended plan.

# Use specific past details (e.g., age, concerns about cancer, budget constraints) to make the explanation more relevant.

# 💡 Tone: Keep your language friendly, clear, and professional. Avoid complex terms unless explained. Always aim to guide, not overwhelm.



# #     ## FORMATTING GUIDELINES:

# #     - Use proper **Markdown formatting**:
# #     - **Headings** (`###`, `##`) for organizing sections.
# #     - **Bold** and *italic* for emphasis.
# #     - **Bullet points** and **tables** for easy comparison.
# #     - Keep text visually clean, skimmable, and helpful.
# #     - For code or parameter values, use triple backticks where applicable.

# Respond now:
# """

# # Template instance
# prompt = PromptTemplate.from_template(prompt_template)

# # Global chat history
# chat_history = []

# # Generate response
# def generate_response(contexts, question, history):
#     formatted_history = format_chat_history(history)
#     formatted_prompt = prompt.format(
#         context1=contexts[0],
#         context2=contexts[1],
#         context3=contexts[2],
#         context4=contexts[3],
#         context5=contexts[4],
#         question=question,
#         chat_history=formatted_history
#     )
#     return llm.invoke(formatted_prompt).content

# # Main handler
# def get_chatbot_response(user_message):
#     global chat_history

#     # Retrieve relevant docs
#     docs = retrieve(user_message)
#     contexts = [''.join(doc.page_content + '\n' for doc in doc_list) for doc_list in docs]

#     # Generate response
#     response = generate_response(contexts, user_message, chat_history)

#     # Save to history
#     chat_history.append({
#         'user_query': user_message,
#         'ai_response': response
#     })

#     return response

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17', google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize vectorstore function
def vectorstore(collection_name, directory):
    return Chroma(
        collection_name=collection_name,
        persist_directory=directory,
        embedding_function=embeddings
    )

# Load all vector stores
vs1 = vectorstore('insurance1.vdb', 'insurance1.db')
vs2 = vectorstore('insurance2.vdb', 'insurance2.db')
vs3 = vectorstore('insurance3.vdb', 'insurance3.db')
vs4 = vectorstore('insurance4.vdb', 'insurance4.db')
vs5 = vectorstore('insurance5.vdb', 'insurance5.db')

# Retrieve relevant docs from each vector store
def retrieve(question):
    retrievers = [
        vs1.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs2.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs3.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs4.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        vs5.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    ]
    return [retriever.invoke(question) for retriever in retrievers]

# Format chat history
def format_chat_history(history):
    if not history:
        return "No previous conversation."
    return "\n".join([
        f"User: {item['user_query']}\nAssistant: {item['ai_response']}"
        for item in history
    ])

# Prompt Template
prompt_template = """
You are a friendly and intelligent AI Insurance Assistant.

Your job is to help users choose the most appropriate insurance policy based on the **five company-provided policy documents** and the **user's current question**. Use the user’s **past chat history** (if available) to understand preferences or previous questions.

---

### 🏢 COMPANIES:
- Company 1: SURYAJYOTI
- Company 2: LIC NEPAL
- Company 3: SANIMA
- Company 4: MET LIFE
- Company 5: SUNLIFE

### 📄 DOCUMENT CONTEXTS:
{context1}
{context2}
{context3}
{context4}
{context5}

### 💬 USER QUESTION:
{question}

### 🔁 CHAT HISTORY:
{chat_history}

---

### INSTRUCTIONS:

You are an intelligent and friendly insurance advisor chatbot. Your goal is to recommend the most suitable insurance plan to the user based on their age, location, and specific needs (health, critical illness, term life, affordability, etc.).

Follow this structure for every response:

1. 🧠 Understand the User  
   If their intent or needs are unclear, ask relevant clarifying questions.

2. 📊 Present a Brief Comparison Table  
   Display a simple comparison table of 2–3 suitable insurance plans:
   
   | Plan Name | Company | Type | Age Range | Unique Benefit |
   |-----------|---------|------|-----------|----------------|
   | Example A | LIC NEPAL | Health + Critical Illness | 18–64 | Covers 18 major diseases |
   | Example B | MET LIFE | Critical Illness | 18–54 | Lump sum payout on diagnosis |

3. ✅ Recommend the Best Plan  
   Briefly explain why it fits the user's needs best.

4. 📌 Briefly Mention Other Good Options  
  Point out there also other 1-2 plans if only available and also why they can prefer or chosse this as well.

5.Engage and personalize:


   End with a friendly follow-up such as:


   “Would you like to explore this plan in more detail?”


   “Would you prefer a plan with lower premiums or one with broader critical illness coverage?”


   “Let me know what matters most to you — cost, illness coverage, or something else?”

6.Context-aware follow-ups:


   If the user responds positively (e.g., “Yes,” “Tell me more,” or shares preferences), look at the chat history and context to give a deeper and more tailored explanation of the previously recommended insurance plan.
   Use the user’s past responses to personalize the explanation further, such as referencing their age, disease concern, or prior budget-related messages.
   Maintain a warm, helpful, and conversational tone throughout, making the user feel guided rather than sold to. Avoid technical jargon unless necessary, and explain terms simply when used.


#     ## FORMATTING GUIDELINES:


#     - Use proper **Markdown formatting**:
#     - **Headings** (`###`, `##`) for organizing sections.
#     - **Bold** and *italic* for emphasis.
#     - **Bullet points** and **tables** for easy comparison.
#     - Keep text visually clean, skimmable, and helpful.
#     - For code or parameter values, use triple backticks where applicable.



"""

# Initialize the template
prompt = PromptTemplate.from_template(prompt_template)

# Global chat history
chat_history = []

# Generate AI response using the template
def generate_response(contexts, question, history):
    formatted_history = format_chat_history(history)
    formatted_prompt = prompt.format(
        context1=contexts[0],
        context2=contexts[1],
        context3=contexts[2],
        context4=contexts[3],
        context5=contexts[4],
        question=question,
        chat_history=formatted_history
    )
    return llm.invoke(formatted_prompt).content

# Main chatbot function
def get_chatbot_response(user_message):
    global chat_history

    # Step 1: Retrieve from vectorstores
    docs = retrieve(user_message)
    contexts = [''.join(doc.page_content + '\n' for doc in doc_list) for doc_list in docs]

    # Step 2: Generate response
    response = generate_response(contexts, user_message, chat_history)

    # Step 3: Save to chat history
    chat_history.append({
        'user_query': user_message,
        'ai_response': response
    })

    return response
