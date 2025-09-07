## Library Assistant Usage:
## Users can ask about:
## 1. Book availability (only for registered members)
##    - Example: "Can I get 1 copy of Quran?"
## 2. Search for a book
##    - Example: "Search for Python Crash Course"
## 3. Library timings
##    - Example: "What are the library hours?"
## 4. List all books (only for registered members)
##    - Example: "Books available"
## The assistant will also inform how many copies of a book are available.
## Non-library questions will get the response:
##    "I only answer library-related questions."



import re
from typing import Dict
import os
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    function_tool,
    InputGuardrail,
    AsyncOpenAI,
    ModelSettings,
    OpenAIChatCompletionsModel,
    GuardrailFunctionOutput,
    RunContextWrapper
    
    
    
)
from pydantic import BaseModel

def dynamic_instruction(func):
    func._is_dynamic_instruction = True
    return func

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["OPENAI_API_KEY"] = ""

class UserContext(BaseModel):
    name: str
    member_id: str = ""

@dynamic_instruction
def personalized_instruction(context:UserContext) -> str:
    return f"Always greet {context.name} by name in your responses."

BOOK_DB: Dict[str, int] = {
    "Quran": 1,
    "atomic habits": 1,
    "to kill a mockingbird": 1,
    "python crash course": 1,
    "introduction to algorithms": 1
}

@function_tool
def search_book(title: str) -> str:
    title_lower = title.lower()
    if title_lower in BOOK_DB:
        return f"'{title}' is in our library! Copies Available:{BOOK_DB[title_lower]}"
    return f"Sorry,'{title}' is not in our library"
@function_tool
def check_availability(title: str, member_id: str) -> str:
    if not member_id:
        return "Only registered members can check availability"

    copies_found = re.findall(r'\d+', title)
    requested_copies = int(copies_found[0]) if copies_found else 1

    title_clean = re.sub(r'\d+', '', title.lower()).strip()

   
    matched_books = [book for book in BOOK_DB if book.lower() == title_clean]
    if not matched_books:
        return f"Sorry,'{title}' is not available in our library"

    book_name = matched_books[0]
    available = BOOK_DB[book_name]
       
    if requested_copies <= available:
        return f"Sorry,only {available} copy{'ies' if available > 1 else ''} of '{book_name}' are available."                                              
    else:
        return f"Your request is approved! You can get '{book_name}'({requested_copies} copies)."

        
@function_tool
def library_timings() -> str:
    return "Library is open from 8 AM to 8 PM, Monday to Saturday"

def list_books_logic(member_id: str) -> str:
    if not member_id:
        return "Only registered members can check availability"
    return "Books available in our library:\n" + \
           "\n".join([f"{title}: {count} copies" for title, count in BOOK_DB.items()])
@function_tool
def list_books() -> str:
    
    return "Book available in our library:\n" + \
           "\n".join([f"{title}: {count} copies" for title, count in BOOK_DB.items()])

def library_guardrail(context: dict, agent, user_input: str) -> GuardrailFunctionOutput:
    allowed_keywords = [
        "search", "find", "borrow", "return",
        "available", "availability", "timings",
        "hours", "open", "library", "books", "get","want","copies","copy"
    ]
   
    q_clean = re.sub(r'\d+', '', user_input.lower()).strip()
    matched_books = [book for book in BOOK_DB if book.lower() in q_clean]
    if any(k in q_clean for k in allowed_keywords) or matched_books:
        return GuardrailFunctionOutput(
            output_info=user_input,
            tripwire_triggered=False
        )
        
    return GuardrailFunctionOutput(
        output_info="I only answer library-related questions.",
        tripwire_triggered=True
    )

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
model_settings = ModelSettings()
model_settings.model ="gemini-2.0-flash"
model_settings.api_key =GEMINI_API_KEY
model_settings.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_settings.temperature =0.0
if __name__=="__main__":
    print("Library Assistant starting...\n")
    name = input("Enter your name:").strip()
    member_id = input("Enter your member ID (leave empty if not registered): ").strip()

    if not member_id:
        print("You are not registered. Some features like checking availability will be restricted")
        member_id = ""

    user = UserContext(name=name, member_id=member_id)
     
    instructions = personalized_instruction(user) + """
             - When user asks for 'timings' or 'hours', respond with:
             -'Library is open from 8 AM to 8 PM, Monday to Saturday.'
             - When user asks 'books available', list all books and copies.
             - If the book is not in the library, response with:'Sorry, the book is not available.'
             - Only answer library-related questions.
             - If the user says "can I get X copies of [book]", use the check_availability tool.
             - If the user says can i get or i want this many number of copies of a book if the book is available in the library and the number of copies are present too 'say yes sure you can your request has been approved'"""
    
    libraryassistant_agent = Agent[UserContext](
        name=f"{user.name}'s Library Assistant",
        model=model,
        instructions=instructions,
        model_settings=model_settings,
        tools=[search_book,check_availability, library_timings, list_books],
        input_guardrails=[InputGuardrail(library_guardrail)]
    )
    wrapped_agent = RunContextWrapper(
       agent = libraryassistant_agent,
       context ={"user":user}
    )
    runner = Runner()
    print("\nYou can ask the library assistant about books, availability, and timings. ")
while True:
    query = input("Your Query: ").strip()
    if query.lower() in["exit", "quit"]:
        print("Goodbye! Have a nice day")
        break
    if "books available" in query.lower():
        response_text = list_books_logic(user.member_id)
        print(f"Assistant: {response_text}\n")
    else:
        try:
            response = runner.run_sync(wrapped_agent, query)
            print(f"Assistant:{response.final_output}\n")
        except Exception as e:
            print("Error:", e)
