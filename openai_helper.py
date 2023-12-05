from openai import OpenAI
from tenacity import retry as tenacity_retry
from tenacity import stop_after_attempt, wait_exponential
import dotenv
import json
import os

dotenv.load_dotenv()

openai_client = OpenAI()

def retry_in_production(*args, **kwargs):
    if os.environ.get('ENABLE_RETRIES') == 'True':
        return tenacity_retry(*args, **kwargs)
    else:
        def decorator(f):
            return f
        return decorator

def get_openai_model_fallback(model):
    fallback_models = {
        'gpt-4-1106-preview':'gpt-4',
        'gpt-4':'gpt-3.5-turbo-16k-0613',
        'gpt-3.5-turbo-16k-0613': 'gpt-3.5-turbo'
        }
    new_model = fallback_models.get(model)
    
    return new_model

def call_openai_chat_with_model_fallbacks(**kwargs):
    model = kwargs.get('model', 'gpt-3.5-turbo')
    # Try to run the function 3 times. After that, update the model kwarg to a fall back and try again. Repeat until success or until we run out of fallbacks.
    for i in range(10):
        for i in range(3):
            try:
                response = openai_client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                print(f"error calling openai chat with model {model} for the {i}th time. Error is {e}")
                continue
        prev_model=model
        model = get_openai_model_fallback(model)
        print(f"Falling back from {prev_model} to {model} after 3 failed attempts.")
        if model is None:
            raise Exception('Ran out of fallback models.')
        kwargs['model'] = model
        continue

@retry_in_production(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_chat_generation(prompt=None,
                        system_message=None,
                        existing_messages=[],
                        model="gpt-3.5-turbo",
                        functions=None,
                        function_call=None,
                        temperature=0.9,
                        max_tokens=1000,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        json_output=False):
    
    kwargs = {k:v for k,v in locals().items() if v is not None and k not in ['prompt','messages','existing_messages','json_output','system_message']}
    messages = []
    if len(existing_messages) > 0:
        messages = existing_messages
    if system_message:
        messages.insert(0,{"role":"system","content":system_message})
    if prompt:
        messages.append({"role":"user","content":prompt})
    elif len(messages) == 0:
       return None

    kwargs['messages'] = messages
    
    if json_output:
        kwargs['response_format'] = {"type": "json_object"}

    if functions:
        response = call_openai_chat_with_model_fallbacks(**kwargs)
        try:
            function_args_str = response.choices[0].message.function_call.arguments
            function_args = json.loads(function_args_str)
            return {'args':function_args}
        except: 
            print("Error parsing JSON output from OpenAI:")
            print(response)
    else:
        response = call_openai_chat_with_model_fallbacks(**kwargs)

        response_text = response.choices[0].message.content
        
        if json_output:
            try:
                json_output = json.loads(response_text)
                return {'json': json_output}
            except: 
                print("Error parsing JSON output from OpenAI:")
                print(response_text)
        else:
            return {'text': response_text}