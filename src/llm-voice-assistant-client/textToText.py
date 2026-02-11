from openai import OpenAI
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize
import nltk
from datetime import datetime
import os
import json
import requests
from mcp_client import MCPClient
import asyncio
import threading
import json
import tomllib

with open("config.toml", "rb") as f:
    data = tomllib.load(f)

class textToText:
    def __init__(self, llm_model, base_url, llm_api_key):
        # Supported languages by TTS
        self.availableTools = []
        languagesPiperTTS = [Language.ARABIC, Language.CATALAN, Language.CZECH, Language.WELSH, Language.DANISH, Language.GERMAN, Language.GREEK, Language.ENGLISH, Language.SPANISH, Language.PERSIAN, Language.FINNISH, Language.FRENCH, Language.HUNGARIAN, Language.ICELANDIC, Language.ITALIAN, Language.GEORGIAN, Language.KAZAKH, Language.DUTCH, Language.POLISH, Language.PORTUGUESE, Language.ROMANIAN, Language.RUSSIAN, Language.SLOVAK, Language.SERBIAN, Language.SWEDISH, Language.SWAHILI, Language.TURKISH, Language.UKRAINIAN, Language.VIETNAMESE, Language.CHINESE]
        languagesKokoroTTS = [Language.CHINESE, Language.ENGLISH, Language.FRENCH, Language.HINDI, Language.ITALIAN, Language.JAPANESE, Language.PORTUGUESE, Language.SPANISH]
        languages = list(set(languagesPiperTTS + languagesKokoroTTS))

        self.detector = LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().with_minimum_relative_distance(0.3).build() # Eager load language detection models
        
        self.llm_model = llm_model
        
        # If Ollama is installed automatically download LLM.
        ollama_installed = self.ollamaDownloadModel(base_url, llm_model)

        if ollama_installed == True:
            self.client = OpenAI(
                base_url = base_url,
                api_key = llm_api_key, # Required even if unused.
            )

        else:
            self.client = OpenAI(
                base_url = base_url,
                api_key = llm_api_key, # Required even if unused.
            )

        nltk.download('punkt_tab')

        # Without this the first response from the LLM is very slow. Getting a response first speeds it up but only by running this complete function first fully reduces the delay. I am not sure as why.
        messages = self.chatWithHistory({"language": "en", "transcript": "Copy me verbatim 'The quick brown fox jumps over the lazy dog.'"}, "chat-history/workAround.json", "Follow the user's instructions.")
        for message in messages:
            message = 0
        os.remove("chat-history/workAround.json")

    async def connnectToMCP(self):
        self.mcp = MCPClient()
        await self.mcp.connect_to_server(r"mcpserver.py")
        self.availableTools = self.mcp.tools

    def setupMCPSync(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever,
            daemon=True
        )
        self.thread.start()

        fut = asyncio.run_coroutine_threadsafe(
            self.connnectToMCP(),
            self.loop
        )
        fut.result()
   

    def ollamaDownloadModel(self, base_url, llm_model):
        try:
            ollama_url = f"{base_url}/api/show"

            response = requests.post(ollama_url, data=json.dumps({"model": llm_model}), stream=False)

            if response.status_code == 404:
                ollama_url = f"{base_url}/api/pull"

                response = requests.post(ollama_url, data=json.dumps({"model": llm_model}), stream=True)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunk_data = json.loads(chunk.decode('utf-8'))
                        completed = chunk_data.get("completed")
                        total = chunk_data.get("total")
                        if completed is not None:
                            percentage = round((completed / total) * 100, 2)
                            print(f"\x1b[2KDownloading {llm_model}: {percentage}%", end="\r")

            return True

        except requests.exceptions.RequestException:
            print("Ollama isn't installed so model can't be downloaded automatically")

            return False

    def langDetect(self, text, transcription):
        language = self.detector.detect_language_of(text)
        
        if language != None:
            print(language, language.iso_code_639_1.name)
            language = language.iso_code_639_1.name
        
        language = transcription["language"] if language == None else language
        return language.lower()
    
    def callFunctionThread(self, tool_name, tool_args):
        future = asyncio.run_coroutine_threadsafe(
            self.mcp.session.call_tool(tool_name, tool_args),
            self.loop
        )
        try:
            return future.result()
        except Exception as e:
            print(f"Tool {tool_name} did not return in time: {e}")
            return None

    def chatWithHistory(self, transcription, chatHistoryFile, systemPrompt=""):
        # Add date and time into to system prompt
        systemPrompt = systemPrompt.format(
            date=datetime.now().strftime("%Y-%m-%d (%A)"),
            time=datetime.now().strftime("%I:%M %p")
        )
        
        # Try to load the chat history from a file
        if os.path.exists(chatHistoryFile):
            with open(chatHistoryFile, "r") as file:
                json_string = file.read()

            chatHistory = json.loads(json_string)

        # If the file doesn't exist, create it with an initial system prompt
        else:
            json_string = json.dumps(
                [
                    {
                    'role': 'system',
                    'content': systemPrompt,
                    }
                ]
            )

            with open(chatHistoryFile, 'w') as file:
                file.write(json_string)

            with open(chatHistoryFile, "r") as file:
                json_string = file.read()

            chatHistory = json.loads(json_string)

        # Append the user's message to the chat history
        chatHistory.append(
            {
            'role': 'user',
            'content': f'<context>Current date: {datetime.now().strftime("%Y-%m-%d (%A)")}\nCurrent time: {datetime.now().strftime("%I:%M %p")}]</context>\n\n {transcription["transcript"]}',
            }
        )

        stream = self.client.chat.completions.create(
            model = self.llm_model, 
            messages = chatHistory,
            stream = True,
            tools = self.availableTools
        )

        sentence = 1
        response = ""
        needed_tools = []
        sentences = []
        for chunk in stream:
            if chunk.choices[0].delta.content != None:
                part = chunk.choices[0].delta.content

                response = response + part

                # Chunk the response into sentences and yield each one as it is completed
                sentences = sent_tokenize(response)
                while sentence < len(sentences):
                    language = self.langDetect(sentences[sentence - 1], transcription)
                    yield { "sentence": sentences[sentence - 1], "language": language }
                    sentence += 1
                else:
                    yield { "sentence": "" }

            if chunk.choices[0].delta.tool_calls != None:
                #The second chunk has the name of the function, each later chunk has a part of the args. Get the name, and concatinate the args into one object from the separate chunks and
                # pass that as a param to a tool call function.
                #Let it handle multiple tool call requests at once.
                tool_calls = chunk.choices[0].delta.tool_calls

                # You should index by tool_call.id instead for multiple mcp tools at once.
                for i, tool_call in enumerate(tool_calls):
                    if tool_call.function.name is not None:
                        needed_tools.append({"name": tool_call.function.name, "arguments_buffer": ""})
                    needed_tools[i]["arguments_buffer"] += tool_call.function.arguments

        # Parses arg parts into one arg then calls tool with args.
        for tool in needed_tools:
            if tool["arguments_buffer"]:
                tool["arguments"] = json.loads(tool["arguments_buffer"])
            else:
                tool["arguments"] = {}
            del tool["arguments_buffer"]

            try:
                tool_response = self.callFunctionThread(tool["name"], tool["arguments"])
                print("Tool response: ", tool_response)
                tool_response_text = tool_response.structuredContent['result']

                chatHistory.append({
                    "role": "tool",
                    "tool_name": tool["name"],
                    "arguments": tool["arguments"],
                    "content": tool_response_text
                })

                chatHistory.append({
                    "role": "system",
                    "content": (
                        "Explain the tool result in ONE concise sentence. Do not add extra commentary."
                    )
                })

                second_stream = self.client.chat.completions.create(
                    model = self.llm_model, 
                    messages = chatHistory,
                    stream = True,
                )

                sentence = 1
                # Stops first llm response that comes when llm gives tool arguments from being spoken.
                response = ""
                sentences = []
                
                for chunk in second_stream:
                    if chunk.choices[0].delta.content != None:
                        part = chunk.choices[0].delta.content

                        response = response + part

                        # Chunk the response into sentences and yield each one as it is completed
                        sentences = sent_tokenize(response)
                        while sentence < len(sentences):
                            language = self.langDetect(sentences[sentence - 1], transcription)
                            yield { "sentence": sentences[sentence - 1], "language": language }
                            sentence += 1
                        else:
                            yield { "sentence": "" }

                # Chunk the response into sentences and yield each one as it is completed

                # sentences = sent_tokenize(tool_response_text)
                # while sentence < len(sentences):
                #     language = self.langDetect(sentences[sentence - 1], transcription)
                #     yield { "sentence": sentences[sentence - 1], "language": language }
                #     sentence += 1
                # else:
                #     yield { "sentence": "" }

                # Add tool result to chat history
                

            except Exception as e:
                print(f"Error calling tool or gettings response of: {tool['name']}: {e}")

        language = self.langDetect(sentences[sentence - 1], transcription)
        yield { "sentence": sentences[sentence - 1], "language": language }

        if response:
            chatHistory.append(
                {
                'role': 'assistant',
                'content': response,
                }
            )

        print(response, needed_tools)


        # Append the assistant's response to the chat history

        # Delete the info role so it doesn't hog the LLM's context window


        user_indexes = [i for i, msg in enumerate(chatHistory) if msg.get("role") == "user"]
        if len(user_indexes) >= data["max_history"]:
            del chatHistory[1 : user_indexes[-data["max_history"]]]

        # Write chat history to file
        json_string = json.dumps(chatHistory, indent=2)
        with open(chatHistoryFile, 'w') as file:
            file.write(json_string)