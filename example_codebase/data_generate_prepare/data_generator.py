import json
import time
import openai
import requests
import sseclient
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class DataBot:
    def __init__(self):
        self.data_message_log = [
            {
                "role": "system",
                "content": """ 
			Prompt to generate data from gpt-4
			Desired output format:

			{
			"instruction": "",
			"input": "",
			"output": ""
			}

			Please make sure, you must generate one response at a time.
                """,
            }
        ]
        self.model = "gpt-4"
        self.bot_name = "DataBot"

    def filter_json(self, text):
        start_pos = text.find("{")
        end_pos = text.rfind("}") + 1
        json_string = text[
            start_pos:end_pos
        ]  # if doesnot find the json_string will be empty
        return json_string

    def streaming_response(self, message_log):
        responses = ""
        req_url = "https://api.openai.com/v1/chat/completions"
        req_headers = {
            "Accept": "text/event-stream",
            "Authorization": "Bearer " + openai.api_key,
        }
        req_data = {
            "model": self.model,
            "messages": message_log,
            "max_tokens": 100,
            "temperature": 0,
            "stop": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": True,
        }
        print(f"The model {self.model} successfully")
        request = requests.post(
            req_url, stream=True, headers=req_headers, json=req_data
        )
        client = sseclient.SSEClient(request)

        for event in client.events():
            if event.data != "[DONE]":
                if "content" in json.loads(event.data)["choices"][0]["delta"]:
                    print(
                        json.loads(event.data)["choices"][0]["delta"]["content"],
                        end="",
                        flush=True,
                    )
                    responses += json.loads(event.data)["choices"][0]["delta"][
                        "content"
                    ]
        print()
        return responses

    def send_message(self, message_log):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=message_log,
                temperature=0.7,
                max_tokens=1024,
                n=1,
                stop=None,
                timeout=20,
                frequency_penalty=0,
                presence_penalty=0,
            )
            print(f"\n The model {self.model} loaded successfully")
            for choice in response.choices:
                if "text" in choice:
                    return choice.text
            return response["choices"][0]["message"]["content"]

        except openai.error.RateLimitError as e:
            print("Rate limit exceeded. Waiting for 30 seconds.")
            time.sleep(30)
        except Exception as e:
            print("Error: ", e)
            return None

    def generate_responses(self, count):
        responses = []
        message_log = self.data_message_log.copy()

        with ThreadPoolExecutor() as executor:
            future_to_response = {
                executor.submit(self.send_message, message_log): message_log
                for _ in range(count)
            }
            with tqdm(total=count, desc="Generating Responses") as pbar:
                for future in concurrent.futures.as_completed(future_to_response):
                    message_log = future_to_response[future]
                    try:
                        response = future.result()
                        if response is not None:
                            try:
                                result = json.loads(response)
                                responses.append(result)
                                # message_log.append({
                                #     "role": "system",
                                #     "content": response
                                # })
                            except TypeError as te:
                                print("TypeError: Unable to parse JSON Response")
                    except Exception as e:
                        print("Error: ", e)
                    pbar.update(1)

        with open("dataset/interview_dataset.json", "w") as json_file:
            json.dump(responses, json_file, indent=4)

        return responses

if __name__ == '__main__':
    max_data = 7000
    data_bot = DataBot()
    start_time = perf_counter()
    responses = data_bot.generate_responses(max_data)
    print(f"The {max_data} data generated in {perf_counter() - start_time:.2f} seconds")
    # print(responses)
