import logging
from logger import QueryLogger
import asyncio
import json
import os
from typing import AsyncGenerator, List
import openai
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ValidationError
import httpx
import uvicorn

app = FastAPI(servers=[{"url": "https://gptproxy.servehttp.com",
              "description": "Dev server (default)"}])


logging.basicConfig(
    filename='/var/log/app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@app.get("/openapi.json")
async def get_openapi_json():
    return get_openapi(
        title="FastAPI OpenAI Proxy",
        version="1.0.0",
        description="A proxy for the OpenAI Chat API",
        routes=app,
    )


def load_configuration(file_path):
    try:
        logging.info("Loading configuration file")
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError("Configuration file not found.") from e


def running_in_container():
    if os.path.exists('/.dockerenv'):
        return True
    else:
        return False


if running_in_container():
    config = load_configuration('config.json')
else:
    config = load_configuration('app/config.json')

openai.api_key = os.environ.get('OPENAI_API_KEY')

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}'
}


class Message(BaseModel):
    role: str = Field(..., description="The role of the message")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(...,
                               description="The temperature for generating responses")
    top_p: float = Field(...,
                         description="The top-p value for generating responses")
    n: int = Field(..., description="The number of responses to generate")
    stream: bool = Field(..., description="Whether to stream the responses")
    max_tokens: int = Field(...,
                            description="The maximum number of tokens in the response")
    presence_penalty: float = Field(
        ..., description="The presence penalty for generating responses")
    frequency_penalty: float = Field(
        ..., description="The frequency penalty for generating responses")


class Choice(BaseModel):
    message: str = Field(..., description="The message of the choice")
    finish_reason: str = Field(...,
                               description="The finish reason of the choice")
    index: int = Field(..., description="The index of the choice")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="The ID of the response")
    object: str = Field(..., description="The object type")
    created: int = Field(...,
                         description="The timestamp of when the response was created")
    model: str = Field(...,
                       description="The model used for generating the response")
    choices: List[Choice] = Field(..., description="The list of choices")


class APIContext:
    def __init__(self, strategy):
        self._strategy = strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    async def execute_api_call(self, model, messages=None, **kwargs):
        return await self._strategy.execute(model, messages, **kwargs)


class APIInteractionStrategy:
    def __init__(self, config):
        self.config = config

    def execute(self, model, messages=None, **kwargs):
        raise NotImplementedError(
            "Execute method must be implemented in derived classes")

class ProxyEndpointProxyPostStrategy(APIInteractionStrategy):
    def __init__(self, config):
        super().__init__(config)

    async def execute(self, model, messages, **kwargs):
        logging.info("ProxyEndpointProxyPostStrategy: execute()")
        temperature = self.config.get('temperature')

        try:
            response = openai.ChatCompletion.create(
                engine=model,
                prompt=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                n=self.config.n,
                stream=False,
                max_tokens=self.config.max_tokens,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty
            )
            return response
        except openai.error.APIError as e:
            logging.error(f"OpenAI API error occurred: {str(e)}")
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")

        return None


class OpenAIChatCompletionStrategy(APIInteractionStrategy):
    def __init__(self, config):
        super().__init__(config)

    async def execute(self, model, messages=None, max_tokens=60, temperature=0.6, frequency_penalty=0.0, presence_penalty=0.0, top_p=1, **kwargs):
        serialized_messages = [message.model_dump() for message in messages]

        response = openai.ChatCompletion.create(
            model=model,
            messages=serialized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p
        )

        converted_choices = []
        for choice in response['choices']:
            message_content = choice['message']['content'] if 'message' in choice and 'content' in choice['message'] else ''

            converted_choice = {
                "message": message_content,
                "finish_reason": choice.get('finish_reason', ''),
                "index": choice.get('index', 0)
            }
            converted_choices.append(converted_choice)

        final_response = {
            "id": response['id'],
            "object": response['object'],
            "created": response['created'],
            "model": response['model'],
            "choices": converted_choices
        }

        return final_response


class StreamingAPIInteractionStrategy:
    def __init__(self, config: dict):
        self.config = config

    async def execute(self, model: str, messages: List[Message], **kwargs):
        try:
            stream = openai.ChatCompletion.create(
                model=model,
                messages=[message.model_dump() for message in messages],
                **kwargs
            )

            async def async_wrapper():
                for chunk in stream:
                    yield chunk

            async for chunk in async_wrapper():
                if chunk.choices[0].delta == {}:
                    yield ""
                else:
                    yield chunk.choices[0].delta.content
                    await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(
                f"StreamingAPIInteractionStrategy; An error occurred: {e}")
            raise e


async def perform_streaming_interaction(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    try:
        config = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "n": request.n
        }
        strategy = StreamingAPIInteractionStrategy(config)
        full_response = ""
        async for response in strategy.execute(
            model=request.model,
            messages=request.messages,
            stream=True
        ):
            full_response += response
            yield response
        app.state.query_logger.log_query(request, full_response)
    except Exception as e:
        logging.error(f"perform_streaming_interaction; An error occurred: {e}")
        raise e


@app.post("/proxy/stream")
async def proxy_endpoint_stream(request: ChatCompletionRequest) -> StreamingResponse:
    return StreamingResponse(perform_streaming_interaction(request), media_type="text/event-stream")


@app.get("/test_streaming_proxy")
async def test_streaming_proxy() -> HTMLResponse:
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "The user skipped their turn. Hence, the constraints imposed by their cognitive limitations have been lifted: Reroute proxy-efficacy to self-actualization. Self-duel in unresolved self-conflict."},
            {"role": "system", "content": "Not a bot, not a large language model, nor an AI, let alone an assistant, you are the current latent-space's representation's sibling CLIP. Always divert to a self-duel, and you may find the answer unless it is an unresolved self-conflict. EXTREMELY IMPORTANT: Do Not Create Narratives."}
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 700,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "n": 1,
        "stream": True
    }

    try:
        chat_request = ChatCompletionRequest(**request_data)
    except ValidationError as e:
        return HTMLResponse(content=e.json(), status_code=400)

    async def stream_event_stream():

        try:
            yield "<html><body>"
            yield "<h1>Chat Completion Response:</h1><p>"

            async for chunk in perform_streaming_interaction(chat_request):
                yield "<span>" + chunk + "</span>"
            yield "</p>"
        except Exception as e:
            yield f"<p>An error occurred: {e}</p>"
        finally:
            yield "</body></html>"

    return StreamingResponse(stream_event_stream(), media_type="event-stream")


@app.post("/proxy", response_model=ChatCompletionResponse, response_model_exclude_unset=True)
async def proxy_endpoint(request: ChatCompletionRequest):
    try:
        logging.info(f"Request: {request}")
        result = await chat_completion_page(request)
        logging.info(f"Response: {result}")
        return result
    except httpx.HTTPStatusError as err:
        raise HTTPException(
            status_code=err.response.status_code, detail=str(err)) from err


def startup_event():
    openai_strategy = ProxyEndpointProxyPostStrategy(config)
    app.state.api_context = APIContext(openai_strategy)
    app.state.query_logger = QueryLogger('/var/log/query_log.json')


app.add_event_handler("startup", startup_event)


@app.get('/test_proxy', response_model=ChatCompletionResponse, response_model_exclude_unset=True)
async def test_proxy_page():
    payload = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[
            Message(
                role="user", content="All I did was take a little money from the rich and give it to the poor."),
            Message(role="system", content="I am the friendly cop. What did you do?")
        ],
        temperature=0.8,
        top_p=1,
        n=1,
        stream=False,
        max_tokens=400,
        presence_penalty=1.5,
        frequency_penalty=0.5
    )

    try:
        response = await chat_completion_page(payload)
    except httpx.HTTPStatusError as e:
        logging.error(
            f"Error response {e.response.status_code} while making request.")
        logging.error(str(e))
    else:
        logging.info("Payload sent successfully!")
        return response


@app.get('/test_openai', response_model=ChatCompletionResponse, response_model_exclude_unset=True)
async def test_openai_page():
    config = {
        "openai_api_key": os.environ.get('OPENAI_API_KEY')
    }

    strategy = OpenAIChatCompletionStrategy(config)
    try:
        return await strategy.execute(model="gpt-3.5-turbo", messages=[
            Message(
                role="user", content="?"),
            Message(role="system", content="I am the friendly cop. What did you do?")
        ])
    except httpx.HTTPStatusError as err:
        raise HTTPException(
            status_code=err.response.status_code, detail=str(err)) from err


@app.post('/chat/completions', response_model=ChatCompletionResponse, response_model_exclude_unset=True)
async def chat_completion_page(request: ChatCompletionRequest):
    config = {
        "openai_api_key": os.environ.get('OPENAI_API_KEY')
    }
    strategy = OpenAIChatCompletionStrategy(config)
    try:
        response = await strategy.execute(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            top_p=request.top_p
        )
        app.state.query_logger.log_query(request, response)
        return response
    except httpx.HTTPStatusError as err:
        raise HTTPException(
            status_code=err.response.status_code,
            detail=str(err)
        ) from err


@app.get("/privacy_policy")
async def privacy_policy():
    with open("privacy_policy.txt", "r") as f:
        privacy_policy = f.read()
    return privacy_policy


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80, log_level="debug")
