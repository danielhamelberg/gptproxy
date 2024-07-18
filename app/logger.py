# logger.py

import logging
import json
from datetime import datetime

class QueryLogger:
    def __init__(self, log_file_path):
        self.logger = logging.getLogger('query_logger')
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def log_query(self, request, response):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request': self._serialize_request(request),
            'response': self._serialize_response(response)
        }
        self.logger.info(json.dumps(log_entry))

    def _serialize_request(self, request):
        return {
            'model': request.model,
            'messages': [message.dict() for message in request.messages],
            'temperature': request.temperature,
            'top_p': request.top_p,
            'n': request.n,
            'stream': request.stream,
            'max_tokens': request.max_tokens,
            'presence_penalty': request.presence_penalty,
            'frequency_penalty': request.frequency_penalty
        }

    def _serialize_response(self, response):
        if isinstance(response, dict):
            return response
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return str(response)