from typing import List


def init_function_calling(labels: List[str], tool_choices: str, open_ai_api_call: bool):
    if open_ai_api_call:
        call = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "classification",
                        "description": "Use this function to return your response to the user question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": labels,
                                    "description": "The permitted categories to response to the question.",
                                },
                            },
                            "required": ["category"],
                        },
                    },
                }
            ],
            "tool_choice": tool_choices,
        }
    else:
        # Anthropic call
        call = {
            "tools": [
                {
                    "name": "classification",
                    "description": "Use this function to return your response to the user question.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": labels,
                                "description": "The permitted categories to response to the question.",
                            },
                        },
                        "required": ["category"],
                    },
                },
            ],
            "tool_choice": tool_choices,
        }
    return call
