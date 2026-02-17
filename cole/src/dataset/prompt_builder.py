import logging
from typing import List


class PromptBuilder:
    """Builder class for creating prompt strings with dynamic data."""

    def __init__(self):
        self.premise: List[str] = []
        self.end: List[str] = []
        self.data: List[str] = []
        self.data_only = False

    def add_data(self, data):
        self.data.append(data)
        return self

    def add_end(self, end):
        self.end.append(end)
        return self

    def set_data_only(self, data_only):
        self.data_only = data_only
        return self

    def add_premise(self, premise):
        self.premise.append(premise)
        return self

    def build(self):
        """Builds and returns the prompt as a string based on data, premise and end that were added to the builder."""
        if len(self.data) == 0:
            logging.warning(
                "This prompt did not contain any data, was that intentional ?"
            )

        data = "\n".join(self.data)
        if self.data_only:
            return data

        end = "".join(self.end)
        premise = "".join(self.premise)
        return f"{premise}\n{data}\n{end}"
