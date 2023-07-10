import huggingface_hub
import json
import os
import requests
import termcolor


DEBUG_PRINT_COLOR = "yellow"
LOCALHOST = "localhost"


class HuggingfaceAvailableFunctions:
    def __init__(self, huggingface_api_key: str, is_verbose: bool = False) -> None:
        print(termcolor.colored("HuggingfaceAvailableFunctions.__init__()", DEBUG_PRINT_COLOR))
        print()

        self.huggingface_tasks = requests.get("https://huggingface.co/api/tasks").json()
        self.huggingface_inference_client = huggingface_hub.InferenceClient(token=huggingface_api_key)
        self.is_verbose = is_verbose

        with open(os.path.join(os.path.dirname(__file__), "huggingface_available_functions.json"), "r") as f:
            self.available_functions_list = json.load(f)

        if self.is_verbose:
            print(termcolor.colored(f"{self.available_functions_list = }", DEBUG_PRINT_COLOR))
            print()

    def get_available_functions_list(self) -> list:
        print(termcolor.colored("HuggingfaceAvailableFunctions.get_available_functions_list()", DEBUG_PRINT_COLOR))
        print()

        return self.available_functions_list

    def call_function(self, function_name: str, function_arguments: dict) -> str:
        print(termcolor.colored("HuggingfaceAvailableFunctions.call_function()", DEBUG_PRINT_COLOR))
        print()

        function_to_call = getattr(self, function_name.replace("-", "_"))
        function_content = function_to_call(function_name, function_arguments)
        return function_content

    """
    Computer Vision
    """

    # https://huggingface.co/tasks/image-classification
    def image_classification(self, function_name: str, function_arguments: dict) -> str:
        function_content = \
            json.dumps(self.huggingface_inference_client.image_classification(function_arguments["data"]))
        return function_content

    # https://huggingface.co/tasks/object-detection
    def object_detection(self, function_name: str, function_arguments: dict) -> str:
        recommended_model = self.huggingface_tasks[function_name]["models"][0]["id"]
        function_content = self.huggingface_inference_client.post(
            data=function_arguments["data"],
            model=recommended_model
        ).text
        return function_content

    """
    Multimodal
    """

    # https://huggingface.co/tasks/document-question-answering
    def document_question_answering(self, function_name: str, function_arguments: dict) -> str:
        # TODO: Pass the user prompt to this function as an additional argument
        # TODO: so that the document-question-answering model can answer to the user prompt.
        pass

    # https://huggingface.co/tasks/image-to-text
    def image_to_text(self, function_name: str, function_arguments: dict) -> str:
        function_content = self.huggingface_inference_client.image_to_text(function_arguments["data"])
        return function_content

    # https://huggingface.co/tasks/visual-question-answering
    def visual_question_answering(self, function_name: str, function_arguments: dict) -> str:
        # TODO: Pass the user prompt to this function as an additional argument
        # TODO: so that the document-question-answering model can answer to the user prompt.
        pass
